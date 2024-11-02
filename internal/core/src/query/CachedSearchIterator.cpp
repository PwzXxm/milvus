// Copyright (C) 2019-2024 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include "query/CachedSearchIterator.h"
#include <algorithm>

namespace milvus::query {

CachedSearchIterator::CachedSearchIterator(
    const milvus::index::VectorIndex& index,
    const knowhere::DataSetPtr& dataset,
    const SearchInfo& search_info,
    const BitsetView& bitset) {
    nq_ = dataset->GetRows();
    result_pool_.resize(nq_);
    Init(search_info);

    const auto search_json = index.PrepareSearchParams(search_info);
    auto expected_iterators =
        index.VectorIterators(dataset, search_json, bitset);
    if (expected_iterators.has_value()) {
        iterators_ = std::move(expected_iterators.value());
    } else {
        PanicInfo(ErrorCode::UnexpectedError,
                  "Failed to create iterators from index");
    }
}

CachedSearchIterator::CachedSearchIterator(
    const dataset::SearchDataset& dataset,
    const void* vec_data,
    const int64_t row_count,
    const SearchInfo& search_info,
    const BitsetView& bitset,
    const milvus::DataType& data_type) {
    nq_ = dataset.num_queries;
    result_pool_.resize(nq_);
    Init(search_info);

    auto expected_iterators = GetBruteForceSearchIterators(
        dataset, vec_data, row_count, search_info, bitset, data_type);
    if (expected_iterators.has_value()) {
        iterators_ = std::move(expected_iterators.value());
    } else {
        PanicInfo(ErrorCode::UnexpectedError,
                  "Failed to create iterators from index");
    }
}

void CachedSearchIterator::InitializeIterators(
    const dataset::SearchDataset& dataset,
    const SearchInfo& search_info,
    const BitsetView& base_bitset,
    const milvus::DataType& data_type,
    const GetChunkDataFunc& get_chunk_data,
    const GetBitsetViewFunc& get_chunk_bitset) {
    
    int64_t offset = 0;
    for (int64_t chunk_id = 0; chunk_id < num_chunks_; ++chunk_id) {
        seg_start_offset_for_chunk_[chunk_id] = offset;

        auto [chunk_data, chunk_size] = get_chunk_data(chunk_id);
        auto [sub_view, _] = get_chunk_bitset(base_bitset, offset, chunk_size);

        auto expected_iterators = GetBruteForceSearchIterators(dataset,
                                                             chunk_data,
                                                             chunk_size,
                                                             search_info,
                                                             sub_view,
                                                             data_type);
        if (expected_iterators.has_value()) {
            auto& chunk_iterators = expected_iterators.value();
            iterators_.insert(iterators_.end(),
                            std::make_move_iterator(chunk_iterators.begin()),
                            std::make_move_iterator(chunk_iterators.end()));
        } else {
            PanicInfo(ErrorCode::UnexpectedError,
                     "Failed to create iterators from index");
        }
        offset += chunk_size;
    }
}

CachedSearchIterator::CachedSearchIterator(
    const dataset::SearchDataset& dataset,
    const segcore::VectorBase* vec_data,
    const int64_t row_count,
    const SearchInfo& search_info,
    const BitsetView& bitset,
    const milvus::DataType& data_type) {
    const int64_t vec_size_per_chunk = vec_data->get_size_per_chunk();
    num_chunks_ = upper_div(row_count, vec_size_per_chunk);
    nq_ = dataset.num_queries;
    result_pool_.resize(nq_);
    Init(search_info);

    iterators_.reserve(nq_ * num_chunks_);
    seg_start_offset_for_chunk_.resize(num_chunks_);

    InitializeIterators(
        dataset, 
        search_info,
        bitset,
        data_type,
        [&vec_data, vec_size_per_chunk, row_count](int64_t chunk_id) -> std::pair<const void*, int64_t> {
            const auto chunk_data = vec_data->get_chunk_data(chunk_id);
            int64_t chunk_size = std::min(vec_size_per_chunk, 
                                        row_count - chunk_id * vec_size_per_chunk);
            return {chunk_data, chunk_size};
        },
        [](const BitsetView& bitset, int64_t offset, int64_t chunk_size) -> BitsetViewWithMem {
            return {bitset.subview(offset, chunk_size), {}};
        }
    );
}

CachedSearchIterator::CachedSearchIterator(
    const std::shared_ptr<ChunkedColumnBase>& column,
    const dataset::SearchDataset& dataset,
    const SearchInfo& search_info,
    const BitsetView& bitset,
    const milvus::DataType& data_type,
    const GetBitsetViewFunc& get_bitset_view_with_mem) {
    num_chunks_ = column->num_chunks();
    nq_ = dataset.num_queries;
    result_pool_.resize(nq_);
    Init(search_info);

    iterators_.reserve(nq_ * num_chunks_);
    seg_start_offset_for_chunk_.resize(num_chunks_);

    InitializeIterators(
        dataset,
        search_info,
        bitset,
        data_type,
        [&column](int64_t chunk_id) {
            const char* chunk_data = column->Data(chunk_id);
            int64_t chunk_size = column->chunk_row_nums(chunk_id);
            return std::make_pair(static_cast<const void*>(chunk_data), chunk_size);
        },
        get_bitset_view_with_mem
    );
}

void
CachedSearchIterator::NextBatch(const SearchInfo& search_info,
                                SearchResult& search_result) {
    if (iterators_.empty()) {
        return;
    }

    if (iterators_.size() != nq_ * num_chunks_) {
        PanicInfo(ErrorCode::UnexpectedError,
                  "Iterator size mismatch, expect %d, but got %d",
                  nq_ * num_chunks_,
                  iterators_.size());
    }

    ValidateSearchInfo(search_info);

    if (result_pool_.size() != nq_) {
        PanicInfo(ErrorCode::UnexpectedError,
                  "Result pool size mismatch, expect %d, but got %d",
                  nq_,
                  result_pool_.size());
    }

    // TODO: refactor this logic
    auto last_bound = search_info.iterator_v2_info_.value().last_bound;
    if (last_bound.has_value()) {
        last_bound = last_bound.value() * sign_;
    }
    for (size_t query_idx = 0; query_idx < nq_; ++query_idx) {
        auto& rst_queue = result_pool_[query_idx];
        while (!rst_queue.empty() && last_bound.has_value() &&
               rst_queue.top().first <= last_bound.value()) {
            rst_queue.pop();
        }
        while (rst_queue.size() < batch_size_ * 2) {
            auto rst_size = rst_queue.size();
            RefillIteratorResultPool(query_idx, search_info);
            if (rst_queue.size() == rst_size) {
                break;
            }
        }
    }

    search_result.total_nq_ = nq_;
    search_result.unity_topK_ = batch_size_;
    search_result.seg_offsets_.resize(nq_ * batch_size_);
    search_result.distances_.resize(nq_ * batch_size_);

    WriteSearchResultFromResultPool(search_result);
}

void
CachedSearchIterator::WriteSearchResultFromResultPool(SearchResult& search_result) {
    for (size_t query_idx = 0; query_idx < nq_; ++query_idx) {
        auto rst_dist = search_result.distances_.begin() + query_idx * batch_size_;
        auto rst_seg_offset = search_result.seg_offsets_.begin() + query_idx * batch_size_;

        // TODO: optimize this, not copy
        // but do not need to pop, pop based on last_bound later
        auto rst_queue = result_pool_[query_idx];
        for (size_t i = 0; i < batch_size_; ++i) {
            if (rst_queue.empty()) {
                rst_dist[i] = 1.0f / 0.0f;
                rst_seg_offset[i] = -1;
                continue;
            }

            auto dis_id_pair = rst_queue.top();
            rst_dist[i] = dis_id_pair.first * sign_;
            rst_seg_offset[i] = dis_id_pair.second;
            rst_queue.pop();
        }
    }
}

std::vector<std::vector<CachedSearchIterator::DisIdPair>>
CachedSearchIterator::IteratorsSearch(const SearchInfo& search_info) {
    std::vector<std::vector<DisIdPair>> rst_vec;
    rst_vec.reserve(nq_);
    for (size_t query_idx = 0; query_idx < nq_; ++query_idx) {
        rst_vec.push_back(GetBatchedNextResults(query_idx, search_info));
    }
    return rst_vec;
}

void
CachedSearchIterator::ValidateSearchInfo(const SearchInfo& search_info) {
    if (!search_info.iterator_v2_info_.has_value()) {
        PanicInfo(ErrorCode::UnexpectedError,
                  "Iterator v2 SearchInfo is not set");
    }

    auto iterator_v2_info = search_info.iterator_v2_info_.value();
    if (iterator_v2_info.batch_size != batch_size_) {
        PanicInfo(ErrorCode::UnexpectedError,
                  "Batch size mismatch, expect %d, but got %d",
                  batch_size_,
                  iterator_v2_info.batch_size);
    }
}

void
CachedSearchIterator::RefillIteratorResultPool(const size_t query_idx, const SearchInfo& search_info) {
    auto rst = GetBatchedNextResults(query_idx, search_info);
    // TODO: optimize this
    for (const auto& dis_id_pair : rst) {
        result_pool_[query_idx].push(dis_id_pair);
    }
}

CachedSearchIterator::DisIdPair
CachedSearchIterator::ConvertIteratorResult(
    const std::pair<int64_t, float>& iter_rst, const size_t chunk_idx) {
    DisIdPair rst;
    rst.first = iter_rst.second * sign_;
    rst.second = iter_rst.first + seg_start_offset_for_chunk_[chunk_idx];
    return rst;
}

std::optional<CachedSearchIterator::DisIdPair>
CachedSearchIterator::GetNextValidResult(
    const size_t iterator_idx,
    const std::optional<double>& last_bound,
    const size_t chunk_id) {
    auto& iterator = iterators_[iterator_idx];
    while (iterator->HasNext()) {
        auto result = ConvertIteratorResult(iterator->Next(), chunk_id);
        if (!last_bound.has_value() || result.first > last_bound.value()) {
            return result;
        }
    }
    return std::nullopt;
}

// TODO: Optimize this method
void
CachedSearchIterator::MergeChunksResults(
    size_t query_idx,
    const std::optional<double>& last_bound,
    std::vector<DisIdPair>& rst) {


    auto cmp = [](const auto& lhs, const auto& rhs) {
        return lhs.second.first > rhs.second.first;
    };
    std::priority_queue<std::pair<size_t, DisIdPair>,
                        std::vector<std::pair<size_t, DisIdPair>>,
                        decltype(cmp)>
        heap(cmp);

    for (size_t chunk_id = 0; chunk_id < num_chunks_; ++chunk_id) {
        const size_t iterator_idx = query_idx + chunk_id * nq_;
        if (auto next_result = GetNextValidResult(iterator_idx, last_bound, chunk_id)) {
            heap.emplace(iterator_idx, *next_result);
        }
    }

    while (!heap.empty() && rst.size() < batch_size_) {
        const auto [iterator_idx, cur_rst] = heap.top();
        heap.pop();
        rst.push_back(cur_rst);
        if (auto next_result = GetNextValidResult(
                iterator_idx, last_bound, iterator_idx / nq_)) {
            heap.emplace(iterator_idx, *next_result);
        }
    }
}

std::vector<CachedSearchIterator::DisIdPair>
CachedSearchIterator::GetBatchedNextResults(size_t query_idx,
                                            const SearchInfo& search_info) {
    auto last_bound = search_info.iterator_v2_info_.value().last_bound;
    if (last_bound.has_value()) {
        last_bound = last_bound.value() * sign_;
    }

    std::vector<DisIdPair> rst;
    rst.reserve(batch_size_);

    if (num_chunks_ == 1) {
        auto& iterator = iterators_[query_idx];
        while (iterator->HasNext() && rst.size() < batch_size_) {
            auto result = ConvertIteratorResult(iterator->Next());
            if (!last_bound.has_value() || result.first > last_bound.value()) {
                rst.emplace_back(result);
            }
        }
    } else {
        MergeChunksResults(query_idx, last_bound, rst);
    }
    // std::sort(rst.begin(), rst.end());
    // if (sign_ == -1) {
    //     std::for_each(rst.begin(), rst.end(), [this](DisIdPair& x) {
    //         x.first = x.first * sign_;
    //     });
    // }
    // while (rst.size() < batch_size_) {
    //     rst.emplace_back(1.0f / 0.0f, -1);
    // }
    return rst;
}

void
CachedSearchIterator::WriteSingleQuerySearchResult(
    SearchResult& search_result,
    const size_t idx,
    std::vector<DisIdPair>& rst,
    const int64_t round_decimal) {
    const float multiplier = pow(10.0, round_decimal);

    std::transform(rst.begin(),
                   rst.end(),
                   search_result.distances_.begin() + idx * batch_size_,
                   [multiplier, round_decimal](DisIdPair& x) {
                       if (round_decimal != -1) {
                           x.first =
                               std::round(x.first * multiplier) / multiplier;
                       }
                       return x.first;
                   });

    std::transform(rst.begin(),
                   rst.end(),
                   search_result.seg_offsets_.begin() + idx * batch_size_,
                   [](const DisIdPair& x) { return x.second; });
}

void
CachedSearchIterator::Init(const SearchInfo& search_info) {
    if (search_info.iterator_v2_info_.has_value()) {
        auto iterator_v2_info = search_info.iterator_v2_info_.value();
        batch_size_ = iterator_v2_info.batch_size;
    }

    if (PositivelyRelated(search_info.metric_type_)) {
        sign_ = -1;
    } else {
        sign_ = 1;
    }
}

}  // namespace milvus::query
