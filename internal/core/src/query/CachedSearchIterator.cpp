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
    const auto search_json = index.PrepareSearchParams(search_info);
    auto expected_iterators =
        index.VectorIterators(dataset, search_json, bitset);
    if (expected_iterators.has_value()) {
        iterators_ = std::move(expected_iterators.value());
    } else {
        PanicInfo(ErrorCode::UnexpectedError,
                  "Failed to create iterators from index");
    }
    nq_ = dataset->GetRows();
    Init(search_info);
}

CachedSearchIterator::CachedSearchIterator(
    const dataset::SearchDataset& dataset,
    const void* vec_data,
    const int64_t row_count,
    const SearchInfo& search_info,
    const BitsetView& bitset,
    const milvus::DataType& data_type) {
    auto expected_iterators = GetBruteForceSearchIterators(
        dataset, vec_data, row_count, search_info, bitset, data_type);
    if (expected_iterators.has_value()) {
        iterators_ = std::move(expected_iterators.value());
    } else {
        PanicInfo(ErrorCode::UnexpectedError,
                  "Failed to create iterators from index");
    }
    nq_ = dataset.num_queries;
    Init(search_info);
}

CachedSearchIterator::CachedSearchIterator(
    const dataset::SearchDataset& dataset,
    const segcore::VectorBase* vec_data,
    const int64_t row_count,
    const SearchInfo& search_info,
    const BitsetView& bitset,
    const milvus::DataType& data_type) {
    const int64_t max_size_per_chunk = vec_data->get_size_per_chunk();
    const int64_t max_chunk = upper_div(row_count, max_size_per_chunk);

    iterators_.reserve(max_chunk);
    for (int64_t chunk_id = 0; chunk_id < max_chunk; ++chunk_id) {
        const auto chunk_data = vec_data->get_chunk_data(chunk_id);
        auto element_begin = chunk_id * max_size_per_chunk;
        auto element_end =
            std::min(row_count, (chunk_id + 1) * max_size_per_chunk);
        auto cur_chunk_size = element_end - element_begin;

        // free bitset view here will not cause memory leak, because it is used
        // only during construction of BF iterator
        BitsetView sub_view = bitset.subview(element_begin, cur_chunk_size);
        auto expected_iterators = GetBruteForceSearchIterators(dataset,
                                                               chunk_data,
                                                               cur_chunk_size,
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
    }
    nq_ = dataset.num_queries;
    chunk_size_ = max_size_per_chunk;
    Init(search_info);
}

void
CachedSearchIterator::NextBatch(const SearchInfo& search_info,
                                SearchResult& search_result) {
    if (iterators_.empty()) {
        return;
    }

    ValidateSearchInfo(search_info);

    search_result.total_nq_ = nq_;
    search_result.unity_topK_ = batch_size_;
    search_result.seg_offsets_.resize(nq_ * batch_size_);
    search_result.distances_.resize(nq_ * batch_size_);
    IteratorsSearch(search_info, search_result);
}

void
CachedSearchIterator::IteratorsSearch(const SearchInfo& search_info,
                                      SearchResult& search_result) {
    for (size_t idx = 0; idx < iterators_.size(); idx += chunk_size_) {
        const size_t iter_size = std::min(chunk_size_, iterators_.size() - idx);
        auto rst = GetBatchedNextResults(idx, iter_size, search_info);
        WriteSingleQuerySearchResult(
            search_result, idx, rst, search_info.round_decimal_);
    }
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
CachedSearchIterator::RefillIteratorResultPool() {
    // Implementation...
}

void
CachedSearchIterator::MergeIteratorResults(size_t iter_idx,
                                            size_t iter_size,
                                            const std::optional<double>& last_bound,
                                            std::vector<DisIdPair>& rst) {
    using IdxResultPair = std::pair<size_t, DisIdPair>;

    auto cmp = [](const IdxResultPair& lhs, const IdxResultPair& rhs) {
        return lhs.second.first > rhs.second.first;
    };
    std::priority_queue<IdxResultPair,
                        std::vector<IdxResultPair>,
                        decltype(cmp)>
        pq(cmp);

    for (size_t i = 0; i < iter_size; ++i) {
        const size_t idx = iter_idx + i;
        auto& iterator = iterators_[idx];
        if (iterator->HasNext()) {
            auto result = iterator->Next();
            result.first *= sign_;
            pq.emplace(idx, result);
        }
    }

    while (!pq.empty() && rst.size() < batch_size_) {
        const auto& [idx, result] = pq.top();
        pq.pop();
        if (iterators_[idx]->HasNext()) {
            pq.emplace(idx, iterators_[idx]->Next());
        }
        if (last_bound.has_value() && result.first <= last_bound.value()) {
            continue;
        }
        rst.emplace_back(result);
    }
}

std::vector<CachedSearchIterator::DisIdPair>
CachedSearchIterator::GetBatchedNextResults(
    size_t iter_idx,
    size_t iter_size,
    const SearchInfo& search_info) {
    const auto last_bound = search_info.iterator_v2_info_.value().last_bound;

    std::vector<DisIdPair> rst;
    rst.reserve(batch_size_);

    if (iter_size == 1) {
        auto& iterator = iterators_[iter_idx];
        while (iterator->HasNext() && rst.size() < batch_size_) {
            auto result = iterator->Next();
            result.first *= sign_;
            if (last_bound.has_value() && result.first <= last_bound.value()) {
                continue;
            }
            rst.emplace_back(result);
        }
    } else if (iter_size > 1) {
        MergeIteratorResults(iter_idx, iter_size, last_bound, rst);
    } else {
        PanicInfo(ErrorCode::UnexpectedError, "Invalid iterator size");
    }
    std::sort(rst.begin(), rst.end());
    if (sign_ == -1) {
        std::for_each(rst.begin(), rst.end(), [this](DisIdPair& x) {
            x.first = x.first * sign_;
        });
    }
    while (rst.size() < batch_size_) {
        rst.emplace_back(1.0f / 0.0f, -1);
    }
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
                   [](const DisIdPair& x) {
                       return x.second;
                   });
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