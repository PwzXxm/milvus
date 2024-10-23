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

#pragma once

#include <cstddef>
#include <vector>
#include <optional>
#include "common/QueryInfo.h"
#include "common/QueryResult.h"
#include "common/BitsetView.h"
#include "knowhere/dataset.h"
#include "index/VectorIndex.h"
#include "query/SearchBruteForce.h"
#include "segcore/InsertRecord.h"

namespace milvus::query {

class CachedSearchIterator {
public:
    CachedSearchIterator(const milvus::index::VectorIndex& index,
                         const knowhere::DataSetPtr& dataset,
                         const SearchInfo& search_info,
                         const BitsetView& bitset);

    CachedSearchIterator(const dataset::SearchDataset& dataset,
                         const void* vec_data,
                         const int64_t row_count,
                         const SearchInfo& search_info,
                         const BitsetView& bitset,
                         const milvus::DataType& data_type);

    CachedSearchIterator(const dataset::SearchDataset& dataset,
                         const segcore::VectorBase* vec_data,
                         const int64_t row_count,
                         const SearchInfo& search_info,
                         const BitsetView& bitset,
                         const milvus::DataType& data_type);

    void NextBatch(const SearchInfo& search_info, SearchResult& search_result);

private:
    using DisIdPair = std::pair<float, int64_t>;

    DisIdPair
    ConvertIteratorResult(const std::pair<int64_t, float>& iter_rst,
                          const size_t chunk_idx = 0);
    std::optional<DisIdPair>
    GetNextValidResult(size_t iterator_idx,
                       const std::optional<double>& last_bound,
                       const size_t chunk_id = 0);
    void MergeChunksResults(size_t query_idx,
                             const std::optional<double>& last_bound,
                             std::vector<DisIdPair>& rst);

    void IteratorsSearch(const SearchInfo& search_info, SearchResult& search_result);
    void ValidateSearchInfo(const SearchInfo& search_info);
    void RefillIteratorResultPool();
    std::vector<DisIdPair>
    GetBatchedNextResults(size_t query_idx,
                          const SearchInfo& search_info);
    void WriteSingleQuerySearchResult(SearchResult& search_result,
                                      const size_t idx,
                                      std::vector<DisIdPair>& rst,
                                      const int64_t round_decimal);

    void Init(const SearchInfo& search_info);

    int64_t batch_size_ = 0;
    std::vector<knowhere::IndexNode::IteratorPtr> iterators_;
    int8_t sign_ = 1;
    int64_t vec_size_per_chunk_ = 1;
    size_t num_chunks_ = 1;
    size_t nq_ = 0;
};

}  // namespace milvus::query

/*
namespace milvus::query {

// maybe one-one or one-many, provide universal interface for both
// same metric type
class CachedSearchIterator {
 public:
    // could be one or multiple queries
    CachedSearchIterator(const milvus::index::VectorIndex& index,
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
        nq_ = iterators_.size();
        Init(search_info);
    };

    // Only for BruteForce
    CachedSearchIterator(const knowhere::DataSetPtr& query_dataset,
                         const void* vec_data,
                         const int64_t row_count,
                         const SearchInfo& search_info,
                         const BitsetView& bitset,
                         const milvus::DataType& data_type) {
        nq_ = query_dataset->GetRows();
        Init(search_info);
    }

    void
    NextBatch(const SearchInfo& search_info, SearchResult& search_result) {
        if (iterators_.empty()) {
            return;
        }

        ValidateSearchInfo(search_info);

        search_result.total_nq_ = nq_;
        search_result.unity_topK_ = batch_size_;
        search_result.seg_offsets_.resize(nq_ * batch_size_);
        search_result.distances_.resize(nq_ * batch_size_);
        for (size_t idx = 0; idx < nq_; ++idx) {
            IteratorsSearch(idx, search_info, search_result);
        }
    }

 private:
    using DisIdPair = std::pair<int64_t, float>;

    void
    IteratorsSearch(const size_t idx,
        const SearchInfo& search_info,
                         SearchResult& search_result) {
        if (chunk_size_ == 1) {
            auto& iterator = iterators_[idx];
            const auto& iterator_v2_info =
                search_info.iterator_v2_info_.value();
            auto rst = GetBatchedNextResults(
                search_result, iterator, iterator_v2_info.last_bound);
            WriteSingleQuerySearchResult(
                search_result, idx, rst, search_info.round_decimal_);
        } else if (chunk_size_ > 1) {
            // TODO
        } else {
            PanicInfo(ErrorCode::UnexpectedError, "Invalid chunk size");
        }
    }

    void
    ValidateSearchInfo(const SearchInfo& search_info) {
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
    RefillIteratorResultPool() {
    }

    std::vector<DisIdPair>
    GetBatchedNextResults(SearchResult& search_result,
                         knowhere::IndexNode::IteratorPtr& iterator,
                         const std::optional<double>& last_bound) {
        std::vector<DisIdPair> rst;
        rst.reserve(batch_size_);
        // TODO: support batched next in Knowhere to optimize this
        while (iterator->HasNext() && rst.size() < batch_size_) {
            auto result = iterator->Next();
            if (sign_ == -1) {
                result.second *= sign_;
            }
            if (last_bound.has_value() && result.first <= last_bound.value()) {
                continue;
            }
            rst.emplace_back(result);
        }
        std::sort(rst.begin(), rst.end());
        if (sign_ == -1) {
            std::for_each(rst.begin(), rst.end(), [this](DisIdPair& x) {x.second = x.second * sign_;});
        }
        while (rst.size() < batch_size_) {
            rst.emplace_back(DisIdPair(-1, (1.0f / 0.0f)));
        }
        return rst;
    }

    void
    WriteSingleQuerySearchResult(SearchResult& search_result,
                      const size_t idx,
                      std::vector<DisIdPair>& rst,
                      const int64_t round_decimal) {
        std::copy_n(rst.begin(), batch_size_, search_result.seg_offsets_.begin() + idx * batch_size_);
        std::copy_n(rst.begin(), batch_size_, search_result.distances_.begin() + idx * batch_size_);
    }


    void
    Init(const SearchInfo& search_info) {
        if (search_info.iterator_v2_info_.has_value()) {
            auto iterator_v2_info = search_info.iterator_v2_info_.value();
            batch_size_ = iterator_v2_info.batch_size;
        }

        if (PositivelyRelated(search_info.metric_type_)) {
            sign_ = -1;
        } else {
            sign_ = 1;
        }

        // result_pool_ = std::make_unique<SignedPriorityQueue>(search_info.metric_type_);
    }

    /*
    class SignedPriorityQueue {
     public:
        explicit SignedPriorityQueue(const knowhere::MetricType& metric_type) {
            if (PositivelyRelated(metric_type)) {
                comparator_ = [](const DisIdPair& lhs, const DisIdPair& rhs) {
                    return lhs.second < rhs.second;
                };
            } else {
                comparator_ = [](const DisIdPair& lhs, const DisIdPair& rhs) {
                    return lhs.second > rhs.second;
                };
            }
        }

        void
        Push(const DisIdPair& result) {
            queue_.push(result);
        }

        DisIdPair
        Pop() {
            auto result = queue_.top();
            queue_.pop();
            return result;
        }

        bool
        Empty() const {
            return queue_.empty();
        }

     private:
        std::function<bool(const DisIdPair&, const DisIdPair&)> comparator_;
        std::priority_queue<DisIdPair,
                            std::vector<DisIdPair>,
                            decltype(comparator_)>
            queue_;
    };

    int64_t batch_size_ = 0;
    std::vector<knowhere::IndexNode::IteratorPtr> iterators_;
    int8_t sign_ = 1;
    size_t chunk_size_ = 1;
    size_t nq_ = 0;
    // std::unique_ptr<SignedPriorityQueue> result_pool_;
};
}  // namespace milvus::query
*/
