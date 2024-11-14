// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include <gtest/gtest.h>
#include <memory>
#include <random>
#include "common/QueryInfo.h"
#include "knowhere/comp/index_param.h"
#include "query/CachedSearchIterator.h"
#include "index/VectorIndex.h"
#include "index/IndexFactory.h"
#include "knowhere/dataset.h"
#include "segcore/InsertRecord.h"
#include "mmap/ChunkedColumn.h"
#include "test_utils/DataGen.h"

using namespace milvus;
using namespace milvus::query;
using namespace milvus::segcore;
using namespace milvus::index;

namespace {
enum class ConstructorType {
    VectorIndex = 0,
    RawData,
    VectorBase,
    ChunkedColumn
};

const int64_t kDim = 16;
const int64_t kNumVectors = 1000;
const int64_t kNumQueries = 5;
const int64_t kBatchSize = 100;
const MetricType kMetricType = knowhere::metric::L2;
const int64_t kSizePerChunk = 128;

using Param = ConstructorType;
}  // namespace

class CachedSearchIteratorTest : public ::testing::TestWithParam<Param> {
 protected:
    static SearchInfo search_info_;
    static DataType data_type_;
    static MetricType metric_type_;
    static int64_t dim_;
    static int64_t nb_;
    static int64_t nq_;

    // these hold data
    static FixedVector<float> base_dataset_;
    static FixedVector<float> query_dataset_;

    static IndexBasePtr index_flat_;
    static IndexBasePtr index_hnsw_;

    static knowhere::DataSetPtr knowhere_query_dataset_;
    static dataset::SearchDataset search_dataset_;

    static std::unique_ptr<ConcurrentVector<float>> vector_base_;

    static std::unique_ptr<ChunkedColumnBase> column_;

    static CachedSearchIterator::GetBitsetViewFunc get_bitset_view_;

    // std::unique_ptr<CachedSearchIterator> CreateIterator() {
    //     auto [constructor_type, metric_type] = GetParam();
    //     switch (constructor_type) {
    //         case ConstructorType::VectorIndex:
    //             return std::make_unique<CachedSearchIterator>(
    //                 index_flat_, knowhere_query_dataset_, search_info_, bitset_);

    //         case ConstructorType::RawData:
    //             return std::make_unique<CachedSearchIterator>(
    //                 dataset_, vectors_.data(), nb_,
    //                 search_info_, bitset_, data_type_);

    //         case ConstructorType::VectorBase:
    //             return std::make_unique<CachedSearchIterator>(
    //                 dataset_, &vector_base_, nb_,
    //                 search_info_, bitset_, data_type_);

    //         case ConstructorType::ChunkedColumn:
    //             return std::make_unique<CachedSearchIterator>(
    //                 column_, dataset_, search_info_,
    //                 bitset_, data_type_, get_bitset_view_);

    //         default:
    //             return nullptr;
    //     }
    // }

    static void
    BuildIndex() {
        auto dataset = knowhere::GenDataSet(nb_, dim_, base_dataset_.data());

        // build Flat
        milvus::index::CreateIndexInfo create_index_info;
        create_index_info.field_type = data_type_;
        create_index_info.metric_type = metric_type_;
        create_index_info.index_type = knowhere::IndexEnum::INDEX_FAISS_IDMAP;
        create_index_info.index_engine_version =
            knowhere::Version::GetCurrentVersion().VersionNumber();
        index_flat_ = milvus::index::IndexFactory::GetInstance().CreateIndex(
            create_index_info, milvus::storage::FileManagerContext());

        auto build_conf = knowhere::Json {
            {knowhere::meta::METRIC_TYPE, knowhere::metric::L2},
            {knowhere::meta::DIM, std::to_string(dim_)}};
        index_flat_->BuildWithDataset(dataset, build_conf);
        ASSERT_EQ(index_flat_->Count(), nb_);

        // build HNSW
        create_index_info.index_type = knowhere::IndexEnum::INDEX_HNSW;
        index_hnsw_ = milvus::index::IndexFactory::GetInstance().CreateIndex(
            create_index_info, milvus::storage::FileManagerContext());
        build_conf[knowhere::indexparam::M] = 10;
        build_conf[knowhere::indexparam::EFCONSTRUCTION] = 128;
        index_hnsw_->BuildWithDataset(dataset, build_conf);
        ASSERT_EQ(index_hnsw_->Count(), nb_);
    }

    static void
    SetUpVectorBase() {
        vector_base_ = std::make_unique<ConcurrentVector<float>>(kSizePerChunk);
        std::vector<FieldDataPtr> field_datas;
        auto num_chunks = (nb_ + kSizePerChunk - 1) / kSizePerChunk;
        for (int64_t i = 0; i < num_chunks; ++i) {
            
        }
        vector_base_->fill_chunk_data(field_datas);
    }

    static void
    SetUpChunkedColumn() {
    }

    static void
    SetUpTestSuite() {
        auto schema = std::make_shared<Schema>();
        auto fakevec_id = schema->AddDebugField(
            "fakevec", DataType::VECTOR_FLOAT, dim_, metric_type_);

        // generate base dataset
        base_dataset_ =
            segcore::DataGen(schema, nb_).get_col<float>(fakevec_id);

        // generate query dataset
        query_dataset_ =
            segcore::DataGen(schema, nq_).get_col<float>(fakevec_id);
        knowhere_query_dataset_ = knowhere::GenDataSet(nq_, dim_, query_dataset_.data());
        search_dataset_ = dataset::SearchDataset{
            .metric_type = metric_type_,
            .num_queries = nq_,
            .topk = kBatchSize,
            .round_decimal = -1,
            .dim = dim_,
            .query_data = query_dataset_.data(),
        };

        BuildIndex();
        SetUpVectorBase();
        SetUpChunkedColumn();
    }

    static void
    TearDownTestSuite() {
        base_dataset_.clear();
        query_dataset_.clear();
        index_flat_.reset();
        index_hnsw_.reset();
        knowhere_query_dataset_.reset();
        vector_base_.reset();
        column_.reset();
    }

    void
    SetUp() override {
    }

    void
    TearDown() override {
    }
};

SearchInfo CachedSearchIteratorTest::search_info_ = SearchInfo{
    .topk_ = kBatchSize,
    .metric_type_ = kMetricType,
};
DataType CachedSearchIteratorTest::data_type_ = DataType::VECTOR_FLOAT;
int64_t CachedSearchIteratorTest::dim_ = kDim;
int64_t CachedSearchIteratorTest::nb_ = kNumVectors;
int64_t CachedSearchIteratorTest::nq_ = kNumQueries;
MetricType CachedSearchIteratorTest::metric_type_ = kMetricType;
IndexBasePtr CachedSearchIteratorTest::index_flat_ = nullptr;
IndexBasePtr CachedSearchIteratorTest::index_hnsw_ = nullptr;
knowhere::DataSetPtr CachedSearchIteratorTest::knowhere_query_dataset_ = nullptr;
dataset::SearchDataset CachedSearchIteratorTest::search_dataset_;
FixedVector<float> CachedSearchIteratorTest::base_dataset_;
FixedVector<float> CachedSearchIteratorTest::query_dataset_;
std::unique_ptr<ConcurrentVector<float>> CachedSearchIteratorTest::vector_base_ =
    nullptr;
std::unique_ptr<ChunkedColumnBase> CachedSearchIteratorTest::column_ = nullptr;
CachedSearchIterator::GetBitsetViewFunc
    CachedSearchIteratorTest::get_bitset_view_ = nullptr;

/********* Testcases Start **********/

TEST_P(CachedSearchIteratorTest, EmptySearch) {
}

/********* Testcases End **********/

static const std::vector<ConstructorType> constructor_types = {
    ConstructorType::VectorIndex,
    ConstructorType::RawData,
    ConstructorType::VectorBase,
    ConstructorType::ChunkedColumn,
};

INSTANTIATE_TEST_SUITE_P(CachedSearchIteratorTests,
                         CachedSearchIteratorTest,
                         ::testing::ValuesIn(constructor_types),
                         [](const testing::TestParamInfo<Param>& info) {
                             std::string constructor_type_str;
                             switch (info.param) {
                                 case ConstructorType::VectorIndex:
                                     constructor_type_str = "VectorIndex";
                                     break;
                                 case ConstructorType::RawData:
                                     constructor_type_str = "RawData";
                                     break;
                                 case ConstructorType::VectorBase:
                                     constructor_type_str = "VectorBase";
                                     break;
                                 case ConstructorType::ChunkedColumn:
                                     constructor_type_str = "ChunkedColumn";
                                     break;
                                 default:
                                     constructor_type_str =
                                         "Unknown constructor type";
                             };
                             return constructor_type_str;
                         });
