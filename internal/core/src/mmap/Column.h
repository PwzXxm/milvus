// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <folly/io/IOBuf.h>
#include <sys/mman.h>
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <queue>
#include <string>
#include <vector>

#include "common/Array.h"
#include "common/Common.h"
#include "common/EasyAssert.h"
#include "common/File.h"
#include "common/FieldMeta.h"
#include "common/FieldData.h"
#include "common/Span.h"
#include "fmt/format.h"
#include "log/Log.h"
#include "mmap/Utils.h"
#include "common/FieldData.h"
#include "common/FieldDataInterface.h"
#include "common/Array.h"
#include "knowhere/dataset.h"
#include "monitor/prometheus_client.h"
#include "storage/MmapChunkManager.h"

namespace milvus {

constexpr size_t DEFAULT_PK_VRCOL_BLOCK_SIZE = 1;
constexpr size_t DEFAULT_MEM_VRCOL_BLOCK_SIZE = 32;
constexpr size_t DEFAULT_MMAP_VRCOL_BLOCK_SIZE = 256;

/**
 * ColumnBase and its subclasses are designed to store and retrieve the raw data
 * of a field.
 *
 * It has 3 types of constructors corresponding to 3 MappingTypes:
 *
 * 1. MAP_WITH_ANONYMOUS: ColumnBase(size_t reserve_size, const FieldMeta& field_meta)
 *
 *   This is used when we store the entire data in memory. Upon return, a piece
 *   of unwritten memory is allocated and the caller can fill the memory with data by
 *   calling AppendBatch/Append.
 *
 * 2. MAP_WITH_FILE: ColumnBase(const File& file, size_t size, const FieldMeta& field_meta)
 *
 *   This is used when the raw data has already been written into a file, and we
 *   simply mmap the file to memory and interpret the memory as a column. In this
 *   mode, since the data is already in the file/mmapped memory, calling AppendBatch
 *   and Append is not allowed.
 *
 * 3. MAP_WITH_MANAGER: ColumnBase(size_t reserve,
 *                                 const DataType& data_type,
 *                                 storage::MmapChunkManagerPtr mcm,
 *                                 storage::MmapChunkDescriptorPtr descriptor)
 *
 *   This is used when we want to mmap but don't want to download all the data at once.
 *   Instead, we download the data in chunks, cache and mmap each chunk as a single
 *   ColumnBase. Upon return, a piece of unwritten mmaped memory is allocated by the chunk
 *   manager, and the caller should fill the memory with data by calling AppendBatch
 *   and Append.
 *
 * - Types with fixed length can use the Column subclass directly.
 * - Types with variable lengths:
 *   - SparseFloatColumn:
 *     - To store sparse float vectors.
 *     - All 3 modes are supported.
 *   - VariableColumn:
 *      - To store string like types such as VARCHAR and JSON.
 *      - MAP_WITH_MANAGER is not yet supported(as of 2024.09.11).
 *   - ArrayColumn:
 *     - To store ARRAY types.
 *      - MAP_WITH_MANAGER is not yet supported(as of 2024.09.11).
 *
 */
class ColumnBase {
    /**
     * - data_ points at a piece of memory of size data_cap_size_ + padding_.
     *   Based on mapping_type_, such memory can be:
     *   - an anonymous memory region, allocated by mmap(MAP_ANON)
     *   - a file-backed memory region, mapped by mmap(MAP_FILE)
     *   - a memory region managed by MmapChunkManager, allocated by
     *     MmapChunkManager::Allocate()
     *
     * Memory Layout of `data_`:
     *
     * |<--        data_cap_size_         -->|<-- padding_ -->|
     * |<-- data_size_ -->|<-- free space -->|
     *
     * AppendBatch/Append should first check if there's enough space for new data.
     * If not, call Expand() to expand the space.
     *
     * - only the first data_cap_size_ bytes can be used to store actual data.
     * - padding at the end is to ensure when all values are empty, we don't try
     *   to allocate/mmap 0 bytes memory, which will cause mmap() to fail.
     * - data_size_ is the number of bytes currently used to store actual data.
     * - num_rows_ is the number of rows currently stored.
     *
     */
 public:
    enum class MappingType {
        MAP_WITH_ANONYMOUS = 0,
        MAP_WITH_FILE = 1,
        MAP_WITH_MANAGER = 2,
    };
    // MAP_WITH_ANONYMOUS ctor
    ColumnBase(size_t reserve_rows, const FieldMeta& field_meta)
        : mapping_type_(MappingType::MAP_WITH_ANONYMOUS) {
        auto data_type = field_meta.get_data_type();
        SetPaddingSize(data_type);

        // We don't pre-allocate memory for variable length data type, data_
        // will be allocated by Expand() when AppendBatch/Append is called.
        if (IsVariableDataType(data_type)) {
            return;
        }

        data_cap_size_ = field_meta.get_sizeof() * reserve_rows;

        // use anon mapping so we are able to free these memory with munmap only
        size_t mapped_size = data_cap_size_ + padding_;
        data_ = static_cast<char*>(mmap(nullptr,
                                        mapped_size,
                                        PROT_READ | PROT_WRITE,
                                        MAP_PRIVATE | MAP_ANON,
                                        -1,
                                        0));
        AssertInfo(data_ != MAP_FAILED,
                   "failed to create anon map: {}, map_size={}",
                   strerror(errno),
                   mapped_size);

        UpdateMetricWhenMmap(mapped_size);
    }

    // MAP_WITH_MANAGER ctor
    // reserve is number of bytes to allocate(without padding)
    ColumnBase(size_t reserve,
               const DataType& data_type,
               storage::MmapChunkManagerPtr mcm,
               storage::MmapChunkDescriptorPtr descriptor)
        : mcm_(mcm),
          mmap_descriptor_(descriptor),
          num_rows_(0),
          data_size_(0),
          data_cap_size_(reserve),
          mapping_type_(MappingType::MAP_WITH_MANAGER) {
        AssertInfo((mcm != nullptr) && descriptor != nullptr,
                   "use wrong mmap chunk manager and mmap chunk descriptor to "
                   "create column.");

        if (IsVariableDataType(data_type)) {
            AssertInfo(data_type == DataType::VECTOR_SPARSE_FLOAT,
                       "MAP_WITH_MANAGER mode only support VECTOR_SPARSE_FLOAT "
                       "if data type is variable length.");
        }
        SetPaddingSize(data_type);
        size_t mapped_size = data_cap_size_ + padding_;
        data_ = (char*)mcm_->Allocate(mmap_descriptor_, (uint64_t)mapped_size);
        AssertInfo(data_ != nullptr,
                   "fail to create with mmap manager: map_size = {}",
                   mapped_size);
    }

    // MAP_WITH_FILE ctor
    // size is number of bytes of the file, with padding
    // !!! The incoming file must have padding written at the end of the file.
    // Subclasses of variable length data type, if they used this constructor,
    // must set num_rows_ by themselves.
    ColumnBase(const File& file, size_t size, const FieldMeta& field_meta)
        : mapping_type_(MappingType::MAP_WITH_FILE) {
        auto data_type = field_meta.get_data_type();
        SetPaddingSize(data_type);
        if (!IsVariableDataType(data_type)) {
            auto type_size = field_meta.get_sizeof();
            num_rows_ = size / type_size;
        }
        AssertInfo(size >= padding_,
                   "file size {} is less than padding size {}",
                   size,
                   padding_);

        // in MAP_WITH_FILE, no extra space written in file, so data_size_ is
        // the same as data_cap_size_.
        data_size_ = size - padding_;
        data_cap_size_ = data_size_;
        // use exactly same size of file, padding shall be written in file already
        // see also https://github.com/milvus-io/milvus/issues/34442
        data_ = static_cast<char*>(
            mmap(nullptr, size, PROT_READ, MAP_SHARED, file.Descriptor(), 0));
        AssertInfo(data_ != MAP_FAILED,
                   "failed to create file-backed map, err: {}",
                   strerror(errno));
        madvise(data_, size, MADV_WILLNEED);

        UpdateMetricWhenMmap(size);
    }

    virtual ~ColumnBase() {
        if (data_ != nullptr) {
            size_t mapped_size = data_cap_size_ + padding_;
            if (mapping_type_ != MappingType::MAP_WITH_MANAGER) {
                if (munmap(data_, mapped_size)) {
                    AssertInfo(true,
                               "failed to unmap variable field, err={}",
                               strerror(errno));
                }
            }
            UpdateMetricWhenMunmap(data_cap_size_ + padding_);
        }
    }

    ColumnBase(ColumnBase&&) = delete;

    // Data() points at an addr that contains the elements
    virtual const char*
    Data() const {
        return data_;
    }

    // MmappedData() returns the mmaped address
    const char*
    MmappedData() const {
        return data_;
    }

    size_t
    NumRows() const {
        return num_rows_;
    };

    // returns the number of bytes used to store actual data
    size_t
    DataByteSize() const {
        return data_size_;
    }

    // returns the ballpark number of bytes used by this object
    size_t
    MemoryUsageBytes() const {
        return data_cap_size_ + padding_;
    }

    virtual SpanBase
    Span() const = 0;

    // used for sequential access for search
    virtual BufferView
    GetBatchBuffer(int64_t start_offset, int64_t length) {
        PanicInfo(ErrorCode::Unsupported,
                  "GetBatchBuffer only supported for VariableColumn");
    }

    virtual std::vector<std::string_view>
    StringViews() const {
        PanicInfo(ErrorCode::Unsupported,
                  "StringViews only supported for VariableColumn");
    }

    virtual void
    AppendBatch(const FieldDataPtr data) {
        size_t required_size = data_size_ + data->Size();
        if (required_size > data_cap_size_) {
            Expand(required_size * 2);
        }

        std::copy_n(static_cast<const char*>(data->Data()),
                    data->Size(),
                    data_ + data_size_);
        data_size_ = required_size;
        num_rows_ += data->Length();
    }

    // Append one row
    virtual void
    Append(const char* data, size_t size) {
        size_t required_size = data_size_ + size;
        if (required_size > data_cap_size_) {
            Expand(required_size * 2);
        }

        std::copy_n(data, size, data_ + data_size_);
        data_size_ = required_size;
        num_rows_++;
    }

 protected:
    // new_size should not include padding, padding will be added in Expand()
    void
    Expand(size_t new_size) {
        if (new_size == 0) {
            return;
        }
        AssertInfo(
            mapping_type_ == MappingType::MAP_WITH_ANONYMOUS ||
                mapping_type_ == MappingType::MAP_WITH_MANAGER,
            "expand function only use in anonymous or with mmap manager");
        size_t new_mapped_size = new_size + padding_;
        if (mapping_type_ == MappingType::MAP_WITH_ANONYMOUS) {
            auto data = static_cast<char*>(mmap(nullptr,
                                                new_mapped_size,
                                                PROT_READ | PROT_WRITE,
                                                MAP_PRIVATE | MAP_ANON,
                                                -1,
                                                0));
            UpdateMetricWhenMmap(new_mapped_size);

            AssertInfo(data != MAP_FAILED,
                       "failed to expand map: {}, new_map_size={}",
                       strerror(errno),
                       new_size + padding_);

            if (data_ != nullptr) {
                std::memcpy(data, data_, data_size_);
                if (munmap(data_, data_cap_size_ + padding_)) {
                    auto err = errno;
                    size_t mapped_size = new_size + padding_;
                    munmap(data, mapped_size);
                    UpdateMetricWhenMunmap(mapped_size);

                    // TODO: error handling is problematic:
                    // if munmap fails, exception will be thrown and caught by
                    // the cgo call, but the program continue to run. and the
                    // successfully newly mmaped data will not be assigned to data_
                    // and got leaked.
                    AssertInfo(
                        false,
                        "failed to unmap while expanding: {}, old_map_size={}",
                        strerror(err),
                        data_cap_size_ + padding_);
                }
                UpdateMetricWhenMunmap(data_cap_size_ + padding_);
            }

            data_ = data;
            data_cap_size_ = new_size;
        } else if (mapping_type_ == MappingType::MAP_WITH_MANAGER) {
            auto data = mcm_->Allocate(mmap_descriptor_, new_mapped_size);
            AssertInfo(data != nullptr,
                       "fail to create with mmap manager: map_size = {}",
                       new_mapped_size);
            std::memcpy(data, data_, data_size_);
            // allocate space only append in one growing segment, so no need to munmap()
            data_ = (char*)data;
            data_cap_size_ = new_size;
        }
    }

    char* data_{nullptr};
    // capacity in bytes
    size_t data_cap_size_{0};
    size_t padding_{0};
    size_t num_rows_{0};

    // length in bytes
    size_t data_size_{0};
    storage::MmapChunkDescriptorPtr mmap_descriptor_ = nullptr;
    const MappingType mapping_type_;

 private:
    void
    SetPaddingSize(const DataType& type) {
        padding_ = PaddingSize(type);
    }

    void
    UpdateMetricWhenMmap(size_t mapped_size) {
        if (mapping_type_ == MappingType::MAP_WITH_ANONYMOUS) {
            milvus::monitor::internal_mmap_allocated_space_bytes_anon.Observe(
                mapped_size);
            milvus::monitor::internal_mmap_in_used_space_bytes_anon.Increment(
                mapped_size);
            milvus::monitor::internal_mmap_in_used_count_anon.Increment();
        } else if (mapping_type_ == MappingType::MAP_WITH_FILE) {
            milvus::monitor::internal_mmap_allocated_space_bytes_file.Observe(
                mapped_size);
            milvus::monitor::internal_mmap_in_used_space_bytes_file.Increment(
                mapped_size);
            milvus::monitor::internal_mmap_in_used_count_file.Increment();
        }
        // else: does not update metric for MAP_WITH_MANAGER, MmapChunkManagerPtr
        // will update metric itself.
    }

    void
    UpdateMetricWhenMunmap(size_t mapped_size) {
        if (mapping_type_ == MappingType::MAP_WITH_ANONYMOUS) {
            milvus::monitor::internal_mmap_in_used_space_bytes_anon.Decrement(
                mapped_size);
            milvus::monitor::internal_mmap_in_used_count_anon.Decrement();
        } else if (mapping_type_ == MappingType::MAP_WITH_FILE) {
            milvus::monitor::internal_mmap_in_used_space_bytes_file.Decrement(
                mapped_size);
            milvus::monitor::internal_mmap_in_used_count_file.Decrement();
        }
        // else: does not update metric for MAP_WITH_MANAGER, MmapChunkManagerPtr
        // will update metric itself.
    }

    storage::MmapChunkManagerPtr mcm_ = nullptr;
};

class Column : public ColumnBase {
 public:
    // MAP_WITH_ANONYMOUS ctor
    Column(size_t cap, const FieldMeta& field_meta)
        : ColumnBase(cap, field_meta) {
    }

    // MAP_WITH_FILE ctor
    Column(const File& file, size_t size, const FieldMeta& field_meta)
        : ColumnBase(file, size, field_meta) {
    }

    // MAP_WITH_MANAGER ctor
    Column(size_t reserve,
           const DataType& data_type,
           storage::MmapChunkManagerPtr mcm,
           storage::MmapChunkDescriptorPtr descriptor)
        : ColumnBase(reserve, data_type, mcm, descriptor) {
    }

    ~Column() override = default;

    SpanBase
    Span() const override {
        return SpanBase(data_, num_rows_, data_cap_size_ / num_rows_);
    }
};

class SparseFloatColumn : public ColumnBase {
 public:
    // MAP_WITH_ANONYMOUS ctor
    SparseFloatColumn(const FieldMeta& field_meta)
        : ColumnBase(/*reserve_rows= */ 0, field_meta) {
    }

    // MAP_WITH_FILE ctor
    SparseFloatColumn(const File& file,
                      size_t size,
                      const FieldMeta& field_meta,
                      std::vector<uint64_t>&& indices = {})
        : ColumnBase(file, size, field_meta) {
        AssertInfo(!indices.empty(),
                   "SparseFloatColumn indices should not be empty.");
        num_rows_ = indices.size();
        // so that indices[num_rows_] - indices[num_rows_ - 1] is the byte size of
        // the last row.
        indices.push_back(data_size_);
        dim_ = 0;
        for (size_t i = 0; i < num_rows_; i++) {
            auto vec_size = indices[i + 1] - indices[i];
            AssertInfo(
                vec_size % knowhere::sparse::SparseRow<float>::element_size() ==
                    0,
                "Incorrect sparse vector byte size: {}",
                vec_size);
            vec_.emplace_back(
                vec_size / knowhere::sparse::SparseRow<float>::element_size(),
                (uint8_t*)(data_) + indices[i],
                false);
            dim_ = std::max(dim_, vec_.back().dim());
        }
    }

    // MAP_WITH_MANAGER ctor
    SparseFloatColumn(storage::MmapChunkManagerPtr mcm,
                      storage::MmapChunkDescriptorPtr descriptor)
        : ColumnBase(
              /*reserve= */ 0, DataType::VECTOR_SPARSE_FLOAT, mcm, descriptor) {
    }

    ~SparseFloatColumn() override = default;

    // returned pointer points at a list of knowhere::sparse::SparseRow<float>
    const char*
    Data() const override {
        return static_cast<const char*>(static_cast<const void*>(vec_.data()));
    }

    SpanBase
    Span() const override {
        PanicInfo(ErrorCode::Unsupported,
                  "SparseFloatColumn::Span() not supported");
    }

    void
    AppendBatch(const FieldDataPtr data) override {
        AssertInfo(
            mapping_type_ != MappingType::MAP_WITH_FILE,
            "SparseFloatColumn::AppendBatch not supported for MAP_WITH_FILE");

        size_t required_size = data_size_ + data->Size();
        if (required_size > data_cap_size_) {
            Expand(required_size * 2);
            // after expanding, the address of each row in vec_ become invalid.
            // the number of elements of each row is still correct, update the
            // address of each row to the new data_.
            size_t bytes = 0;
            for (size_t i = 0; i < num_rows_; i++) {
                auto count = vec_[i].size();
                auto row_bytes = vec_[i].data_byte_size();
                // destroy the old object and placement new a new one
                vec_[i].~SparseRow<float>();
                new (&vec_[i]) knowhere::sparse::SparseRow<float>(
                    count, (uint8_t*)(data_) + bytes, false);
                bytes += row_bytes;
            }
        }
        dim_ = std::max(
            dim_,
            std::static_pointer_cast<FieldDataSparseVectorImpl>(data)->Dim());

        auto ptr = static_cast<const knowhere::sparse::SparseRow<float>*>(
            data->Data());

        for (size_t i = 0; i < data->Length(); ++i) {
            auto row_bytes = ptr[i].data_byte_size();
            std::memcpy(data_ + data_size_, ptr[i].data(), row_bytes);
            vec_.emplace_back(
                ptr[i].size(), (uint8_t*)(data_) + data_size_, false);
            data_size_ += row_bytes;
        }
        num_rows_ += data->Length();
    }

    void
    Append(const char* data, size_t size) override {
        PanicInfo(
            ErrorCode::Unsupported,
            "SparseFloatColumn::Append not supported, use AppendBatch instead");
    }

    int64_t
    Dim() const {
        return dim_;
    }

 private:
    int64_t dim_ = 0;
    std::vector<knowhere::sparse::SparseRow<float>> vec_;
};

template <typename T>
class VariableColumn : public ColumnBase {
 public:
    using ViewType =
        std::conditional_t<std::is_same_v<T, std::string>, std::string_view, T>;

    // MAP_WITH_ANONYMOUS ctor
    VariableColumn(size_t reserve_rows,
                   const FieldMeta& field_meta,
                   size_t block_size)
        : ColumnBase(reserve_rows, field_meta), block_size_(block_size) {
    }

    // MAP_WITH_FILE ctor
    VariableColumn(const File& file,
                   size_t size,
                   const FieldMeta& field_meta,
                   size_t block_size)
        : ColumnBase(file, size, field_meta), block_size_(block_size) {
    }

    ~VariableColumn() override = default;

    SpanBase
    Span() const override {
        PanicInfo(ErrorCode::NotImplemented,
                  "span() interface is not implemented for variable column");
    }

    std::vector<std::string_view>
    StringViews() const override {
        std::vector<std::string_view> res;
        char* pos = data_;
        for (size_t i = 0; i < num_rows_; ++i) {
            uint32_t size;
            size = *reinterpret_cast<uint32_t*>(pos);
            pos += sizeof(uint32_t);
            res.emplace_back(std::string_view(pos, size));
            pos += size;
        }
        return res;
    }

    [[nodiscard]] std::vector<ViewType>
    Views() const {
        std::vector<ViewType> res;
        char* pos = data_;
        for (size_t i = 0; i < num_rows_; ++i) {
            uint32_t size;
            size = *reinterpret_cast<uint32_t*>(pos);
            pos += sizeof(uint32_t);
            res.emplace_back(ViewType(pos, size));
            pos += size;
        }
        return res;
    }

    BufferView
    GetBatchBuffer(int64_t start_offset, int64_t length) override {
        if (start_offset < 0 || start_offset > num_rows_ ||
            start_offset + length > num_rows_) {
            PanicInfo(ErrorCode::OutOfRange, "index out of range");
        }

        char* pos = data_ + indices_[start_offset / block_size_];
        for (size_t j = 0; j < start_offset % block_size_; j++) {
            uint32_t size;
            size = *reinterpret_cast<uint32_t*>(pos);
            pos += sizeof(uint32_t) + size;
        }

        return BufferView{pos, data_size_ - (pos - data_)};
    }

    ViewType
    operator[](const int i) const {
        if (i < 0 || i > num_rows_) {
            PanicInfo(ErrorCode::OutOfRange, "index out of range");
        }
        size_t batch_id = i / block_size_;
        size_t offset = i % block_size_;

        // located in batch start location
        char* pos = data_ + indices_[batch_id];
        for (size_t j = 0; j < offset; j++) {
            uint32_t size;
            size = *reinterpret_cast<uint32_t*>(pos);
            pos += sizeof(uint32_t) + size;
        }

        uint32_t size;
        size = *reinterpret_cast<uint32_t*>(pos);
        return ViewType(pos + sizeof(uint32_t), size);
    }

    std::string_view
    RawAt(const int i) const {
        return std::string_view((*this)[i]);
    }

    void
    Append(FieldDataPtr chunk) {
        for (auto i = 0; i < chunk->get_num_rows(); i++) {
            indices_.emplace_back(data_size_);
            auto data = static_cast<const T*>(chunk->RawValue(i));
            data_size_ += sizeof(uint32_t) + data->size();
        }
        load_buf_.emplace(std::move(chunk));
    }

    void
    Seal(std::vector<uint64_t> indices = {}) {
        if (!indices.empty()) {
            indices_ = std::move(indices);
        }

        num_rows_ = indices_.size();

        // for variable length column in memory mode only
        if (data_ == nullptr) {
            size_t total_size = data_size_;
            data_size_ = 0;
            Expand(total_size);

            while (!load_buf_.empty()) {
                auto chunk = std::move(load_buf_.front());
                load_buf_.pop();

                // data_ as: |size|data|size|data......
                for (auto i = 0; i < chunk->get_num_rows(); i++) {
                    auto current_size = (uint32_t)chunk->Size(i);
                    std::memcpy(
                        data_ + data_size_, &current_size, sizeof(uint32_t));
                    data_size_ += sizeof(uint32_t);
                    auto data = static_cast<const T*>(chunk->RawValue(i));
                    std::memcpy(
                        data_ + data_size_, data->c_str(), data->size());
                    data_size_ += data->size();
                }
            }
        }

        shrink_indice();
    }

 protected:
    void
    shrink_indice() {
        std::vector<uint64_t> tmp_indices;
        tmp_indices.reserve((indices_.size() + block_size_ - 1) / block_size_);

        for (size_t i = 0; i < indices_.size();) {
            tmp_indices.push_back(indices_[i]);
            i += block_size_;
        }

        indices_.swap(tmp_indices);
    }

 private:
    // loading states
    std::queue<FieldDataPtr> load_buf_{};
    // raw data index, record indices located 0, block_size_, 2 * block_size_, 3 * block_size_
    size_t block_size_;
    std::vector<uint64_t> indices_{};
};

class ArrayColumn : public ColumnBase {
 public:
    // MAP_WITH_ANONYMOUS ctor
    ArrayColumn(size_t reserve_rows, const FieldMeta& field_meta)
        : ColumnBase(reserve_rows, field_meta),
          element_type_(field_meta.get_element_type()) {
    }

    // MAP_WITH_FILE ctor
    ArrayColumn(const File& file, size_t size, const FieldMeta& field_meta)
        : ColumnBase(file, size, field_meta),
          element_type_(field_meta.get_element_type()) {
    }

    ~ArrayColumn() override = default;

    SpanBase
    Span() const override {
        return SpanBase(views_.data(), views_.size(), sizeof(ArrayView));
    }

    [[nodiscard]] const std::vector<ArrayView>&
    Views() const {
        return views_;
    }

    ArrayView
    operator[](const int i) const {
        return views_[i];
    }

    ScalarArray
    RawAt(const int i) const {
        return views_[i].output_data();
    }

    void
    Append(const Array& array) {
        indices_.emplace_back(data_size_);
        element_indices_.emplace_back(array.get_offsets());
        ColumnBase::Append(static_cast<const char*>(array.data()),
                           array.byte_size());
    }

    void
    Seal(std::vector<uint64_t>&& indices = {},
         std::vector<std::vector<uint64_t>>&& element_indices = {}) {
        if (!indices.empty()) {
            indices_ = std::move(indices);
            element_indices_ = std::move(element_indices);
        }
        num_rows_ = indices_.size();
        ConstructViews();
    }

 protected:
    void
    ConstructViews() {
        views_.reserve(indices_.size());
        for (size_t i = 0; i < indices_.size() - 1; i++) {
            views_.emplace_back(data_ + indices_[i],
                                indices_[i + 1] - indices_[i],
                                element_type_,
                                std::move(element_indices_[i]));
        }
        views_.emplace_back(data_ + indices_.back(),
                            data_size_ - indices_.back(),
                            element_type_,
                            std::move(element_indices_[indices_.size() - 1]));
        element_indices_.clear();
    }

 private:
    std::vector<uint64_t> indices_{};
    std::vector<std::vector<uint64_t>> element_indices_{};
    // Compatible with current Span type
    std::vector<ArrayView> views_{};
    DataType element_type_;
};
}  // namespace milvus
