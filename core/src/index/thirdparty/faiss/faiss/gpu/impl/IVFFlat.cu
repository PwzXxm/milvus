/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include <faiss/gpu/impl/IVFFlat.cuh>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/impl/FlatIndex.cuh>
#include <faiss/gpu/impl/InterleavedCodes.h>
#include <faiss/gpu/impl/IVFAppend.cuh>
#include <faiss/gpu/impl/IVFFlatScan.cuh>
#include <faiss/gpu/impl/IVFFlatScanLargeK.cuh>
#include <faiss/gpu/impl/IVFInterleaved.cuh>
#include <faiss/gpu/impl/RemapIndices.h>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/Float16.cuh>
#include <faiss/gpu/utils/HostTensor.cuh>
#include <faiss/gpu/utils/Transpose.cuh>
#include <faiss/utils/utils.h>
#include <limits>
#include <thrust/host_vector.h>
#include <unordered_map>
#include <numeric>

#include <string.h>
#include <iostream>

namespace faiss { namespace gpu {

IVFFlat::IVFFlat(GpuResources* res,
                 FlatIndex* quantizer,
                 faiss::MetricType metric,
                 float metricArg,
                 bool useResidual,
                 faiss::ScalarQuantizer* scalarQ,
                 bool interleavedLayout,
                 IndicesOptions indicesOptions,
                 MemorySpace space) :
    IVFBase(res,
            metric,
            metricArg,
            quantizer,
            interleavedLayout,
            indicesOptions,
            space),
    useResidual_(useResidual),
    scalarQ_(scalarQ ? new GpuScalarQuantizer(res, *scalarQ) : nullptr) {
}

IVFFlat::~IVFFlat() {
}


size_t
IVFFlat::getGpuVectorsEncodingSize_(int numVecs) const {
  if (interleavedLayout_) {
    // bits per scalar code
    int bits = scalarQ_ ? scalarQ_->bits : 32 /* float */;

    // bytes to encode a block of 32 vectors (single dimension)
    int bytesPerDimBlock = bits * 32 / 8;

    // bytes to fully encode 32 vectors
    int bytesPerBlock = bytesPerDimBlock * dim_;

    // number of blocks of 32 vectors we have
    int numBlocks = utils::divUp(numVecs, 32);

    // total size to encode numVecs
    return bytesPerBlock * numBlocks;
  } else {
    size_t sizePerVector =
      (scalarQ_ ? scalarQ_->code_size : sizeof(float) * dim_);

    return (size_t) numVecs * sizePerVector;
  }
}

size_t
IVFFlat::getCpuVectorsEncodingSize_(int numVecs) const {
  size_t sizePerVector =
    (scalarQ_ ? scalarQ_->code_size : sizeof(float) * dim_);

  return (size_t) numVecs * sizePerVector;
}

std::vector<uint8_t>
IVFFlat::translateCodesToGpu_(std::vector<uint8_t> codes,
                              size_t numVecs) const {
  if (!interleavedLayout_) {
    // same format
    return codes;
  }

  int bitsPerCode = scalarQ_ ? scalarQ_->bits : 32;

  auto up = unpackNonInterleaved(std::move(codes), numVecs, dim_, bitsPerCode);
  return packInterleaved(std::move(up), numVecs, dim_, bitsPerCode);
}

std::vector<uint8_t>
IVFFlat::translateCodesFromGpu_(std::vector<uint8_t> codes,
                                size_t numVecs) const {
  if (!interleavedLayout_) {
    // same format
    return codes;
  }

  int bitsPerCode = scalarQ_ ? scalarQ_->bits : 32;

  auto up = unpackInterleaved(std::move(codes), numVecs, dim_, bitsPerCode);
  return packNonInterleaved(std::move(up), numVecs, dim_, bitsPerCode);
}

void
IVFFlat::appendVectors_(Tensor<float, 2, true>& vecs,
                        Tensor<Index::idx_t, 1, true>& indices,
                        Tensor<int, 1, true>& uniqueLists,
                        Tensor<int, 1, true>& vectorsByUniqueList,
                        Tensor<int, 1, true>& uniqueListVectorStart,
                        Tensor<int, 1, true>& uniqueListStartOffset,
                        Tensor<int, 1, true>& listIds,
                        Tensor<int, 1, true>& listOffset,
                        cudaStream_t stream) {
  //
  // Append the new encodings
  //

  // Calculate residuals for these vectors, if needed
  DeviceTensor<float, 2, true> residuals(
    resources_, makeTempAlloc(AllocType::Other, stream), {vecs.getSize(0), dim_});

  if (useResidual_) {
    quantizer_->computeResidual(vecs, listIds, residuals);
  }

  // Append indices to the IVF lists
  runIVFIndicesAppend(listIds, listOffset, indices,
                      indicesOptions_, deviceListIndexPointers_, stream);

  // Append the encoded vectors to the IVF lists
  if (interleavedLayout_) {
    runIVFFlatInterleavedAppend(listIds,
                                listOffset,
                                uniqueLists,
                                vectorsByUniqueList,
                                uniqueListVectorStart,
                                uniqueListStartOffset,
                                useResidual_ ? residuals : vecs,
                                scalarQ_.get(),
                                deviceListDataPointers_,
                                resources_,
                                stream);
  } else {
    runIVFFlatAppend(listIds,
                     listOffset,
                     useResidual_ ? residuals : vecs,
                     scalarQ_.get(),
                     deviceListDataPointers_,
                     stream);
  }
}

void
IVFFlat::query(Tensor<float, 2, true>& queries,
               Tensor<uint8_t, 1, true>& bitset,
               int nprobe,
               int k,
               Tensor<float, 2, true>& outDistances,
               Tensor<Index::idx_t, 2, true>& outIndices) {
  auto stream = resources_->getDefaultStreamCurrentDevice();

  // These are caught at a higher level
  FAISS_ASSERT(nprobe <= GPU_MAX_SELECTION_K);
  FAISS_ASSERT(k <= GPU_MAX_SELECTION_K);
  nprobe = std::min(nprobe, quantizer_->getSize());

  FAISS_ASSERT(queries.getSize(1) == dim_);

  FAISS_ASSERT(outDistances.getSize(0) == queries.getSize(0));
  FAISS_ASSERT(outIndices.getSize(0) == queries.getSize(0));

  // Reserve space for the quantized information
  DeviceTensor<float, 2, true> coarseDistances(
    resources_, makeTempAlloc(AllocType::Other, stream), {queries.getSize(0), nprobe});
  DeviceTensor<int, 2, true> coarseIndices(
    resources_, makeTempAlloc(AllocType::Other, stream), {queries.getSize(0), nprobe});

  DeviceTensor<uint8_t, 1, true> coarseBitset(
    resources_, makeTempAlloc(AllocType::Other, stream), {0});
  // Find the `nprobe` closest lists; we can use int indices both
  // internally and externally
  quantizer_->query(queries,
                    coarseBitset,
                    nprobe,
                    metric_,
                    metricArg_,
                    coarseDistances,
                    coarseIndices,
                    false);

  DeviceTensor<float, 3, true> residualBase(
    resources_, makeTempAlloc(AllocType::Other, stream),
    {queries.getSize(0), nprobe, dim_});

  if (useResidual_) {
    // Reconstruct vectors from the quantizer
    quantizer_->reconstruct(coarseIndices, residualBase);
  }

  // if (k > 2048) {
  //   runIVFFlatScanLargeK(queries,
  //                     coarseIndices,
  //                     deviceListDataPointers_,
  //                     deviceListIndexPointers_,
  //                     indicesOptions_,
  //                     deviceListLengths_,
  //                     maxListLength_,
  //                     k,
  //                     metric_,
  //                     useResidual_,
  //                     residualBase,
  //                     scalarQ_.get(),
  //                     outDistances,
  //                     outIndices,
  //                     resources_);
  // } else {
  // }

  if (interleavedLayout_) {
    runIVFInterleavedScan(queries,
                          coarseIndices,
                          deviceListDataPointers_,
                          deviceListIndexPointers_,
                          indicesOptions_,
                          deviceListLengths_,
                          k,
                          metric_,
                          useResidual_,
                          residualBase,
                          scalarQ_.get(),
                          outDistances,
                          outIndices,
                          resources_);
  } else {
    runIVFFlatScan(queries,
                   coarseIndices,
                   bitset,
                   deviceListDataPointers_,
                   deviceListIndexPointers_,
                   indicesOptions_,
                   deviceListLengths_,
                   maxListLength_,
                   k,
                   metric_,
                   useResidual_,
                   residualBase,
                   scalarQ_.get(),
                   outDistances,
                   outIndices,
                   resources_);
  }

  // If the GPU isn't storing indices (they are on the CPU side), we
  // need to perform the re-mapping here
  // FIXME: we might ultimately be calling this function with inputs
  // from the CPU, these are unnecessary copies
  if (indicesOptions_ == INDICES_CPU) {
    HostTensor<Index::idx_t, 2, true> hostOutIndices(outIndices, stream);

    ivfOffsetToUserIndex(hostOutIndices.data(),
                         numLists_,
                         hostOutIndices.getSize(0),
                         hostOutIndices.getSize(1),
                         listOffsetToUserIndex_);

    // Copy back to GPU, since the input to this function is on the
    // GPU
    outIndices.copyFrom(hostOutIndices, stream);
  }
}


void Usort(float *dis, int *ind, int sz){
    for(int i = 0;i < sz;i ++) {
        for(int j = i + 1; j < sz; j ++) {
            if(dis[j] < dis[i]){
            std::swap(dis[i], dis[j]);
            std::swap(ind[i], ind[j]);
            }
        }
    }
}

// QuerySlicing
void
IVFFlat::query(Tensor<float, 2, true>& queries,
               Tensor<uint8_t, 1, true>& bitset,
               int nprobe,
               int k,
               Tensor<float, 2, true>& outDistances,
               Tensor<Index::idx_t, 2, true>& outIndices, 
               float* distances,
               Index::idx_t* labels) {

  auto stream = resources_->getDefaultStreamCurrentDevice();

  // These are caught at a higher level
  FAISS_ASSERT(nprobe <= GPU_MAX_SELECTION_K);
  FAISS_ASSERT(k <= GPU_MAX_SELECTION_K);
  nprobe = std::min(nprobe, quantizer_->getSize());

  FAISS_ASSERT(queries.getSize(1) == dim_);

  FAISS_ASSERT(outDistances.getSize(0) == queries.getSize(0));
  FAISS_ASSERT(outIndices.getSize(0) == queries.getSize(0));

  // Reserve space for the quantized information
  DeviceTensor<float, 2, true> coarseDistances(
    resources_, makeTempAlloc(AllocType::Other, stream), {queries.getSize(0), nprobe});
  DeviceTensor<int, 2, true> coarseIndices(
    resources_, makeTempAlloc(AllocType::Other, stream), {queries.getSize(0), nprobe});

  DeviceTensor<uint8_t, 1, true> coarseBitset(
    resources_, makeTempAlloc(AllocType::Other, stream), {0});
  // Find the `nprobe` closest lists; we can use int indices both
  // internally and externally

  HostTensor<int, 2, true> hostOutStoreIndicesTensor ({queries.getSize(0), k * 2});
  HostTensor<float, 2, true> hostOutStoreDistancesTensor ({queries.getSize(0), k * 2});

  float* hostCoarseDistances = new float[queries.getSize(0) * nprobe];
  int* hostCoarseIndices = new int[queries.getSize(0) * nprobe];

  float* hostOutStoreDistances = hostOutStoreDistancesTensor.data();  
  int* hostOutStoreIndices = hostOutStoreIndicesTensor.data();

  HostTensor<Index::idx_t, 2, true> hostOutIndices(outIndices, stream);
  HostTensor<float, 2, true> hostOutDistances(outDistances, stream);
  float* tmpDistances = hostOutDistances.data(); 
  long* tmpIndices = hostOutIndices.data();
  const int nprobeTile = 8;
  
  for (int i = 0; i < nprobe; i += nprobeTile) {
    int curTile = min(nprobeTile, nprobe-i);

    quantizer_->query(queries,
                        coarseBitset,
                        curTile,
                        metric_,
                        metricArg_,
                        coarseDistances,
                        coarseIndices,
                        hostCoarseDistances,
                        hostCoarseIndices,
                        i,
                        curTile,
                        nprobe,
                        false);
    DeviceTensor<float, 3, true> residualBase(
      resources_, makeTempAlloc(AllocType::Other, stream),
      {queries.getSize(0), nprobe, dim_});

    if (useResidual_) {
        // Reconstruct vectors from the quantizer
        quantizer_->reconstruct(coarseIndices, residualBase);
    }

    runIVFFlatScan(queries,
                    coarseIndices,
                    bitset,
                    deviceListDataPointers_,
                    deviceListIndexPointers_,
                    indicesOptions_,
                    deviceListLengths_,
                    maxListLength_,
                    k,
                    metric_,
                    useResidual_,
                    residualBase,
                    scalarQ_.get(),
                    outDistances,
                    outIndices,
                    resources_);

    fromDevice<float,2>(outDistances, tmpDistances, stream);
    fromDevice<long,2>(outIndices, tmpIndices, stream);
    if(i) {
          for(int d = 0; d < queries.getSize(0); d ++) {
            for(int m = 0; m < k; m ++) {
                hostOutStoreDistances[d * 2 * k + k + m] = tmpDistances[d * k + m];
                hostOutStoreIndices[d * 2 * k + k + m] = tmpIndices[d * k + m];
            }
            Usort(hostOutStoreDistances + k * 2 * d, hostOutStoreIndices + k * 2 * d, 2 * k);
        }
    }

    else{
        for(int d = 0; d < queries.getSize(0); d ++) {
            for(int m = 0; m < k; m ++) {
                hostOutStoreDistances[d * 2 * k + m] = tmpDistances[d * k + m];
                hostOutStoreIndices[d * 2 * k + m] = tmpIndices[d * k + m];
            }
        }
    }

  }

    for(int d = 0; d < queries.getSize(0); d ++) {
        for(int m = 0; m < k; m ++) {
            tmpDistances[d * k + m] = hostOutStoreDistances[d * 2 * k + m];
            tmpIndices[d * k + m] = hostOutStoreIndices[d * 2 * k + m];
        }
    }
    outIndices.copyFrom(hostOutIndices, stream);
    outDistances.copyFrom(hostOutDistances, stream);
  // If the GPU isn't storing indices (they are on the CPU side), we
  // need to perform the re-mapping here
  // FIXME: we might ultimately be calling this function with inputs
  // from the CPU, these are unnecessary copies 
  
  if (indicesOptions_ == INDICES_CPU) {
    HostTensor<Index::idx_t, 2, true> hostOutIndices(outIndices, stream);

    ivfOffsetToUserIndex(hostOutIndices.data(),
                         numLists_,
                         hostOutIndices.getSize(0),
                         hostOutIndices.getSize(1),
                         listOffsetToUserIndex_);

    // Copy back to GPU, since the input to this function is on the
    // GPU
    outIndices.copyFrom(hostOutIndices, stream);
  }

  delete [] hostCoarseDistances;
  delete [] hostCoarseIndices;

}

} } // namespace
