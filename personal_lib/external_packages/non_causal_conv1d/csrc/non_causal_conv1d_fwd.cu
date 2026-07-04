/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Modifications Copyright (c) 2026, Sony Group Corporation.
 ******************************************************************************/

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>

#include "non_causal_conv1d.h"
#include "non_causal_conv1d_common.h"
#include "static_switch.h"

template<int kNThreads_, int kWidth_, bool kIsVecLoad_, typename input_t_, typename weight_t_>
struct non_causal_conv1d_fwd_kernel_traits {
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kWidth = kWidth_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : 8;
    static_assert(kWidth <= kNElts);
    static constexpr bool kIsVecLoad = kIsVecLoad_;
    static constexpr int kNLoadsIndex = kNElts / 4;
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNElts, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, 1, cub::BLOCK_LOAD_DIRECT>;
    using BlockLoadIndexT = cub::BlockLoad<int, kNThreads, kNElts, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadIndexVecT = cub::BlockLoad<int4, kNThreads, kNLoadsIndex,!(kIsVecLoad && kNLoadsIndex == 1) ? cub::BLOCK_LOAD_WARP_TRANSPOSE : cub::BLOCK_LOAD_DIRECT>;
    using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNElts, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, 1, cub::BLOCK_STORE_DIRECT>;

    static constexpr int kSmemIOSize = (kIsVecLoad && kNLoadsIndex == 1)
        ? 0
        : std::max({sizeof(typename BlockLoadT::TempStorage), 
                    sizeof(typename BlockStoreT::TempStorage),
                    sizeof(typename BlockLoadIndexT::TempStorage),
                    sizeof(typename BlockLoadIndexVecT::TempStorage)});
    static constexpr int kSmemExchangeSize = kNThreads * kNBytes * kNElts;
    static constexpr int kSmemSize = kSmemIOSize + kSmemExchangeSize;
};

template<typename Ktraits, bool kHasPosIdx>
__global__ __launch_bounds__(Ktraits::kNThreads)
void non_causal_conv1d_fwd_kernel(ConvParamsBase params) {
    constexpr int kWidth = Ktraits::kWidth;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNElts = Ktraits::kNElts;
    static constexpr bool kIsVecLoad = Ktraits::kIsVecLoad;
    using input_t = typename Ktraits::input_t;
    using vec_t = typename Ktraits::vec_t;
    using weight_t = typename Ktraits::weight_t;

    // Shared memory.
    extern __shared__ char smem_[];
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_load_vec = reinterpret_cast<typename Ktraits::BlockLoadVecT::TempStorage&>(smem_);
    auto& smem_load_index = reinterpret_cast<typename Ktraits::BlockLoadIndexT::TempStorage&>(smem_);
    auto& smem_load_index_vec = reinterpret_cast<typename Ktraits::BlockLoadIndexVecT::TempStorage&>(smem_);
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_store_vec = reinterpret_cast<typename Ktraits::BlockStoreVecT::TempStorage&>(smem_);
    vec_t *smem_exchange = reinterpret_cast<vec_t *>(smem_ + Ktraits::kSmemIOSize);

    const int tidx = threadIdx.x;
    const int batch_id = blockIdx.x;
    const int channel_id = blockIdx.y;
    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride
        + channel_id * params.x_c_stride;
    weight_t *weight = reinterpret_cast<weight_t *>(params.weight_ptr) + channel_id * params.weight_c_stride;
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
        + channel_id * params.out_c_stride;
    float bias_val = params.bias_ptr == nullptr ? 0.f : float(reinterpret_cast<weight_t *>(params.bias_ptr)[channel_id]);
    
    int *position_indices = !kHasPosIdx ? nullptr : reinterpret_cast<int *>(params.position_indices_ptr) + batch_id * params.seqlen;

    // Thread 0 will load the last elements of the previous chunk, so we initialize those to 0.
    if (tidx == 0) {
        input_t zeros[kNElts] = {0};
        smem_exchange[kNThreads - 2] = reinterpret_cast<vec_t *>(zeros)[0];

        input_t x_begin[kNElts];
        if constexpr(kIsVecLoad) {
            reinterpret_cast<vec_t*>(x_begin)[0] = reinterpret_cast<vec_t*>(x)[0];
        } else{
            #pragma unroll
            for (int i = 0; i < kNElts; i++) {
                x_begin[i] = x[i];
            }
        }
        smem_exchange[kNThreads - 1] = reinterpret_cast<vec_t *>(x_begin)[0];
    }

    __syncthreads();
    float weight_vals[kWidth];
    #pragma unroll
    for (int i = 0; i < kWidth; ++i) { weight_vals[i] = float(weight[i * params.weight_width_stride]); }

    constexpr int kChunkSize = kNThreads * kNElts;
    const int n_chunks = (params.seqlen + kChunkSize - 1) / kChunkSize;
    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        input_t x_vals_load[3 * kNElts] = {0};
        int position_indices_load[kNElts];
        __syncthreads();
        if constexpr(kIsVecLoad) {
            Ktraits::BlockLoadVecT(smem_load_vec).Load(reinterpret_cast<vec_t*>(x+kNElts), *reinterpret_cast<vec_t (*)[1]>(&x_vals_load[2*kNElts]), (params.seqlen - chunk * kChunkSize-kNElts) / kNElts);
            if constexpr (kHasPosIdx){
                Ktraits::BlockLoadIndexVecT(smem_load_index_vec).Load(reinterpret_cast<int4*>(position_indices), *reinterpret_cast<int4(*)[Ktraits::kNLoadsIndex]>(position_indices_load), (params.seqlen - chunk * kChunkSize) / kNElts * Ktraits::kNLoadsIndex);
            }
        } else {                                                      
            Ktraits::BlockLoadT(smem_load).Load(x+kNElts, *reinterpret_cast<input_t (*)[kNElts]>(&x_vals_load[kNElts*2]), params.seqlen - chunk * kChunkSize-kNElts);
            if constexpr (kHasPosIdx){
                Ktraits::BlockLoadIndexT(smem_load_index).Load(position_indices, position_indices_load, (params.seqlen - chunk * kChunkSize), 0);
            }
        }
        __syncthreads();

        x += kChunkSize;
        if constexpr (kHasPosIdx)   position_indices += kChunkSize;

        // Thread kNThreads - 1 don't write yet, so that thread 0 can read
        // the last elements of the previous chunk.
        if (tidx < kNThreads - 2) { smem_exchange[tidx] = reinterpret_cast<vec_t *>(x_vals_load)[2]; }
        __syncthreads();
        if (tidx < kNThreads - 1) {
            reinterpret_cast<vec_t *>(x_vals_load)[0] = smem_exchange[(kNThreads+tidx-2) % kNThreads];
            reinterpret_cast<vec_t *>(x_vals_load)[1] = smem_exchange[(kNThreads+tidx-1) % kNThreads];
        }
        __syncthreads();
        // Now thread kNThreads - 1 can write the last elements of the current chunk.
        if (tidx >= kNThreads - 2) { smem_exchange[tidx] = reinterpret_cast<vec_t *>(x_vals_load)[2]; }
        __syncthreads();
        if (tidx == kNThreads - 1) {
            reinterpret_cast<vec_t *>(x_vals_load)[0] = smem_exchange[(kNThreads+tidx-2) % kNThreads];
            reinterpret_cast<vec_t *>(x_vals_load)[1] = smem_exchange[(kNThreads+tidx-1) % kNThreads];
        }


        float x_vals[3 * kNElts];
        #pragma unroll
        for (int i = 0; i < 3 * kNElts; ++i) { x_vals[i] = float(x_vals_load[i]); }

        float out_vals[kNElts];
        
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) {
            out_vals[i] = bias_val;
            // #pragma unroll
            int w = 0;
            int w_end = kWidth;
            if constexpr (kHasPosIdx){
                if (position_indices_load[i] == 1){
                    w = 1;
                } else if (position_indices_load[i] == -1){
                    w_end = kWidth -1;
                }
            }
            for (; w < w_end; ++w) {
                out_vals[i] += weight_vals[w] * x_vals[kNElts + i - (kWidth/2 - w)];
            }
        }

        if (params.silu_activation) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) {
                out_vals[i] = out_vals[i] / (1 + expf(-out_vals[i]));
            }
        }

        input_t out_vals_store[kNElts];
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) { out_vals_store[i] = out_vals[i]; }
        if constexpr(kIsVecLoad) {
            Ktraits::BlockStoreVecT(smem_store_vec).Store(reinterpret_cast<vec_t*>(out), reinterpret_cast<vec_t (&)[1]>(out_vals_store), (params.seqlen - chunk * kChunkSize) / kNElts);
        } else {
            Ktraits::BlockStoreT(smem_store).Store(out, out_vals_store, params.seqlen - chunk * kChunkSize);
        }
        out += kChunkSize;
    }
}

template<int kNThreads, int kWidth, typename input_t, typename weight_t>
void non_causal_conv1d_fwd_launch(ConvParamsBase &params, cudaStream_t stream) {
    static constexpr int kNElts = sizeof(input_t) == 4 ? 4 : 8;
    BOOL_SWITCH(params.position_indices_ptr != nullptr, kHasPosIdx, [&] {
        BOOL_SWITCH(params.seqlen % kNElts == 0, kIsVecLoad, [&] {
            using Ktraits = non_causal_conv1d_fwd_kernel_traits<kNThreads, kWidth, kIsVecLoad, input_t, weight_t>;
            constexpr int kSmemSize = Ktraits::kSmemSize;
            dim3 grid(params.batch, params.dim);
            auto kernel = &non_causal_conv1d_fwd_kernel<Ktraits, kHasPosIdx>;
            if (kSmemSize >= 48 * 1024) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
                }
            kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
}

template<typename input_t, typename weight_t>
void non_causal_conv1d_fwd_cuda(ConvParamsBase &params, cudaStream_t stream) {
    switch (params.num_threads) {
        case 64:
            non_causal_conv1d_fwd_launch<64, 3, input_t, weight_t>(params, stream); break;
        case 128:
            non_causal_conv1d_fwd_launch<128, 3, input_t, weight_t>(params, stream); break;
        case 256:
            non_causal_conv1d_fwd_launch<256, 3, input_t, weight_t>(params, stream); break;
        case 512:
            non_causal_conv1d_fwd_launch<512, 3, input_t, weight_t>(params, stream); break;
        default:
            non_causal_conv1d_fwd_launch<128, 3, input_t, weight_t>(params, stream); break;
    }
}

template<int kNThreads_, int kWidth_, int kChunkSizeL_, bool kIsVecLoad_, typename input_t_, typename weight_t_>
struct non_causal_conv1d_channellast_fwd_kernel_traits {
    // The cache line is 128 bytes, and we try to read 16 bytes per thread.
    // So we have 8 threads per "row", so 32 or 64 elements in the channel dimension.
    // That leaves 4 columns per warp, and so 16 columns per block (assuming each block has 128
    // threads). Each each load is 16 x 32|64 elements in the L x C dimensions.
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    static_assert(kNThreads % 32 == 0);
    static constexpr int kNWarps = kNThreads / 32;
    static constexpr int kWidth = kWidth_;
    static constexpr int kChunkSizeL = kChunkSizeL_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : 8;
    static constexpr int kNEltsPerRow = 128 / kNBytes;
    static constexpr int kNThreadsPerRow = kNEltsPerRow / kNElts;  // Always 8 for now
    static_assert(kNThreadsPerRow * kNBytes * kNElts == 128);
    static constexpr int kNColsPerWarp = 32 / kNThreadsPerRow;  // Always 4 for now
    static_assert(kNColsPerWarp * kNThreadsPerRow == 32);
    static constexpr int kNColsPerLoad = kNColsPerWarp * kNWarps;
    static constexpr int kNLoads = kChunkSizeL / kNColsPerLoad;
    static_assert(kNLoads * kNColsPerLoad == kChunkSizeL);
    static constexpr bool kIsVecLoad = kIsVecLoad_;
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;

};

template<typename Ktraits, bool kHasPosIdx>
__global__ __launch_bounds__(Ktraits::kNThreads)
void non_causal_conv1d_channellast_fwd_kernel(ConvParamsBase params) {
    constexpr int kWidth = Ktraits::kWidth;
    constexpr int kWidth_div = Ktraits::kWidth/2;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNElts = Ktraits::kNElts;
    constexpr int kNLoads = Ktraits::kNLoads;
    constexpr int kNThreadsPerC = Ktraits::kNThreadsPerRow;
    constexpr int kLPerLoad = Ktraits::kNColsPerLoad;
    constexpr int kChunkSizeL = Ktraits::kChunkSizeL;
    constexpr int kChunkSizeC = Ktraits::kNEltsPerRow;
    using input_t = typename Ktraits::input_t;
    using vec_t = typename Ktraits::vec_t;
    using weight_t = typename Ktraits::weight_t;

    // Shared memory.
    __shared__ input_t x_smem[2*kWidth_div + kChunkSizeL][kChunkSizeC + kNElts];
    __shared__ int index_smem[kChunkSizeL];

    const int batch_id = blockIdx.x;
    const int chunk_l_id = blockIdx.y;
    const int chunk_c_id = blockIdx.z;
    const int tid = threadIdx.x;
    const int l_idx = tid / kNThreadsPerC;
    const int c_idx = tid % kNThreadsPerC;
    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride
        + (chunk_l_id * kChunkSizeL + l_idx) * params.x_l_stride + chunk_c_id * kChunkSizeC + c_idx * kNElts;
    weight_t *weight = reinterpret_cast<weight_t *>(params.weight_ptr)
        + chunk_c_id * kChunkSizeC * params.weight_c_stride;
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
        + (chunk_l_id * kChunkSizeL + l_idx) * params.out_l_stride + chunk_c_id * kChunkSizeC + c_idx * kNElts;
    int *position_indices = !kHasPosIdx ? nullptr : reinterpret_cast<int *>(params.position_indices_ptr)
        + batch_id * params.seqlen + chunk_l_id * kChunkSizeL;
    
    // Load the elements from the previous chunk that are needed for convolution.
    if (l_idx < kWidth_div) {
        input_t x_vals_load[kNElts] = {0};
        if (chunk_l_id * kChunkSizeL + l_idx - kWidth_div >= 0
            && chunk_l_id * kChunkSizeL + l_idx - kWidth_div < params.seqlen
            && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
            reinterpret_cast<vec_t *>(x_vals_load)[0] = *reinterpret_cast<vec_t *>(x - kWidth_div * params.x_l_stride);
        }
        reinterpret_cast<vec_t *>(x_smem[l_idx])[c_idx] = reinterpret_cast<vec_t *>(x_vals_load)[0];
    }

    #pragma unroll
    for (int l = 0; l < kNLoads; ++l) {
        input_t x_vals_load[kNElts] = {0};
        if (chunk_l_id * kChunkSizeL + l * kLPerLoad + l_idx < params.seqlen
            && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
            reinterpret_cast<vec_t *>(x_vals_load)[0] = *reinterpret_cast<vec_t *>(x + l * kLPerLoad * params.x_l_stride);
        }
        reinterpret_cast<vec_t *>(x_smem[kWidth_div + l * kLPerLoad + l_idx])[c_idx] = reinterpret_cast<vec_t *>(x_vals_load)[0];
    }

    if (l_idx >= kChunkSizeL / kLPerLoad - kWidth_div) { // Load the elements from the future chunk that are needed for convolution.
        input_t x_vals_load[kNElts] = {0};
        if (chunk_l_id * kChunkSizeL + l_idx + (kNLoads-1) * kLPerLoad + kWidth_div >= 0
            && chunk_l_id * kChunkSizeL + l_idx + (kNLoads-1) * kLPerLoad + kWidth_div < params.seqlen
            && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
            reinterpret_cast<vec_t *>(x_vals_load)[0] = *reinterpret_cast<vec_t *>(x + ((kNLoads-1) * kLPerLoad + kWidth_div) * params.x_l_stride);
        }
        reinterpret_cast<vec_t *>(x_smem[kWidth_div + (kNLoads-1) * kLPerLoad + l_idx+ kWidth_div])[c_idx] = reinterpret_cast<vec_t *>(x_vals_load)[0];
    }
    
    __syncthreads();

    constexpr int kLPerThread = std::min(kChunkSizeL * kChunkSizeC / kNThreads, kChunkSizeL);
    static_assert(kLPerThread * kNThreads == kChunkSizeL * kChunkSizeC);
    constexpr int kNThreadsPerRow = kChunkSizeL / kLPerThread;
    static_assert(kNThreadsPerRow * kLPerThread == kChunkSizeL);
    // kChunkSizeL, kLPerThread, kNThreadsPerRow should be powers of 2 for simplicity
    static_assert((kChunkSizeL & (kChunkSizeL - 1)) == 0);
    static_assert((kLPerThread & (kLPerThread - 1)) == 0);
    static_assert((kNThreadsPerRow & (kNThreadsPerRow - 1)) == 0);
    static_assert(kNThreadsPerRow <= 32);

    const int row_idx = tid / kNThreadsPerRow;
    const int col_idx = tid % kNThreadsPerRow;


    if constexpr (kHasPosIdx) {
        if (row_idx == 0){
            #pragma unroll
            for (int l = 0; l < kLPerThread; ++l) {
                int idx = col_idx * kLPerThread + l;
                index_smem[col_idx * kLPerThread + l] = chunk_l_id * kChunkSizeL + idx < params.seqlen ? position_indices[idx] : 0;
            }
        }
        __syncthreads();
    }
    

    float bias_val = params.bias_ptr == nullptr || chunk_c_id * kChunkSizeC + row_idx >= params.dim ? 0.f : float(reinterpret_cast<weight_t *>(params.bias_ptr)[chunk_c_id * kChunkSizeC + row_idx]);
    float weight_vals[kWidth] = {0};
    if (chunk_c_id * kChunkSizeC + row_idx < params.dim) {
        #pragma unroll
        for (int w = 0; w < kWidth; ++w) {
            weight_vals[w] = weight[row_idx * params.weight_c_stride + w * params.weight_width_stride];
        }
    }
    float x_vals[kWidth - 1 + kLPerThread];
    #pragma unroll
    for (int i = 0; i < kWidth - 1 + kLPerThread; ++i) {
        x_vals[i] = float(x_smem[col_idx * kLPerThread + i][row_idx]);
    }

    float out_vals[kLPerThread];
    #pragma unroll
    for (int i = 0; i < kLPerThread; ++i) {
        out_vals[i] = bias_val;

        for (int w=0; w < kWidth; ++w) {
            if constexpr (kHasPosIdx){
                out_vals[i] += ((index_smem[col_idx * kLPerThread + i]==1 && w == 0 ) || (index_smem[col_idx * kLPerThread + i]==-1 && w == 2 ) ) ? 0.f : weight_vals[w] * x_vals[i + w];
            } else {
                out_vals[i] += weight_vals[w] * x_vals[i + w];
            }
        }
        if (params.silu_activation) {out_vals[i] = out_vals[i] / (1 + expf(-out_vals[i])); }
    }

    __syncthreads();
    #pragma unroll
    for (int i = 0; i < kLPerThread; ++i) { x_smem[col_idx * kLPerThread + i][row_idx] = out_vals[i]; }
    __syncthreads();

    #pragma unroll
    for (int l = 0; l < Ktraits::kNLoads; ++l) {
        input_t out_vals_store[kNElts];
        reinterpret_cast<vec_t *>(out_vals_store)[0] = reinterpret_cast<vec_t *>(x_smem[l * kLPerLoad + l_idx])[c_idx];
        if (chunk_l_id * kChunkSizeL + l * kLPerLoad + l_idx < params.seqlen
            && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
            *reinterpret_cast<vec_t *>(out + l * kLPerLoad * params.out_l_stride) = reinterpret_cast<vec_t *>(out_vals_store)[0];
        }
    }

}

template<int kNThreads, int kWidth, typename input_t, typename weight_t>
void non_causal_conv1d_channellast_fwd_launch(ConvParamsBase &params, cudaStream_t stream) {
    BOOL_SWITCH(params.position_indices_ptr != nullptr, kHasPosIdx, [&] {
        using Ktraits = non_causal_conv1d_channellast_fwd_kernel_traits<kNThreads, kWidth, 64, true, input_t, weight_t>;
        constexpr int kChunkSizeL = Ktraits::kChunkSizeL;
        constexpr int kChunkSizeC = Ktraits::kNEltsPerRow;
        const int n_chunks_L = (params.seqlen + kChunkSizeL - 1) / kChunkSizeL;
        const int n_chunks_C = (params.dim + kChunkSizeC - 1) / kChunkSizeC;
        dim3 grid(params.batch, n_chunks_L, n_chunks_C);
        dim3 block(Ktraits::kNThreads);
        auto kernel = &non_causal_conv1d_channellast_fwd_kernel<Ktraits, kHasPosIdx>;

        kernel<<<grid, Ktraits::kNThreads, 0, stream>>>(params);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

template<typename input_t, typename weight_t>
void non_causal_conv1d_channellast_fwd_cuda(ConvParamsBase &params, cudaStream_t stream) {
    switch (params.num_threads) {
        case 64:
            non_causal_conv1d_channellast_fwd_launch<64, 3, input_t, weight_t>(params, stream); break;
        case 128:
            non_causal_conv1d_channellast_fwd_launch<128, 3, input_t, weight_t>(params, stream); break;
        case 256:
            non_causal_conv1d_channellast_fwd_launch<256, 3, input_t, weight_t>(params, stream); break;
        case 512:
            non_causal_conv1d_channellast_fwd_launch<512, 3, input_t, weight_t>(params, stream); break;
        default:
            non_causal_conv1d_channellast_fwd_launch<128, 3, input_t, weight_t>(params, stream); break;
    }
}

template void non_causal_conv1d_fwd_cuda<float, float>(ConvParamsBase &params, cudaStream_t stream);
template void non_causal_conv1d_fwd_cuda<at::Half, float>(ConvParamsBase &params, cudaStream_t stream);
template void non_causal_conv1d_fwd_cuda<at::BFloat16, float>(ConvParamsBase &params, cudaStream_t stream);
template void non_causal_conv1d_fwd_cuda<float, at::Half>(ConvParamsBase &params, cudaStream_t stream);
template void non_causal_conv1d_fwd_cuda<at::Half, at::Half>(ConvParamsBase &params, cudaStream_t stream);
template void non_causal_conv1d_fwd_cuda<at::BFloat16, at::Half>(ConvParamsBase &params, cudaStream_t stream);
template void non_causal_conv1d_fwd_cuda<float, at::BFloat16>(ConvParamsBase &params, cudaStream_t stream);
template void non_causal_conv1d_fwd_cuda<at::Half, at::BFloat16>(ConvParamsBase &params, cudaStream_t stream);
template void non_causal_conv1d_fwd_cuda<at::BFloat16, at::BFloat16>(ConvParamsBase &params, cudaStream_t stream);

template void non_causal_conv1d_channellast_fwd_cuda<float, float>(ConvParamsBase &params, cudaStream_t stream);
template void non_causal_conv1d_channellast_fwd_cuda<at::Half, float>(ConvParamsBase &params, cudaStream_t stream);
template void non_causal_conv1d_channellast_fwd_cuda<at::BFloat16, float>(ConvParamsBase &params, cudaStream_t stream);
template void non_causal_conv1d_channellast_fwd_cuda<float, at::Half>(ConvParamsBase &params, cudaStream_t stream);
template void non_causal_conv1d_channellast_fwd_cuda<at::Half, at::Half>(ConvParamsBase &params, cudaStream_t stream);
template void non_causal_conv1d_channellast_fwd_cuda<at::BFloat16, at::Half>(ConvParamsBase &params, cudaStream_t stream);
template void non_causal_conv1d_channellast_fwd_cuda<float, at::BFloat16>(ConvParamsBase &params, cudaStream_t stream);
template void non_causal_conv1d_channellast_fwd_cuda<at::Half, at::BFloat16>(ConvParamsBase &params, cudaStream_t stream);
template void non_causal_conv1d_channellast_fwd_cuda<at::BFloat16, at::BFloat16>(ConvParamsBase &params, cudaStream_t stream);
//