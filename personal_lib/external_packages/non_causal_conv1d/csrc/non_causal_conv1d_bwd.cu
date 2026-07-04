/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Modifications Copyright (c) 2026, Sony Group Corporation.
 ******************************************************************************/

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>

#include "non_causal_conv1d.h"
#include "non_causal_conv1d_common.h"
#include "static_switch.h"

template<int kNThreads_, int kWidth_, bool kSiluAct_, bool kHasPosIdx_, bool kIsVecLoad_, typename input_t_, typename weight_t_>
struct non_causal_conv1d_bwd_kernel_traits {
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kWidth = kWidth_;
    static constexpr bool kSiluAct = kSiluAct_;
    static constexpr bool kHasPosIdx = kHasPosIdx_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : 8;
    static_assert(kWidth <= kNElts);
    // It's possible that we need to do 2 rounds of exchange if input_t is 16 bits
    // (since then we'd have 8 values of float, and each round we can exchange 4 floats).
    static constexpr int kNExchangeRounds = sizeof(float) / sizeof(input_t);
    static constexpr bool kIsVecLoad = kIsVecLoad_;
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNElts, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, 1, cub::BLOCK_LOAD_DIRECT>;
    using BlockLoadIndexT = cub::BlockLoad<int, kNThreads, kNElts, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadIndexVecT = cub::BlockLoad<int4, kNThreads, kNExchangeRounds,!(kIsVecLoad && kNExchangeRounds == 1) ? cub::BLOCK_LOAD_WARP_TRANSPOSE : cub::BLOCK_LOAD_DIRECT>;
    using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNElts, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, 1, cub::BLOCK_STORE_DIRECT>;
    using BlockReduceFloatT = cub::BlockReduce<float, kNThreads>;
    static constexpr int kSmemIOSize = (kIsVecLoad && kNExchangeRounds == 1)
        ? 0
        : std::max({sizeof(typename BlockLoadT::TempStorage), 
                    sizeof(typename BlockStoreT::TempStorage),
                    sizeof(typename BlockLoadIndexT::TempStorage),
                    sizeof(typename BlockLoadIndexVecT::TempStorage)});
    static constexpr int kSmemExchangeSize = kNThreads * kNBytes * kNElts * (1 + (kSiluAct ? kNExchangeRounds : 0) + (kHasPosIdx ? kNExchangeRounds : 0));
    static constexpr int kSmemSize = std::max({kSmemExchangeSize,
            int(sizeof(typename BlockReduceFloatT::TempStorage))}) + kSmemIOSize;
};

template<typename Ktraits, bool kHasPosIdx>
__global__ __launch_bounds__(Ktraits::kNThreads)
void non_causal_conv1d_bwd_kernel(ConvParamsBwd params) {
    constexpr int kWidth = Ktraits::kWidth;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr bool kSiluAct = Ktraits::kSiluAct;
    static constexpr int kNElts = Ktraits::kNElts;
    constexpr int kNExchangeRounds = Ktraits::kNExchangeRounds;
    static constexpr bool kIsVecLoad = Ktraits::kIsVecLoad;
    using input_t = typename Ktraits::input_t;
    using vec_t = typename Ktraits::vec_t;
    using weight_t = typename Ktraits::weight_t;

    // Shared memory.
    extern __shared__ char smem_[];
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_load_vec = reinterpret_cast<typename Ktraits::BlockLoadVecT::TempStorage&>(smem_);
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_store_vec = reinterpret_cast<typename Ktraits::BlockStoreVecT::TempStorage&>(smem_);
    auto& smem_load_index = reinterpret_cast<typename Ktraits::BlockLoadIndexT::TempStorage&>(smem_);
    auto& smem_load_index_vec = reinterpret_cast<typename Ktraits::BlockLoadIndexVecT::TempStorage&>(smem_);
    
    vec_t *smem_exchange = reinterpret_cast<vec_t *>(smem_ + Ktraits::kSmemIOSize);
    vec_t *smem_exchange_x = reinterpret_cast<vec_t *>(smem_ + Ktraits::kSmemIOSize) + kNThreads * kNExchangeRounds;
    vec_t *smem_exchange_index = reinterpret_cast<vec_t *>(smem_ + Ktraits::kSmemIOSize) + kNThreads * (kNExchangeRounds + 1);
    auto& smem_reduce_float = *reinterpret_cast<typename Ktraits::BlockReduceFloatT::TempStorage*>(smem_ + Ktraits::kSmemIOSize);

    const int tidx = threadIdx.x;
    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride
        + dim_id * params.x_c_stride;
    weight_t *weight = reinterpret_cast<weight_t *>(params.weight_ptr) + dim_id * params.weight_c_stride;
    input_t *dout = reinterpret_cast<input_t *>(params.dout_ptr) + batch_id * params.dout_batch_stride
        + dim_id * params.dout_c_stride;
    input_t *dx = reinterpret_cast<input_t *>(params.dx_ptr) + batch_id * params.dx_batch_stride
        + dim_id * params.dx_c_stride;
    float *dweight = reinterpret_cast<float *>(params.dweight_ptr) + dim_id * params.dweight_c_stride;
    float bias_val = params.bias_ptr == nullptr ? 0.f : float(reinterpret_cast<weight_t *>(params.bias_ptr)[dim_id]);
    int *position_indices = !kHasPosIdx ? nullptr : reinterpret_cast<int *>(params.position_indices_ptr) + batch_id * params.seqlen;
    // Thread kNThreads - 1 will load the first elements of the next chunk so we initialize those to 0.
    if (tidx == 0) {
        
        if constexpr (!kSiluAct) {
            input_t zeros[kNElts] = {0};
            smem_exchange[0] = reinterpret_cast<vec_t *>(zeros)[0];
        } else {
            float zeros[kNElts] = {0};
            #pragma unroll
            for (int r = 0; r < kNExchangeRounds; ++r) {
                smem_exchange[r * kNThreads] = reinterpret_cast<vec_t *>(zeros)[r];
            }
            float zeros_x[kNElts] = {0};
            smem_exchange_x[0] = reinterpret_cast<vec_t *>(zeros_x)[0];
        }
        if constexpr (kHasPosIdx) {
            int zeros[kNElts] = {0};
            #pragma unroll
            for (int r = 0; r < kNExchangeRounds; ++r) {
                smem_exchange_index[r * kNThreads] = reinterpret_cast<vec_t *>(zeros)[r];
            }
        }
    }
    
    float weight_vals[kWidth];
    #pragma unroll
    for (int i = 0; i < kWidth; ++i) { weight_vals[i] = weight[i * params.weight_width_stride]; }
        
    float dweight_vals[kWidth] = {0};
    float dbias_val = 0;

    constexpr int kChunkSize = kNThreads * kNElts;
    const int n_chunks = (params.seqlen + kChunkSize - 1) / kChunkSize;
    x += (n_chunks - 1) * kChunkSize;
    dout += (n_chunks - 1) * kChunkSize;
    dx += (n_chunks - 1) * kChunkSize;
    if constexpr (kHasPosIdx)   position_indices += (n_chunks - 1) * kChunkSize;
    for (int chunk = n_chunks - 1; chunk >= 0; --chunk) {
        input_t x_vals_load[3 * kNElts] = {0};
        input_t x_vals_load_begin[kNElts] = {0};

        input_t dout_vals_load[3 * kNElts] = {0};
        int position_indices_load[3 * kNElts] = {0};
        if constexpr(kIsVecLoad) {
            __syncthreads();
            Ktraits::BlockLoadVecT(smem_load_vec).Load(reinterpret_cast<vec_t*>(x), *reinterpret_cast<vec_t (*)[1]>(&x_vals_load[kNElts]), (params.seqlen - chunk * kChunkSize) / kNElts);
            Ktraits::BlockLoadVecT(smem_load_vec).Load(reinterpret_cast<vec_t*>(dout), *reinterpret_cast<vec_t (*)[1]>(&dout_vals_load[kNElts]), (params.seqlen - chunk * kChunkSize) / kNElts);
            if constexpr (kHasPosIdx)
                Ktraits::BlockLoadIndexVecT(smem_load_index_vec).Load(reinterpret_cast<int4*>(position_indices), *reinterpret_cast<int4(*)[Ktraits::kNExchangeRounds]>(&position_indices_load[kNElts]), (params.seqlen - chunk * kChunkSize) / kNElts * Ktraits::kNExchangeRounds);
        } else {
            __syncthreads();
            Ktraits::BlockLoadT(smem_load).Load(x, *reinterpret_cast<input_t (*)[kNElts]>(&x_vals_load[kNElts]), params.seqlen - chunk * kChunkSize);
            __syncthreads();
            Ktraits::BlockLoadT(smem_load).Load(dout, *reinterpret_cast<input_t (*)[kNElts]>(&dout_vals_load[kNElts]), params.seqlen - chunk * kChunkSize);
            if constexpr (kHasPosIdx)
                Ktraits::BlockLoadIndexT(smem_load_index).Load(position_indices, *reinterpret_cast<int (*)[kNElts]>(&position_indices_load[kNElts]), (params.seqlen - chunk * kChunkSize), 0);
        }
        __syncthreads();
        if (tidx == 0 && chunk > 0) {
            int valid_elements = params.seqlen - chunk * kChunkSize + kNElts;
            #pragma unroll
            for (int i = 0; i < kNElts; i++) {
                if (i < valid_elements) {
                    x_vals_load_begin[i] = x[i - kNElts];
                }
            }
            #pragma unroll
            for (int i = 0; i < kNElts; i++) {
                if (i < valid_elements) {
                    dout_vals_load[i] = dout[i - kNElts];
                }
            }
            if constexpr (kHasPosIdx) {
                #pragma unroll
                for (int i = 0; i < kNElts; i++) {
                    if (i < valid_elements) {
                        position_indices_load[i] = position_indices[i - kNElts];
                    }
                }
            }

        }

        float dout_vals[3 * kNElts], x_vals[3 * kNElts];
        if constexpr (!kSiluAct) {
            __syncthreads();
            // Thread 0 don't write yet, so that thread kNThreads - 1 can read
            // the first elements of the next chunk.
            if (tidx > 0) { smem_exchange[tidx] = reinterpret_cast<vec_t *>(dout_vals_load)[1]; }
            __syncthreads();

            if (tidx > 1) {
                reinterpret_cast<vec_t *>(dout_vals_load)[0] = smem_exchange[tidx - 1];
            }
            reinterpret_cast<vec_t *>(dout_vals_load)[2] = smem_exchange[tidx < kNThreads - 1 ? tidx + 1 : 0];
            __syncthreads();
            // Now thread 0 can write the first elements of the current chunk.
            if (tidx == 0) { smem_exchange[tidx] = reinterpret_cast<vec_t *>(dout_vals_load)[1]; }
            __syncthreads();
            if (tidx == 1) {
                reinterpret_cast<vec_t *>(dout_vals_load)[0] = smem_exchange[tidx - 1];
            }
            
            #pragma unroll
            for (int i = 0; i < 3 * kNElts; ++i) {
                dout_vals[i] = float(dout_vals_load[i]);
                x_vals[i] = float(x_vals_load[i]);
            }
        } else {
            if (tidx == 0) {
                if constexpr(kIsVecLoad) {
                    if (chunk > 0) {
                        reinterpret_cast<vec_t *>(x_vals_load)[0] = reinterpret_cast<vec_t *>(x)[-1];
                    }
                } else {
                    if (chunk > 0) {
                        #pragma unroll
                        for (int i = 0; i < kNElts; ++i) {
                            if (chunk * kChunkSize + i < params.seqlen) { x_vals_load[i] = x[-kNElts + i]; }
                        }
                    }
                }
            } 
            __syncthreads();
            if (tidx == kNThreads - 1) { reinterpret_cast<vec_t *>(x_vals_load)[2] = smem_exchange_x[0]; }
            __syncthreads();
            smem_exchange_x[tidx] = reinterpret_cast<vec_t *>(x_vals_load)[1];
            __syncthreads();
            if (tidx > 0) { reinterpret_cast<vec_t *>(x_vals_load)[0] = smem_exchange_x[tidx - 1]; }
            if (tidx < kNThreads - 1) { reinterpret_cast<vec_t *>(x_vals_load)[2] = smem_exchange_x[tidx + 1]; }
            
            if constexpr (kHasPosIdx){
                // Exchange the indices. It's possible that we need to do 2 rounds of exchange
                // if input_t is 16 bits (since then we'd have 8 values of float)
                __syncthreads();
                if (tidx == kNThreads - 1) {
                    #pragma unroll
                    for (int r = 0; r < kNExchangeRounds; ++r) {
                        reinterpret_cast<vec_t *>(position_indices_load)[2*kNExchangeRounds + r]
                            = smem_exchange_index[r * kNThreads];
                    }
                }
                __syncthreads();
                #pragma unroll
                for (int r = 0; r < kNExchangeRounds; ++r) {
                    smem_exchange_index[r * kNThreads + tidx] = reinterpret_cast<vec_t *>(position_indices_load)[kNExchangeRounds+r]; 
                }
                __syncthreads();
                if (tidx > 0) {
                    #pragma unroll
                    for (int r = 0; r < kNExchangeRounds; ++r) {
                        reinterpret_cast<vec_t *>(position_indices_load)[r]
                            = smem_exchange_index[r * kNThreads + (tidx - 1)];
                    }
                }
                if (tidx < kNThreads - 1) {
                    #pragma unroll
                    for (int r = 0; r < kNExchangeRounds; ++r) {
                        reinterpret_cast<vec_t *>(position_indices_load)[2*kNExchangeRounds + r]
                            = smem_exchange_index[r * kNThreads + tidx + 1];
                    }
                }
            }

            #pragma unroll
            for (int i = 0; i < 3 * kNElts; ++i) { x_vals[i] = float(x_vals_load[i]); }
            // Recompute the output
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) {
                float out_val = bias_val;

                int w = 0;
                int w_end = kWidth;
                if constexpr (kHasPosIdx){
                    if (position_indices_load[kNElts+i] == 1){
                        w = 1;
                    } else if (position_indices_load[kNElts+i] == -1){
                        w_end = kWidth -1;
                    }
                }
                for (; w < w_end; ++w) {
                    out_val += weight_vals[w] * x_vals[kNElts + i - (kWidth/2 - w)];
                }
                float out_sigmoid_val = 1.0f / (1.0f + expf(-out_val));
                dout_vals[kNElts+i] = float(dout_vals_load[kNElts+i]) * out_sigmoid_val
                               * (1.0f + out_val * (1.0f - out_sigmoid_val));
            }

            if (tidx == 0) {
                float x_vals_begin[kNElts];
                #pragma unroll
                for (int i = 0; i < kNElts; ++i) {
                    x_vals_begin[i] = float(x_vals_load_begin[i]);
                }
                #pragma unroll
                for (int i = -kNElts; i < 0; ++i) {
                    float out_val = bias_val;

                    int w = 0;
                    int w_end = kWidth;
                    if constexpr (kHasPosIdx){
                        if (position_indices_load[kNElts+i] == 1){
                            w = 1;
                        } else if (position_indices_load[kNElts+i] == -1){
                            w_end = kWidth -1;
                        }
                    }
                    for (; w < w_end; ++w) {
                        const int x_ind = kNElts + i - (kWidth/2 - w);
                        if (x_ind >= 0) {
                            out_val += weight_vals[w] * x_vals[x_ind];
                        } else{
                            out_val += weight_vals[w] * x_vals_begin[x_ind+kNElts];
                        }
                        
                    }
                    float out_sigmoid_val = 1.0f / (1.0f + expf(-out_val));
                    dout_vals[kNElts+i] = float(dout_vals_load[kNElts+i]) * out_sigmoid_val
                                * (1.0f + out_val * (1.0f - out_sigmoid_val));
                }
            }

            // Exchange the dout_vals. It's possible that we need to do 2 rounds of exchange
            // if input_t is 16 bits (since then we'd have 8 values of float)
            __syncthreads();
            // Thread 0 don't write yet, so that thread kNThreads - 1 can read
            // the first elements of the next chunk.

            if (tidx > 0) {
                #pragma unroll
                for (int r = 0; r < kNExchangeRounds; ++r) {
                    smem_exchange[r * kNThreads + tidx] = reinterpret_cast<vec_t *>(dout_vals)[kNExchangeRounds+r];
                }
            }
            __syncthreads();
    
            if (tidx > 1) {
                #pragma unroll
                for (int r = 0; r < kNExchangeRounds; ++r) {
                    reinterpret_cast<vec_t *>(dout_vals)[r]
                        = smem_exchange[r * kNThreads + (tidx - 1)];
                }
            }
            #pragma unroll
            for (int r = 0; r < kNExchangeRounds; ++r) {
                reinterpret_cast<vec_t *>(dout_vals)[2*kNExchangeRounds + r]
                    = smem_exchange[r * kNThreads + (tidx < kNThreads - 1 ? tidx + 1 : 0)];
            }
            __syncthreads();
            // Now thread 0 can write the first elements of the current chunk.
            if (tidx == 0) {
                #pragma unroll
                for (int r = 0; r < kNExchangeRounds; ++r) {
                    smem_exchange[r * kNThreads + tidx] = reinterpret_cast<vec_t *>(dout_vals)[kNExchangeRounds+r];
                }
            }
            __syncthreads();
            if (tidx == 1) {
                #pragma unroll
                for (int r = 0; r < kNExchangeRounds; ++r) {
                    reinterpret_cast<vec_t *>(dout_vals)[r]
                        = smem_exchange[r * kNThreads + (tidx - 1)];
                }
            }
        }


        if constexpr (kHasPosIdx){
            position_indices -=  kChunkSize;
        }
        dout -= kChunkSize;
        x -= kChunkSize; 

        #pragma unroll
        for (int i = 0; i < kNElts; ++i) { dbias_val += dout_vals[kNElts + i]; }

        float dx_vals[kNElts] = {0};
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) {
            int w = 0;
            int w_end = kWidth;
            if constexpr (kHasPosIdx){
                if (position_indices_load[kNElts+i] == 1){
                    w_end = kWidth -1;
                } else if (position_indices_load[kNElts+i] == -1){
                    w = 1;
                }
            }
            for (; w < w_end; ++w) {
                dx_vals[i] += weight_vals[w] * dout_vals[kNElts + i + (kWidth/2 - w)];
            }
        }

        input_t dx_vals_store[kNElts];
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) { dx_vals_store[i] = dx_vals[i]; }
        if constexpr(kIsVecLoad) {
            Ktraits::BlockStoreVecT(smem_store_vec).Store(reinterpret_cast<vec_t*>(dx), reinterpret_cast<vec_t (&)[1]>(dx_vals_store), (params.seqlen - chunk * kChunkSize) / kNElts);
        } else {
            Ktraits::BlockStoreT(smem_store).Store(dx, dx_vals_store, params.seqlen - chunk * kChunkSize);
        }
        dx -= kChunkSize;
        
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) {
            int w = 0;
            int w_end = kWidth;
            if constexpr (kHasPosIdx){
                if (position_indices_load[kNElts+i] == 1){
                    w_end = kWidth -1;
                } else if (position_indices_load[kNElts+i] == -1){
                    w = 1;
                }
            }
            for (; w < w_end; ++w) {
                dweight_vals[w] += x_vals[kNElts + i] * dout_vals[kNElts + i + (kWidth/2 - w)];
            }
        }
    }

    #pragma unroll
    for (int w = 0; w < kWidth; ++w) {
        __syncthreads();
        dweight_vals[w] = Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(dweight_vals[w]);
        if (tidx == 0) {
            atomicAdd(&reinterpret_cast<float *>(dweight)[w * params.dweight_width_stride], dweight_vals[w]);
        }
    }
    if (params.bias_ptr != nullptr) {
        __syncthreads();
        dbias_val = Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(dbias_val);
        if (tidx == 0) {
            atomicAdd(&reinterpret_cast<float *>(params.dbias_ptr)[dim_id], dbias_val);
        }
    }
}

template<int kNThreads, int kWidth, typename input_t, typename weight_t>
void non_causal_conv1d_bwd_launch(ConvParamsBwd &params, cudaStream_t stream) {
    static constexpr int kNElts = sizeof(input_t) == 4 ? 4 : 8;
    BOOL_SWITCH(params.position_indices_ptr != nullptr, kHasPosIdx, [&] {
        BOOL_SWITCH(params.seqlen % kNElts == 0, kIsVecLoad, [&] {
            BOOL_SWITCH(params.silu_activation, kSiluAct, [&] {
                using Ktraits = non_causal_conv1d_bwd_kernel_traits<kNThreads, kWidth, kSiluAct, kHasPosIdx, kIsVecLoad, input_t, weight_t>;
                constexpr int kSmemSize = Ktraits::kSmemSize;
                dim3 grid(params.batch, params.dim);
                auto kernel = &non_causal_conv1d_bwd_kernel<Ktraits, kHasPosIdx>;
                if (kSmemSize >= 48 * 1024) {
                    C10_CUDA_CHECK(cudaFuncSetAttribute(
                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
                    }
                kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
    });
}

template<typename input_t, typename weight_t>
void non_causal_conv1d_bwd_cuda(ConvParamsBwd &params, cudaStream_t stream) {
    // params.num_threadsをテンプレート引数に使う
    switch (params.num_threads) {
        case 128:
            non_causal_conv1d_bwd_launch<128, 3, input_t, weight_t>(params, stream); break;
        case 256:
            non_causal_conv1d_bwd_launch<256, 3, input_t, weight_t>(params, stream); break;
        default:
            non_causal_conv1d_bwd_launch<128, 3, input_t, weight_t>(params, stream); break;
    }
}

template<int kNThreads_, int kWidth_, int kChunkSizeL_, bool kSiluAct_, bool kIsVecLoad_, typename input_t_, typename weight_t_>
struct non_causal_conv1d_channellast_bwd_kernel_traits {
    // The cache line is 128 bytes, and we try to read 16 bytes per thread.
    // So we have 8 threads per "row", so 32 or 64 elements in the channel dimension.
    // That leaves 4 columns per warp, and so 16 columns per block (assuming each block has 128
    // threads). Each each load is 16 x 32|64 elements in the L x C dimensions.
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr bool kSiluAct = kSiluAct_;
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
void non_causal_conv1d_channellast_bwd_kernel(ConvParamsBwd params) {
    constexpr int kWidth = Ktraits::kWidth;
    constexpr int kWidth_div = Ktraits::kWidth/2;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr bool kSiluAct = Ktraits::kSiluAct;
    constexpr int kNElts = Ktraits::kNElts;
    constexpr int kNWarp = Ktraits::kNWarps;
    constexpr int kNThreadsPerC = Ktraits::kNThreadsPerRow;
    constexpr int kLPerLoad = Ktraits::kNColsPerLoad;
    constexpr int kChunkSizeL = Ktraits::kChunkSizeL;
    constexpr int kChunkSizeC = Ktraits::kNEltsPerRow;
    using input_t = typename Ktraits::input_t;
    using vec_t = typename Ktraits::vec_t;
    using weight_t = typename Ktraits::weight_t;

    // Shared memory.
    __shared__ input_t dout_smem[kChunkSizeL + 2*kWidth_div][kChunkSizeC + kNElts];
    __shared__ input_t x_smem[kChunkSizeL + 4*kWidth_div][kChunkSizeC + kNElts];

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
    input_t *dout = reinterpret_cast<input_t *>(params.dout_ptr) + batch_id * params.dout_batch_stride
        + (chunk_l_id * kChunkSizeL + l_idx) * params.dout_l_stride + chunk_c_id * kChunkSizeC + c_idx * kNElts;
    input_t *dx = reinterpret_cast<input_t *>(params.dx_ptr) + batch_id * params.dx_batch_stride
        + (chunk_l_id * kChunkSizeL + l_idx) * params.dx_l_stride + chunk_c_id * kChunkSizeC + c_idx * kNElts;
    float *dweight = reinterpret_cast<float *>(params.dweight_ptr)
        + chunk_c_id * kChunkSizeC * params.dweight_c_stride;
    int *position_indices = !kHasPosIdx ? nullptr : reinterpret_cast<int *>(params.position_indices_ptr)
        + batch_id * params.seqlen + chunk_l_id * kChunkSizeL;

    #pragma unroll
    for (int l = 0; l < Ktraits::kNLoads; ++l) {
        input_t dout_vals_load[kNElts] = {0};
        input_t x_vals_load[kNElts] = {0};
        if (chunk_l_id * kChunkSizeL + l * kLPerLoad + l_idx < params.seqlen
            && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
            reinterpret_cast<vec_t *>(dout_vals_load)[0] = *reinterpret_cast<vec_t *>(dout + l * kLPerLoad * params.dout_l_stride);
            reinterpret_cast<vec_t *>(x_vals_load)[0] = *reinterpret_cast<vec_t *>(x + l * kLPerLoad * params.x_l_stride);
        }
        reinterpret_cast<vec_t *>(dout_smem[kWidth_div + l * kLPerLoad + l_idx])[c_idx] = reinterpret_cast<vec_t *>(dout_vals_load)[0];
        reinterpret_cast<vec_t *>(x_smem[kWidth_div*2 + l * kLPerLoad + l_idx])[c_idx] = reinterpret_cast<vec_t *>(x_vals_load)[0];
    }
    // Load the elements from the previous chunk or next chunk that are needed for convolution.
    if (l_idx < kWidth_div) {
        input_t dout_vals_load[kNElts] = {0};
        input_t x_vals_load[kNElts] = {0};
        if (chunk_l_id * kChunkSizeL + l_idx - kWidth_div >= 0
            && chunk_l_id * kChunkSizeL + l_idx - kWidth_div < params.seqlen
            && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
            reinterpret_cast<vec_t *>(dout_vals_load)[0] = *reinterpret_cast<vec_t *>(dout - kWidth_div * params.dout_l_stride);
            reinterpret_cast<vec_t *>(x_vals_load)[0] = *reinterpret_cast<vec_t *>(x - kWidth_div * params.x_l_stride);
        }
        reinterpret_cast<vec_t *>(dout_smem[l_idx])[c_idx] = reinterpret_cast<vec_t *>(dout_vals_load)[0];
        reinterpret_cast<vec_t *>(x_smem[kWidth_div+l_idx])[c_idx] = reinterpret_cast<vec_t *>(x_vals_load)[0];
    }
    if (l_idx >= kChunkSizeL / kLPerLoad - kWidth_div) { // Load the elements from the future chunk that are needed for convolution.
        input_t dout_vals_load[kNElts] = {0};
        input_t x_vals_load[kNElts] = {0};
        if (chunk_l_id * kChunkSizeL + l_idx + (Ktraits::kNLoads-1) * kLPerLoad + kWidth_div >= 0
            && chunk_l_id * kChunkSizeL + l_idx + (Ktraits::kNLoads-1) * kLPerLoad + kWidth_div < params.seqlen
            && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
            reinterpret_cast<vec_t *>(dout_vals_load)[0] = *reinterpret_cast<vec_t *>(dout + ((Ktraits::kNLoads-1) * kLPerLoad + kWidth_div) * params.dout_l_stride);
            reinterpret_cast<vec_t *>(x_vals_load)[0] = *reinterpret_cast<vec_t *>(x + ((Ktraits::kNLoads-1) * kLPerLoad + kWidth_div) * params.x_l_stride);
        }
        reinterpret_cast<vec_t *>(dout_smem[kWidth_div*2 + (Ktraits::kNLoads-1) * kLPerLoad + l_idx])[c_idx] = reinterpret_cast<vec_t *>(dout_vals_load)[0];
        reinterpret_cast<vec_t *>(x_smem[kWidth_div*3 + (Ktraits::kNLoads-1) * kLPerLoad + l_idx])[c_idx] = reinterpret_cast<vec_t *>(x_vals_load)[0];
    }

    // Need to load (kWdith - 1) extra x's on the right to recompute the (kChunkSizeL + kWidth - 1) outputs
    if constexpr (kSiluAct) {
        if (l_idx < kWidth_div) {
            input_t x_vals_load[kNElts] = {0};
            if (chunk_l_id * kChunkSizeL + l_idx - 2*kWidth_div >= 0
                && chunk_l_id * kChunkSizeL + l_idx - 2*kWidth_div < params.seqlen
                && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
                reinterpret_cast<vec_t *>(x_vals_load)[0] = *reinterpret_cast<vec_t *>(x - 2*kWidth_div * params.x_l_stride);
            }
            reinterpret_cast<vec_t *>(x_smem[l_idx])[c_idx] = reinterpret_cast<vec_t *>(x_vals_load)[0];
        }
        if (l_idx >= kChunkSizeL / kLPerLoad - kWidth_div) { // Load the elements from the future chunk that are needed for convolution.
            input_t x_vals_load[kNElts] = {0};
            if (chunk_l_id * kChunkSizeL + l_idx + (Ktraits::kNLoads-1) * kLPerLoad + 2*kWidth_div >= 0
                && chunk_l_id * kChunkSizeL + l_idx + (Ktraits::kNLoads-1) * kLPerLoad + 2*kWidth_div < params.seqlen
                && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
                reinterpret_cast<vec_t *>(x_vals_load)[0] = *reinterpret_cast<vec_t *>(x + ((Ktraits::kNLoads-1) * kLPerLoad + 2*kWidth_div) * params.x_l_stride);
            }
            reinterpret_cast<vec_t *>(x_smem[kWidth_div*4 + (Ktraits::kNLoads-1) * kLPerLoad + l_idx])[c_idx] = reinterpret_cast<vec_t *>(x_vals_load)[0];
        }
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

    float bias_val = params.bias_ptr == nullptr || chunk_c_id * kChunkSizeC + row_idx >= params.dim ? 0.f : float(reinterpret_cast<weight_t *>(params.bias_ptr)[chunk_c_id * kChunkSizeC + row_idx]);
    float weight_vals[kWidth] = {0};
    if (chunk_c_id * kChunkSizeC + row_idx < params.dim) {
        #pragma unroll
        for (int w = 0; w < kWidth; ++w) {
            weight_vals[w] = weight[row_idx * params.weight_c_stride + w * params.weight_width_stride];
        }
    }
    float dout_vals[kLPerThread + 2*kWidth_div];
    float x_vals[kLPerThread + 4*kWidth_div];
    #pragma unroll
    for (int i = 0; i < 2*kWidth_div + kLPerThread; ++i) {
        dout_vals[i] = float(dout_smem[col_idx * kLPerThread + i][row_idx]);
    }

    #pragma unroll
    for (int i = 0; i < 4*kWidth_div + kLPerThread; ++i) {
        x_vals[i] = float(x_smem[col_idx * kLPerThread + i][row_idx]);
    }

    int position_indices_load[kLPerThread + 2*kWidth_div];
    if constexpr (kHasPosIdx) {
        #pragma unroll
        for (int i = 0; i < kWidth - 1 + kLPerThread; ++i) {
            const int l_idx = chunk_l_id * kChunkSizeL + col_idx * kLPerThread - kWidth_div + i;
            position_indices_load[i] = l_idx >= 0 && l_idx < params.seqlen ? position_indices[col_idx * kLPerThread - kWidth_div + i] : 0;
        }
    }

    if constexpr (kSiluAct) {  // Recompute the output
        #pragma unroll
        for (int i = 0; i < kLPerThread + 2*kWidth_div; ++i) {
            float out_val = bias_val;
            for (int w=0; w < kWidth; ++w) {
                if constexpr (kHasPosIdx){
                    out_val += ((position_indices_load[i]==1 && w == 0 ) || (position_indices_load[i]==-1 && w == 2 ) ) ? 0.f : weight_vals[w] * x_vals[i + w];
                } else {
                    out_val += weight_vals[w] * x_vals[i + w];
                }
            }
            float out_val_sigmoid = 1.f / (1.f + expf(-out_val));
            dout_vals[i] *= out_val_sigmoid * (1 + out_val * (1 - out_val_sigmoid));
        }
    }

    float dweight_vals[kWidth] = {0};
    SumOp<float> sum_op;
    #pragma unroll
    for (int i = 0; i < kLPerThread; ++i) {
        for (int w=0; w < kWidth; ++w) {
            if constexpr (kHasPosIdx){
                dweight_vals[w] += ((position_indices_load[kWidth_div + i]==1 && w == 2 ) || (position_indices_load[kWidth_div + i]==-1 && w == 0 ) ) ? 0.f : x_vals[2*kWidth_div + i] * dout_vals[i+2*kWidth_div - w];
            } else {
                dweight_vals[w] += x_vals[2*kWidth_div + i] * dout_vals[i+2*kWidth_div - w];
            }
        }
        
    }

    #pragma unroll
    for (int w = 0; w < kWidth; ++w) {
        dweight_vals[w] = Allreduce<kNThreadsPerRow>::run(dweight_vals[w], sum_op);
        if (col_idx == 0 && chunk_c_id * kChunkSizeC + row_idx < params.dim) {
            atomicAdd(&reinterpret_cast<float *>(dweight)[row_idx * params.dweight_c_stride + w * params.dweight_width_stride], dweight_vals[w]);
        }
    }

    if (params.bias_ptr != nullptr) {
        float dbias_val = 0.f;
        #pragma unroll
        for (int i = 0; i < kLPerThread; ++i) { dbias_val += dout_vals[i+kWidth_div]; }
        dbias_val = Allreduce<kNThreadsPerRow>::run(dbias_val, sum_op);
        if (col_idx == 0 && chunk_c_id * kChunkSizeC + row_idx < params.dim) {
            atomicAdd(&reinterpret_cast<float *>(params.dbias_ptr)[chunk_c_id * kChunkSizeC + row_idx], dbias_val);
        }
    }

    float dx_vals[kLPerThread] = {0};
    #pragma unroll
    for (int i = 0; i < kLPerThread; ++i) {
        for (int w=0; w < kWidth; ++w) {
            if constexpr (kHasPosIdx){
                dx_vals[i] += ((position_indices_load[kWidth_div + i]==1 && w == 2 ) || (position_indices_load[kWidth_div + i]==-1 && w == 0 ) ) ? 0.f : weight_vals[w] * dout_vals[2*kWidth_div + i  - w];
            } else {
                dx_vals[i] += weight_vals[w] * dout_vals[2*kWidth_div + i  - w];
            }
        }
        
        
    }

    __syncthreads();
    #pragma unroll
    for (int i = 0; i < kLPerThread; ++i) { x_smem[kWidth - 1 + col_idx * kLPerThread + i][row_idx] = dx_vals[i]; }
    __syncthreads();

    #pragma unroll
    for (int l = 0; l < Ktraits::kNLoads; ++l) {
        input_t dx_vals_store[kNElts];
        reinterpret_cast<vec_t *>(dx_vals_store)[0] = reinterpret_cast<vec_t *>(x_smem[kWidth - 1 + l * kLPerLoad + l_idx])[c_idx];
        if (chunk_l_id * kChunkSizeL + l * kLPerLoad + l_idx < params.seqlen
            && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
            *reinterpret_cast<vec_t *>(dx + l * kLPerLoad * params.dx_l_stride) = reinterpret_cast<vec_t *>(dx_vals_store)[0];
        }
    }

}

template<int kNThreads, int kWidth, typename input_t, typename weight_t>
void non_causal_conv1d_channellast_bwd_launch(ConvParamsBwd &params, cudaStream_t stream) {
    BOOL_SWITCH(params.silu_activation, kSiluAct, [&] {
        BOOL_SWITCH(params.position_indices_ptr != nullptr, kHasPosIdx, [&] {
            BOOL_SWITCH(params.seqlen <= 128, kChunkSizeL64, [&] {
                // kChunkSizeL = 128 is slightly faster than 64 when seqlen is larger
                static constexpr int kChunk = kChunkSizeL64 ? 64 : 128;
                using Ktraits = non_causal_conv1d_channellast_bwd_kernel_traits<kNThreads, kWidth, kChunk, kSiluAct, true, input_t, weight_t>;
                constexpr int kChunkSizeL = Ktraits::kChunkSizeL;
                constexpr int kChunkSizeC = Ktraits::kNEltsPerRow;
                const int n_chunks_L = (params.seqlen + kChunkSizeL - 1) / kChunkSizeL;
                const int n_chunks_C = (params.dim + kChunkSizeC - 1) / kChunkSizeC;
                dim3 grid(params.batch, n_chunks_L, n_chunks_C);
                dim3 block(Ktraits::kNThreads);
                auto kernel = &non_causal_conv1d_channellast_bwd_kernel<Ktraits, kHasPosIdx>;
                kernel<<<grid, Ktraits::kNThreads, 0, stream>>>(params);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
    });
}

template<typename input_t, typename weight_t>
void non_causal_conv1d_channellast_bwd_cuda(ConvParamsBwd &params, cudaStream_t stream) {
    switch (params.num_threads) {
        case 128:
            non_causal_conv1d_channellast_bwd_launch<128, 3, input_t, weight_t>(params, stream); break;
        case 256:
            non_causal_conv1d_channellast_bwd_launch<256, 3, input_t, weight_t>(params, stream); break;
        default:
            non_causal_conv1d_channellast_bwd_launch<128, 3, input_t, weight_t>(params, stream); break;
    }
}

template void non_causal_conv1d_bwd_cuda<float, float>(ConvParamsBwd &params, cudaStream_t stream);
template void non_causal_conv1d_bwd_cuda<at::Half, float>(ConvParamsBwd &params, cudaStream_t stream);
template void non_causal_conv1d_bwd_cuda<at::BFloat16, float>(ConvParamsBwd &params, cudaStream_t stream);
template void non_causal_conv1d_bwd_cuda<float, at::Half>(ConvParamsBwd &params, cudaStream_t stream);
template void non_causal_conv1d_bwd_cuda<at::Half, at::Half>(ConvParamsBwd &params, cudaStream_t stream);
template void non_causal_conv1d_bwd_cuda<at::BFloat16, at::Half>(ConvParamsBwd &params, cudaStream_t stream);
template void non_causal_conv1d_bwd_cuda<float, at::BFloat16>(ConvParamsBwd &params, cudaStream_t stream);
template void non_causal_conv1d_bwd_cuda<at::Half, at::BFloat16>(ConvParamsBwd &params, cudaStream_t stream);
template void non_causal_conv1d_bwd_cuda<at::BFloat16, at::BFloat16>(ConvParamsBwd &params, cudaStream_t stream);

template void non_causal_conv1d_channellast_bwd_cuda<float, float>(ConvParamsBwd &params, cudaStream_t stream);
template void non_causal_conv1d_channellast_bwd_cuda<at::Half, float>(ConvParamsBwd &params, cudaStream_t stream);
template void non_causal_conv1d_channellast_bwd_cuda<at::BFloat16, float>(ConvParamsBwd &params, cudaStream_t stream);
template void non_causal_conv1d_channellast_bwd_cuda<float, at::Half>(ConvParamsBwd &params, cudaStream_t stream);
template void non_causal_conv1d_channellast_bwd_cuda<at::Half, at::Half>(ConvParamsBwd &params, cudaStream_t stream);
template void non_causal_conv1d_channellast_bwd_cuda<at::BFloat16, at::Half>(ConvParamsBwd &params, cudaStream_t stream);
template void non_causal_conv1d_channellast_bwd_cuda<float, at::BFloat16>(ConvParamsBwd &params, cudaStream_t stream);
template void non_causal_conv1d_channellast_bwd_cuda<at::Half, at::BFloat16>(ConvParamsBwd &params, cudaStream_t stream);
template void non_causal_conv1d_channellast_bwd_cuda<at::BFloat16, at::BFloat16>(ConvParamsBwd &params, cudaStream_t stream);
// //