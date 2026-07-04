/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Modifications Copyright (c) 2026, Sony Group Corporation.
 ******************************************************************************/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

#include "non_causal_conv1d.h"

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

#define DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(ITYPE, NAME, ...)                    \
    if (ITYPE == at::ScalarType::Half) {                                            \
        using input_t = at::Half;                                                   \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::BFloat16) {                                 \
        using input_t = at::BFloat16;                                               \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::Float)  {                                   \
        using input_t = float;                                                      \
        __VA_ARGS__();                                                              \
    } else {                                                                        \
        AT_ERROR(#NAME, " not implemented for input type '", toString(ITYPE), "'"); \
    }

#define DISPATCH_WTYPE_FLOAT_AND_HALF_AND_BF16(WTYPE, NAME, ...)                     \
    if (WTYPE == at::ScalarType::Half) {                                             \
        using weight_t = at::Half;                                                   \
        __VA_ARGS__();                                                               \
    } else if (WTYPE == at::ScalarType::BFloat16) {                                  \
        using weight_t = at::BFloat16;                                               \
        __VA_ARGS__();                                                               \
    } else if (WTYPE == at::ScalarType::Float)  {                                    \
        using weight_t = float;                                                      \
        __VA_ARGS__();                                                               \
    } else {                                                                         \
        AT_ERROR(#NAME, " not implemented for weight type '", toString(WTYPE), "'"); \
    }

template<typename input_t, typename weight_t>
void non_causal_conv1d_fwd_cuda(ConvParamsBase &params, cudaStream_t stream);
template <typename input_t, typename weight_t>
void non_causal_conv1d_channellast_fwd_cuda(ConvParamsBase &params, cudaStream_t stream);

template<typename input_t, typename weight_t>
void non_causal_conv1d_bwd_cuda(ConvParamsBwd &params, cudaStream_t stream);
template<typename input_t, typename weight_t>
void non_causal_conv1d_channellast_bwd_cuda(ConvParamsBwd &params, cudaStream_t stream);

void set_conv_params_fwd(ConvParamsBase &params,
                         // sizes
                         const size_t batch,
                         const size_t dim,
                         const size_t seqlen,
                         const size_t width,
                         // device pointers
                         const at::Tensor x,
                         const at::Tensor weight,
                         const at::Tensor out,
                         void* bias_ptr,
                         bool silu_activation,
                         bool explicit_channel_last) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.seqlen = seqlen;
    params.width = width;

    params.silu_activation = silu_activation;

    // Set the pointers and strides.
    params.x_ptr = x.data_ptr();
    params.weight_ptr = weight.data_ptr();
    params.bias_ptr = bias_ptr;
    params.out_ptr = out.data_ptr();
    // All stride are in elements, not bytes.
    params.x_batch_stride = x.stride(0);
    params.weight_c_stride = weight.stride(0);
    params.weight_width_stride = weight.stride(1);
    params.out_batch_stride = out.stride(0);
    
    if (explicit_channel_last){
        params.x_c_stride = x.stride(-1);
        params.x_l_stride = x.stride(1);
        params.out_c_stride = out.stride(-1);
        params.out_l_stride = out.stride(1);
    } else {
        params.x_c_stride = x.stride(1);
        params.x_l_stride = x.stride(-1);
        params.out_c_stride = out.stride(1);
        params.out_l_stride = out.stride(-1);
    }
}


void set_conv_params_bwd(ConvParamsBwd &params,
                         // sizes
                         const size_t batch,
                         const size_t dim,
                         const size_t seqlen,
                         const size_t width,
                         // device pointers
                         const at::Tensor x,
                         const at::Tensor weight,
                         void* bias_ptr,
                         const at::Tensor dout,
                         const at::Tensor dx,
                         const at::Tensor dweight,
                         void* dbias_ptr,
                         bool silu_activation,
                         bool explicit_channel_last) {
    // Pass in "dout" instead of "out", we're not gonna use "out" at all.
    set_conv_params_fwd(params, batch, dim, seqlen, width,
                        x, weight, dout, bias_ptr, silu_activation, explicit_channel_last);

    // Set the pointers and strides.
    params.dout_ptr = dout.data_ptr();
    params.dx_ptr = dx.data_ptr();
    params.dweight_ptr = dweight.data_ptr();
    params.dbias_ptr = dbias_ptr;
    // All stride are in elements, not bytes.
    params.dout_batch_stride = dout.stride(0);
    params.dweight_c_stride = dweight.stride(0);
    params.dweight_width_stride = dweight.stride(1);
    params.dx_batch_stride = dx.stride(0);
    if (explicit_channel_last){
        params.dout_c_stride = dout.stride(2);
        params.dout_l_stride = dout.stride(1);
        params.dx_c_stride = dx.stride(2);
        params.dx_l_stride = dx.stride(1);
    } else {
        params.dout_c_stride = dout.stride(1);
        params.dout_l_stride = dout.stride(2);
        params.dx_c_stride = dx.stride(1);
        params.dx_l_stride = dx.stride(2);
    }
    
}

at::Tensor
non_causal_conv1d_fwd(const at::Tensor &x, const at::Tensor &weight,
                  const c10::optional<at::Tensor> &bias_,
                  const c10::optional<at::Tensor> &position_indices_,
                  bool silu_activation,
                  bool explicit_channel_last,
                  c10::optional<int> num_threads_opt) {
    auto input_type = x.scalar_type();
    auto weight_type = weight.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(weight_type == at::ScalarType::Float || weight_type == at::ScalarType::Half || weight_type == at::ScalarType::BFloat16);

    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(weight.is_cuda());

    const auto sizes = x.sizes();
    const int batch_size = sizes[0];
    const int dim = explicit_channel_last ? sizes[2] : sizes[1];
    const int seqlen = explicit_channel_last ? sizes[1] : sizes[2];
    const int width = weight.size(-1);

    CHECK_SHAPE(weight, dim, width);

    TORCH_CHECK(x.stride(2) == 1, "We only support channel first or explicit channel last");

    TORCH_CHECK(width == 3, "non_causal_conv1d only supports width == 3");

    if (bias_.has_value()) {
        auto bias = bias_.value();
        TORCH_CHECK(bias.scalar_type() == weight_type);
        TORCH_CHECK(bias.is_cuda());
        TORCH_CHECK(bias.stride(-1) == 1);
        CHECK_SHAPE(bias, dim);
    }

    if (position_indices_.has_value()) {
        auto position_indices = position_indices_.value();
        TORCH_CHECK(position_indices.scalar_type() == torch::kInt32);
        TORCH_CHECK(position_indices.is_cuda());
        TORCH_CHECK(position_indices.is_contiguous());
        CHECK_SHAPE(position_indices, batch_size, seqlen);
    }
    at::Tensor out = torch::empty_like(x);

    ConvParamsBase params;
    set_conv_params_fwd(params, batch_size, dim, seqlen, width, x, weight, out,
                        bias_.has_value() ? bias_.value().data_ptr() : nullptr,
                        silu_activation, explicit_channel_last);
    if (num_threads_opt.has_value()) {
        params.num_threads = num_threads_opt.value();
    }

    if (position_indices_.has_value()) {
        params.position_indices_ptr = position_indices_.value().data_ptr();
    } else {
        params.position_indices_ptr = nullptr;
    }
    
    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "non_causal_conv1d_fwd", [&] {
        DISPATCH_WTYPE_FLOAT_AND_HALF_AND_BF16(weight.scalar_type(), "non_causal_conv1d_fwd", [&] {
            if (explicit_channel_last){
                non_causal_conv1d_channellast_fwd_cuda<input_t, weight_t>(params, stream);
            } else {
                non_causal_conv1d_fwd_cuda<input_t, weight_t>(params, stream);
            }
        });
    });
    return out;
}

std::vector<at::Tensor>
non_causal_conv1d_bwd(const at::Tensor &x, const at::Tensor &weight,
                  const c10::optional<at::Tensor> &bias_,
                  at::Tensor &dout,
                  const c10::optional<at::Tensor> &position_indices_,
                  c10::optional<at::Tensor> &dx_,
                  bool silu_activation,
                  bool explicit_channel_last,
                  c10::optional<int> num_threads_opt) {
    auto input_type = x.scalar_type();
    auto weight_type = weight.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(weight_type == at::ScalarType::Float || weight_type == at::ScalarType::Half || weight_type == at::ScalarType::BFloat16);

    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(weight.is_cuda());
    TORCH_CHECK(dout.is_cuda());

    const auto sizes = x.sizes();
    const int batch_size = sizes[0];
    const int dim = explicit_channel_last ? sizes[2] : sizes[1];
    const int seqlen = explicit_channel_last ? sizes[1] : sizes[2];
    const int width = weight.size(-1);

    TORCH_CHECK(width == 3, "non_causal_conv1d only supports width == 3");

    CHECK_SHAPE(weight, dim, width);
    if (explicit_channel_last){
        CHECK_SHAPE(dout, batch_size, seqlen, dim);
    } else {
        CHECK_SHAPE(dout, batch_size, dim, seqlen);
    }
    

    TORCH_CHECK(x.stride(2) == 1, "We only support channel first or explicit channel last");

    if (dout.stride(2) != 1) { dout = dout.contiguous(); }

    if (bias_.has_value()) {
        auto bias = bias_.value();
        TORCH_CHECK(bias.scalar_type() == weight_type);
        TORCH_CHECK(bias.is_cuda());
        TORCH_CHECK(bias.stride(-1) == 1);
        CHECK_SHAPE(bias, dim);
    }

    if (position_indices_.has_value()) {
        auto position_indices = position_indices_.value();
        TORCH_CHECK(position_indices.scalar_type() == torch::kInt32);
        TORCH_CHECK(position_indices.is_cuda());
        TORCH_CHECK(position_indices.is_contiguous());
        CHECK_SHAPE(position_indices, batch_size, seqlen);
    }

    at::Tensor dx;
    if (dx_.has_value()) {
        dx = dx_.value();
        TORCH_CHECK(dx.scalar_type() == input_type);
        TORCH_CHECK(dx.is_cuda());
        if (explicit_channel_last){
            CHECK_SHAPE(dx, batch_size, seqlen, dim);
        } else {
            CHECK_SHAPE(dx, batch_size, dim, seqlen);
        }
        TORCH_CHECK(dx.stride(2) == 1);
    } else {
        dx = torch::empty_like(x);
    }

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)x.get_device()};

    at::Tensor dweight = torch::zeros_like(weight, weight.options().dtype(at::kFloat));
    at::Tensor dbias;
    if (bias_.has_value()) { dbias = torch::zeros_like(bias_.value(), bias_.value().options().dtype(at::kFloat)); }

    ConvParamsBwd params;
    set_conv_params_bwd(params, batch_size, dim, seqlen, width,
                        x, weight, bias_.has_value() ? bias_.value().data_ptr() : nullptr,
                        dout, dx, dweight, bias_.has_value() ? dbias.data_ptr() : nullptr,
                        silu_activation, explicit_channel_last);
    if (num_threads_opt.has_value()) {
        params.num_threads = num_threads_opt.value();
    }


    if (position_indices_.has_value()) {
        params.position_indices_ptr = position_indices_.value().data_ptr();
    } else {
        params.position_indices_ptr = nullptr;
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "non_causal_conv1d_bwd", [&] {
        DISPATCH_WTYPE_FLOAT_AND_HALF_AND_BF16(weight.scalar_type(), "non_causal_conv1d_bwd", [&] {
            if (explicit_channel_last){
                non_causal_conv1d_channellast_bwd_cuda<input_t, weight_t>(params, stream);
            } else {
                non_causal_conv1d_bwd_cuda<input_t, weight_t>(params, stream);
            }
        });
    });
    return {dx, dweight.to(weight.dtype()), bias_.has_value() ? dbias.to(bias_.value().dtype()) : dbias};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("non_causal_conv1d_fwd", &non_causal_conv1d_fwd,
        py::arg("x"), py::arg("weight"), py::arg("bias") = nullptr, py::arg("position_indices") = nullptr,
        py::arg("silu_activation") = false, py::arg("explicit_channel_last") = false, py::arg("num_threads") = py::none(),
        "non_causal conv1d forward");
    m.def("non_causal_conv1d_bwd", &non_causal_conv1d_bwd,
        py::arg("x"), py::arg("weight"), py::arg("bias") = nullptr, py::arg("dout"), py::arg("position_indices") = nullptr,
        py::arg("dx") = nullptr, py::arg("silu_activation") = false, py::arg("explicit_channel_last") = false, py::arg("num_threads") = py::none(),
        "non_causal conv1d backward");
}
//