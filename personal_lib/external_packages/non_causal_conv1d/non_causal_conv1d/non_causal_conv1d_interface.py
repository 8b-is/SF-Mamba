# Copyright (c) 2024, Tri Dao.

import torch
import torch.nn.functional as F
import non_causal_conv1d_cuda


class NoneCausalConv1dFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias=None,
        position_indices=None,
        activation=None,
        explicit_channel_last=False,
        num_threads=None,
    ):
        if activation not in [None, "silu", "swish"]:
            raise NotImplementedError("activation must be None, silu, or swish")
        if x.stride(2) != 1:
            x = x.contiguous()
        bias = bias.contiguous() if bias is not None else None

        ctx.activation = activation in ["silu", "swish"]
        ctx.explicit_channel_last = explicit_channel_last
        ctx.num_threads = num_threads

        out = non_causal_conv1d_cuda.non_causal_conv1d_fwd(
            x, weight, bias, position_indices, ctx.activation, explicit_channel_last
        )
        ctx.save_for_backward(x, weight, bias, position_indices)
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        x, weight, bias, position_indices = ctx.saved_tensors
        if dout.stride(2) != 1:
            dout = dout.contiguous()

        dx, dweight, dbias = non_causal_conv1d_cuda.non_causal_conv1d_bwd(
            x,
            weight,
            bias,
            dout,
            position_indices, 
            None,
            ctx.activation,
            ctx.explicit_channel_last
        )

        return (
            dx,
            dweight,
            dbias if bias is not None else None,
            None,
            None,
            None,
            None
        )


def non_causal_conv1d_fn(
    x,
    weight,
    bias=None,
    position_indices=None,
    activation=None,
    explicit_channel_last=False,
    num_threads=None,
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    position_indices: (batch, seqlen)
    explicit_channel_last: if True, the input and output tensors are in (batch, seqlen, dim) format. If False, they are in (batch, dim, seqlen) format.
    activation: either None or "silu" or "swish",
    num_threads: number of threads to use in CUDA kernel. If None, it will be determined by the implementation.

    out: (batch, dim, seqlen)
    """
    return NoneCausalConv1dFn.apply(
        x,
        weight,
        bias,
        position_indices,
        activation,
        explicit_channel_last,
        num_threads,
    )


def non_causal_conv1d_ref(
    x,
    weight,
    bias=None,
    activation=None,
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    activation: either None or "silu" or "swish"

    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    out = F.conv1d(x, weight.unsqueeze(1), bias, padding=weight.shape[1]//2, groups=weight.shape[0])

    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return out

