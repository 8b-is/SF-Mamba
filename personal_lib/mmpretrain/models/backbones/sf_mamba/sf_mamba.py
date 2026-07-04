import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp, DropPath, trunc_normal_
from einops import rearrange, repeat
from functools import partial
from mmpretrain.registry import MODELS

try:
    from mamba_ssm import selective_scan_fn
except ImportError:
    selective_scan_fn = None

try:
    from non_causal_conv1d import non_causal_conv1d_fn
except:
    non_causal_conv1d_fn = None

try:
    from .n_over_batch_lut import BestBatchFoldingEstimator
except:
    from n_over_batch_lut import BestBatchFoldingEstimator

TRITON_AVAILABLE = True
try:
    from .swap_token_kernel import triton_swap
except:
    try:
        from swap_token_kernel import triton_swap
    except:
        triton_swap = None
        TRITON_AVAILABLE = False
try:
    seqlen = 8
    data = torch.randn(8, 64, 32).cuda()
    triton_swap(data, seqlen)
    del data
    del seqlen
except:
    TRITON_AVAILABLE = False


################## Modules #####################


class DropPathWoEdge(nn.Module):
    def __init__(
        self,
        drop_prob: float = 0.0,
        drop_path_wo_edge=True,
        scale_by_keep: bool = True,
        scale_edge: bool = False,
    ):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
        self.drop_path_wo_edge = drop_path_wo_edge
        self.scale_edge = scale_edge

    def forward(self, x, indices=None):
        return self.drop_path_wo_edge_(
            x,
            indices,
            self.drop_prob,
            self.training,
            self.scale_by_keep,
            self.scale_edge,
        )

    def drop_path_wo_edge_(
        self,
        x,
        indices,
        drop_prob: float = 0.0,
        training: bool = False,
        scale_by_keep: bool = True,
        scale_edge: bool = False,
    ):
        """
        Drop path without edge tokens (=auxiliary tokens).
        """
        if drop_prob == 0.0 or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (
            x.ndim - 1
        )  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if indices is not None and scale_edge:
            mask = (indices != 0).to(random_tensor.dtype)
            mask[:, 0] = 1  # in some case, ommitted
            random_tensor = (random_tensor + mask.unsqueeze(-1)).clamp(max=1)

        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)

        if indices is not None and (not scale_edge):
            mask = (indices != 0).to(random_tensor.dtype)
            mask[:, 0] = 1  # in some case, ommitted

            random_tensor = (random_tensor + mask.unsqueeze(-1)).clamp(
                max=1 / keep_prob
            )

        return x * random_tensor


class Downsample(nn.Module):
    def __init__(self, dim, down_act=None):
        """
        dim: Input channel dimension for this downsampling operator.
        down_act: Optional activation name inserted after the downsampling conv.
        """
        super().__init__()

        if down_act == "gelu":
            act_fn0 = nn.GELU()
        elif down_act == "elu":
            act_fn0 = nn.ELU()
        elif down_act is not None:
            act_fn0 = nn.ReLU()

        reduction = [nn.Conv2d(dim, 2 * dim, 3, 2, 1, bias=False)]
        if down_act is not None:
            reduction.append(act_fn0)
        self.reduction = nn.Sequential(*reduction)

    def forward(self, x):
        x = self.reduction(x)
        return x


class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        norm=nn.BatchNorm2d,
        act_func="relu",
    ):
        super(ConvModule, self).__init__()

        layer = []

        layer.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_size, kernel_size),
                stride=(stride, stride),
                padding=(padding, padding),
                dilation=(dilation, dilation),
                groups=groups,
                bias=bias,
            )
        )

        if norm:
            layer.append(norm(out_channels))

        if act_func == "gelu":
            act_fn0 = nn.GELU()
        elif act_func == "elu":
            act_fn0 = nn.ELU()
        elif act_func is not None:
            act_fn0 = nn.ReLU()

        if act_func is not None:
            layer.append(act_fn0)

        self.layer = nn.Sequential(*layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        return x


class DownsampleDepthWise(nn.Module):
    def __init__(self, dim, down_act="relu", ratio=4.0, down_norm=None):
        super().__init__()
        in_channels = dim
        out_channels = 2 * dim

        self.reduction = nn.Sequential(
            ConvModule(
                in_channels,
                int(out_channels * ratio),
                kernel_size=1,
                norm=None,
                act_func=down_act,
            ),
            ConvModule(
                int(out_channels * ratio),
                int(out_channels * ratio),
                kernel_size=3,
                stride=2,
                padding=1,
                groups=int(out_channels * ratio),
                norm=None,
                act_func=down_act,
            ),
            ConvModule(
                int(out_channels * ratio),
                out_channels,
                kernel_size=1,
                act_func=None,
                norm=None if down_norm is None else nn.BatchNorm2d,
            ),
        )

    def forward(self, x):
        x = self.reduction(x)
        return x


class LayerNorm2d(nn.LayerNorm):
    """
    normalized_shape (int or list or torch.Size): input shape from an expected input of size
    elementwise_affine: a boolean value that when set to ``True``, this module
        has learnable per-element affine parameters initialized to ones (for weights)
        and zeros (for biases). Default: ``True``
    bias: If set to ``False``, the layer will not learn an additive bias (only relevant if
        :attr:`elementwise_affine` is ``True``). Default: ``True``
    """
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )
        x = x.permute(0, 3, 1, 2)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, mid_patch_dim=64, dim=96, embed_act=(True, True)):
        """
        in_channels: Channel count of the input image tensor.
        mid_patch_dim: Intermediate channel width used in the first projection step.
        dim: Output embedding width produced by patch projection.
        embed_act: Two-element activation configuration for the stem conv blocks.
        """
        super().__init__()

        if embed_act[0] == "gelu":
            act_fn0 = nn.GELU()
        elif embed_act[0] == "elu":
            act_fn0 = nn.ELU()
        elif embed_act[0] is not None:
            act_fn0 = nn.ReLU()
        else:
            act_fn0 = nn.Identity()

        if embed_act[1] == "gelu":
            act_fn1 = nn.GELU()
        elif embed_act[1] == "elu":
            act_fn1 = nn.ELU()
        elif embed_act[1] is not None:
            act_fn1 = nn.ReLU()
        else:
            act_fn1 = nn.Identity()

        self.conv_down = nn.Sequential(
            nn.Conv2d(in_channels, mid_patch_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(mid_patch_dim, eps=1e-4),
            act_fn0,
            nn.Conv2d(mid_patch_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            act_fn1,
        )

    def forward(self, x):
        x = self.conv_down(x)
        return x


class MVModule(nn.Module):
    """
    Mamba mixer with MambaVision configurations. (Causal conv1d -> conv1d, 1/2 channel dimension split)
    """
    def __init__(
        self,
        dim,
        d_state=8,
        expand=1,
        dt_rank="auto",
        state_swap="v2",
    ):
        """
        dim: Input channel dimension for this module.
        d_state: Internal SSM state size used by Mamba modules.
        expand: Expansion factor for the hidden dimension of the module.
        dt_rank: Rank of the dynamic time projection. If set to "auto", it will be set to ceil(dim/16).
        state_swap: Version of the state swap mechanism to use.
        """

        super().__init__()
        self.dim = dim
        self.expand = expand
        # MambaVision split the hidden dimension into two equal parts to save computational cost.
        self.d_inner = int(self.expand * self.dim / 2)

        self.dt_rank = math.ceil(self.dim / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.dim, self.d_inner*2, bias=False)
        self.B_dim = d_state
        self.C_dim = d_state
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.B_dim + self.C_dim, bias=False
        )
        self.conv1d = nn.Conv1d(
            self.d_inner*2, self.d_inner*2, bias=False, kernel_size=3, groups=self.d_inner*2
        )
        self.dt_proj = nn.Conv1d(self.dt_rank, self.d_inner, 1, bias=True)
        self.out_proj = nn.Linear(self.d_inner*2, self.dim, bias=False)
        
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        self.state_swap = state_swap

        self._init_weights()

    def forward(self, x, indices=None):
        """
        x: (B, L, D)
        indices: (B, L) or None. 1 and -1 indicate the left and right edge of the pre-batch-folded sample.
        Returns: (B, L, D)
        """

        xz = self.in_proj(x)

        A = -torch.exp(self.A_log.float())

        if self.training:
            dtype = xz.dtype
            with torch.cuda.amp.autocast(enabled=False):
                xz = non_causal_conv1d_fn(
                    xz.contiguous().float(),
                    self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    position_indices=indices,
                    activation="silu",
                    explicit_channel_last=True,
                )
                x, z = xz.chunk(2, dim=2)
                dt, B, C = torch.split(
                    rearrange(self.x_proj(x), "b l d-> b d l"),
                    [self.dt_rank, self.B_dim, self.C_dim],
                    dim=1,
                )
                dt = self.dt_proj(dt)
                x = rearrange(x, "b l d-> b d l")

                y = selective_scan_fn(
                    x,
                    dt,
                    A,
                    B,
                    C,
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    return_last_state=False,
                    position_indices=indices,
                    need_grad=self.training,
                )
            y = y.to(dtype)
            z = z.to(dtype)
        else:
            xz = non_causal_conv1d_fn(
                xz,
                self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                position_indices=indices,
                activation="silu",
                explicit_channel_last=True,
            )
            x, z = xz.chunk(2, dim=2)
            dt, B, C = torch.split(
                rearrange(self.x_proj(x), "b l d-> b d l"),
                [self.dt_rank, self.B_dim, self.C_dim],
                dim=1,
            )
            dt = self.dt_proj(dt)
            x = rearrange(x, "b l d-> b d l")

            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=False,
                position_indices=indices,
                need_grad=self.training,
            )
        y = rearrange(y, "b d l -> b l d")
        y = torch.cat([y, z], dim=2)

        out = self.out_proj(y)

        return out

    def _init_weights(self):
        dt_min=0.001
        dt_max=0.1
        dt_scale=1.0
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj._no_reinit = True
        self.A_log._no_reinit = True
        self.D._no_reinit = True


class Attention(nn.Module):
    fused_attn = True
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


################## Layers #####################


class ConvLayer(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()

        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        feat = self.act(self.norm1(self.conv1(x)))
        feat = self.norm2(self.conv2(feat))
        x = x + self.drop_path(feat)
        return x


class MambaLayer(nn.Module):
    def __init__(
        self,
        dim,
        d_state=8,
        mlp_ratio=4.0,
        drop_path=0.0,
        drop_path_wo_edge=False,
        scale_edge=False,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        state_swap="v2",
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.state_swap = state_swap
        self.dim = dim
        self.mixer = MVModule(
            dim=dim, d_state=d_state, state_swap=state_swap
        )
        self.drop_path = DropPathWoEdge(
            drop_path, drop_path_wo_edge, scale_by_keep=True, scale_edge=scale_edge
        )
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer
        )

    def forward(self, x, indices=None):
        mix_out = self.mixer(self.norm1(x), indices=indices)

        x = x + self.drop_path(mix_out, indices)
        x = x + self.drop_path(self.mlp(self.norm2(x)), indices)
        return x


class AttentionLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.dim = dim
        self.mixer = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        self.drop_path = DropPath(drop_path)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer
        )

    def forward(self, x):
        x = x + self.drop_path(self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


################## Stages #####################


class ConvStage(nn.Module):
    """
    Conv Stage
    """

    def __init__(
        self,
        dim,
        depth,
        downsample=True,
        down_depthwise=False,
        down_depthwise_ratio=4.0,
        drop_path=0.0,
        classify=True,
        need_last_norm=False,
    ):
        """
        dim: Base channel size processed by this stage.
        depth: Count of convolutional residual blocks in the stage.
        downsample: Enables a spatial downsampling transition at stage output.
        down_depthwise: Uses depthwise-style downsampling instead of standard conv.
        down_depthwise_ratio: Expansion multiplier for depthwise downsampling layers.
        drop_path: Stochastic-depth probability (scalar or per-block list).
        classify: If True, run in classification mode (single output branch).
        need_last_norm: Apply terminal normalization when emitting dense features.
        """

        super().__init__()

        self.blocks = nn.ModuleList(
            [
                ConvLayer(
                    dim=dim,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                )
                for i in range(depth)
            ]
        )

        if not downsample:
            self.downsample = nn.Identity()
        elif down_depthwise:
            self.downsample = DownsampleDepthWise(dim=dim, ratio=down_depthwise_ratio)
        else:
            self.downsample = Downsample(dim=dim)

        self.classify = classify
        if not self.classify:
            if need_last_norm:
                self.out_norm = LayerNorm2d(dim)
            else:
                self.out_norm = nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        if self.classify:
            return self.downsample(x)
        else:
            return self.downsample(x), self.out_norm(x)


class HybridStage(nn.Module):
    """
    Hybrid Mamba+Attention Stage with Auxialiary Token Swapping
    """

    def __init__(
        self,
        dim,
        depth,
        trans_depth,
        num_heads,
        use_window=False,
        window_size=None,
        trans_window_size=None,
        downsample=True,
        down_depthwise=False,
        down_depthwise_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        d_state=8,
        state_swap=None,
        drop_path_wo_edge=True,
        scale_edge=False,
        classify=True,
        need_last_norm=False,
    ):
        """
        Inputs:
            dim: Channel width used throughout this hybrid stage.
            depth: Total number of stage blocks (Mamba + Transformer).
            trans_depth: Number of attention-based blocks allocated in the stage.
            num_heads: Multi-head attention count for Transformer blocks.
            use_window: If enabled, process tokens in local windows instead of full maps.
            window_size: Window resolution for the Mamba/attention tokenization path.
            trans_window_size: Optional alternate window resolution for Transformer blocks.
            downsample: Whether to append a downsampling op at stage end.
            down_depthwise: Select depthwise downsampling variant when downsampling.
            down_depthwise_ratio: Expansion factor used by depthwise downsampling layers.
            qkv_bias: Adds bias terms in Q/K/V linear projections when True.
            drop: Dropout probability applied in Transformer branches.
            drop_path: Stochastic-depth setting for residual branches.
            d_state: Internal SSM state size used by Mamba mixers.
            state_swap: Auxiliary-token swapping mode (supports "v2" or None).
            drop_path_wo_edge: Keep auxiliary edge tokens from being dropped.
            scale_edge: Scale auxiliary edge-token paths during stochastic depth.
            classify: Classification mode flag controlling return behavior.
            need_last_norm: Attach final normalization for dense prediction outputs.
        """

        super().__init__()

        self.num_mamba = depth - trans_depth
        self.mamba_blocks = nn.ModuleList(
            [
                MambaLayer(
                    dim=dim,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    drop_path_wo_edge=drop_path_wo_edge,
                    scale_edge=scale_edge,
                    state_swap=state_swap,
                    d_state=d_state,
                )
                for i in range(self.num_mamba)
            ]
        )
        self.num_trans = trans_depth

        self.trans_blocks = nn.ModuleList(
            [
                AttentionLayer(
                    dim=dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                )
                for i in range(self.num_trans)
            ]
        )

        if not downsample:
            self.downsample = nn.Identity()
        elif down_depthwise:
            self.downsample = DownsampleDepthWise(dim=dim, ratio=down_depthwise_ratio)
        else:
            self.downsample = Downsample(dim=dim)

        self.use_window = use_window
        self.classify = classify
        if not self.classify:
            if need_last_norm:
                self.out_norm = LayerNorm2d(dim)
            else:
                self.out_norm = nn.Identity()

        self.window_size = (
            window_size
            if isinstance(window_size, (tuple, list))
            else [window_size, window_size]
        )
        if trans_window_size is None:
            self.trans_window_size = None
        else:
            self.trans_window_size = (
                trans_window_size
                if isinstance(trans_window_size, (tuple, list))
                else [trans_window_size, trans_window_size]
            )

        if self.use_window and self.trans_window_size is not None:
            assert (
                self.window_size[0] % self.trans_window_size[0] == 0
                and self.window_size[1] % self.trans_window_size[1] == 0
            ), "Transformer window size must divide Mamba window size."

        self.dim = dim
        self.d_state = d_state

        self.state_swap = state_swap
        self.indices = None
        self.pre_b = -1
        self.pre_seqlen = -1
        self.n = -1

        self.best_n_estimator = BestBatchFoldingEstimator()

    def forward(self, x):
        batch, C, H, W = x.shape

        #### BCHW to BLC
        if not self.use_window:
            x = rearrange(x, "b c h w -> b (h w) c")
        else:
            pad_r = (self.window_size[0] - W % self.window_size[0]) % self.window_size[
                0
            ]
            pad_b = (self.window_size[1] - H % self.window_size[1]) % self.window_size[
                1
            ]
            if pad_r > 0 or pad_b > 0:
                x = torch.nn.functional.pad(x, (0, pad_r, 0, pad_b))
                _, _, H_padded, W_padded = x.shape
            else:
                H_padded, W_padded = H, W
            if self.window_size[0] == W_padded and self.window_size[1] == H_padded:
                x = rearrange(x, "b c h w -> b (h w) c")
            else:
                x = self.local_windowing(x, self.window_size[1], self.window_size[0])
            batch = x.shape[0]

        #### Initialize auxiliary swap tokens with mean pooling
        if self.state_swap == "v2":
            x_mean = x.mean(dim=1, keepdim=True)
            x = torch.cat([x_mean, x, x_mean], dim=1)

        #### Batch Folding
        seqlen = x.shape[1]
        if seqlen != self.pre_seqlen or self.pre_b != batch:
            if self.training:
                self.n = batch
            else:
                self.n = self.best_n_estimator.get_best_n(
                    batch, self.dim, seqlen, self.d_state
                )

            if self.n != batch:
                self.indices = torch.zeros(
                    self.n,
                    (batch // self.n) * seqlen,
                    device=x.device,
                    dtype=torch.int32,
                )
                self.indices[:, ::seqlen] = 1
                self.indices[:, seqlen - 1 :: seqlen] = -1
            else:
                self.indices = None
            self.pre_b = batch
            self.pre_seqlen = seqlen
        x = rearrange(x, "(n b) l c -> n (b l) c", n=self.n)

        #### Apply Mamba Blocks
        for i, blk in enumerate(self.mamba_blocks):
            x = blk(x, indices=self.indices)
            if self.state_swap is not None and i < len(self.mamba_blocks) - 1:
                if self.training:
                    st_token = x[:, ::seqlen].clone()
                    ed_token = x[:, seqlen - 1 :: seqlen].clone()
                    x[:, ::seqlen] = ed_token
                    x[:, seqlen - 1 :: seqlen] = st_token
                elif TRITON_AVAILABLE:
                    x = triton_swap(x, seqlen)
                else:
                    st_token = x[:, ::seqlen].clone()
                    x[:, ::seqlen] = x[:, seqlen - 1 :: seqlen]
                    x[:, seqlen - 1 :: seqlen] = st_token

        #### Apply Transformer Blocks
        if self.num_trans > 0:
            x = rearrange(
                x, "n (b l) c -> (n b) l c", b=batch // self.n
            )  # Restore Batch Folding

            if self.trans_window_size is None:  # Use the same window size as Mamba
                for i, blk in enumerate(self.trans_blocks):
                    x = blk(x)

                    if self.state_swap == "v2":
                        if (
                            i == 0
                        ):  # Discard auxialiary tokens after first Transformer block
                            x = x[:, 1:-1]

                if not self.use_window:
                    x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
                else:
                    if self.window_size[0] == W_padded and self.window_size[1] == H_padded:
                        x = rearrange(x, "b (h w) c -> b c h w", h=H_padded, w=W_padded)
                    else:
                        x = self.reverse_local_windowing(
                            x, self.window_size[1], self.window_size[0], H_padded, W_padded
                        )

                    if pad_r > 0 or pad_b > 0:
                        x = x[:, :, :H, :W].contiguous()

            else:  # Different window size for Transformer
                if self.state_swap is not None:
                    st_ed_token = x[:, [0, -1]]
                    x = x[:, 1:-1]

                if self.use_window:
                    if self.window_size[0] == W_padded and self.window_size[1] == H_padded:
                        x = rearrange(x, "b (h w) c -> b h w c", h=H_padded, w=W_padded)
                    else:
                        x = self.reverse_local_windowing_ch_last(
                            x, self.window_size[1], self.window_size[0], H_padded, W_padded
                        )
                    x = self.local_windowing_ch_last(
                        x, self.trans_window_size[1], self.trans_window_size[0]
                    )

                else:
                    x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

                    pad_r_t = (
                        self.trans_window_size[0] - W % self.trans_window_size[0]
                    ) % self.trans_window_size[0]
                    pad_b_t = (
                        self.trans_window_size[1] - H % self.trans_window_size[1]
                    ) % self.trans_window_size[1]
                    if pad_r_t > 0 or pad_b_t > 0:
                        x = torch.nn.functional.pad(
                            x, (0, max(0, pad_r_t), 0, max(0, pad_b_t))
                        )
                    _, _, H_padded, W_padded = x.shape

                    x = self.local_windowing(
                        x, self.trans_window_size[1], self.trans_window_size[0]
                    )
                batch = x.shape[0]

                if (
                    self.state_swap == "v2"
                ):  # if not self.use_window and self.state_swap == "v2":
                    st_ed_token = st_ed_token.repeat_interleave(
                        batch // st_ed_token.size(0), dim=0
                    )
                    x = torch.cat([st_ed_token, x], dim=1)

                for i, blk in enumerate(self.trans_blocks):
                    x = blk(x)

                    if (
                        self.state_swap == "v2"
                    ):  # if not self.use_window and self.state_swap == "v2":
                        if (
                            i == 0
                        ):  # Discard auxialiary tokens after first Transformer block
                            x = x[:, 2:]

                x = self.reverse_local_windowing(
                    x, self.trans_window_size[1], self.trans_window_size[0], H_padded, W_padded
                )
                x = x[:, :, :H, :W].contiguous()

        else:
            if not self.use_window:
                if self.state_swap == "v2":
                    x = rearrange(x, "n (b l) c -> (n b) c l", l=seqlen)
                    x = x[:, :, 1:-1]
                    x = rearrange(x, "b c (h w) -> b c h w", h=H, w=W)
                else:
                    x = rearrange(x, "n (b h w) c -> (n b) c h w", h=H, w=W)
            else:
                x = rearrange(x, "n (b l) c -> (n b) l c", b=batch // self.n)
                if self.state_swap == "v2":
                    x = x[:, 1:-1]
                x = self.reverse_local_windowing(
                    x, self.window_size[1], self.window_size[0], H_padded, W_padded
                )
                if pad_r > 0 or pad_b > 0:
                    x = x[:, :, :H, :W].contiguous()

        if self.classify:
            return self.downsample(x)
        else:
            return self.downsample(x), self.out_norm(x)

    def local_windowing(self, x, window_size_h, window_size_w):
        """
        Inputs:
            x: Feature tensor laid out as (batch, channels, height, width).
            window_size_h: Vertical span of each extracted patch window.
            window_size_w: Horizontal span of each extracted patch window.
        Output:
            Flattened tokens for every local region with shape
            (batch * num_windows, window_size_h * window_size_w, channels).
        """
        B, C, H, W = x.shape
        x = x.view(
            B, C, H // window_size_h, window_size_h, W // window_size_w, window_size_w
        )
        windows = x.permute(0, 2, 4, 3, 5, 1).reshape(
            -1, window_size_h * window_size_w, C
        )
        return windows

    def local_windowing_ch_last(self, x, window_size_h, window_size_w):
        """
        Inputs:
            x: Channel-last feature map in (batch, height, width, channels) format.
            window_size_h: Number of rows included in one local window.
            window_size_w: Number of columns included in one local window.
        Output:
            Sequence-form window features with size
            (batch * num_windows, window_size_h * window_size_w, channels).
        """
        B, H, W, C = x.shape
        x = x.view(
            B, H // window_size_h, window_size_h, W // window_size_w, window_size_w, C
        )
        windows = x.permute(0, 1, 3, 2, 4, 5).reshape(
            -1, window_size_h * window_size_w, C
        )
        return windows

    def reverse_local_windowing(self, x, window_size_h, window_size_w, H, W):
        """
        Inputs:
            x: windowed feature map with shape
                (batch * num_windows, window_size_h, window_size_w, channels).
            window_size_h: Window extent along the height axis.
            window_size_w: Window extent along the width axis.
            H: Target image height after stitching windows back.
            W: Target image width after stitching windows back.
        Output:
            Reconstructed channel-first tensor of shape (batch, channels, H, W).
        """
        x_shape = x.shape
        B = int(x_shape[0] / (H * W / window_size_h / window_size_w))
        x = x.reshape(
            B, H // window_size_h, W // window_size_w, window_size_h, window_size_w, -1
        )
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, x_shape[2], H, W)
        return x

    def reverse_local_windowing_ch_last(
        self, x, window_size_h, window_size_w, H, W
    ):
        """
        Inputs:
            x: windowed feature map with shape
                (batch * num_windows, window_size_h, window_size_w, channels).
            window_size_h: Height covered by each window.
            window_size_w: Width covered by each window.
            H: Final image height.
            W: Final image width.
        Output:
            Restored channel-last feature map with layout (batch, H, W, channels).
        """
        x_shape = x.shape
        B = int(x_shape[0] / (H * W / window_size_h / window_size_w))
        x = x.reshape(
            B, H // window_size_h, W // window_size_w, window_size_h, window_size_w, -1
        )
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, x_shape[2])
        return x


################## Backbone #####################


@MODELS.register_module()
class SFMamba(nn.Module):
    """
    SFMamba Backbone
    """

    def __init__(
        self,
        in_channels=3,
        mid_patch_dim=32,
        dim=80,
        conv_stages=[0, 1],
        depths=[1, 3, 8, 4],
        trans_depths=[None, None, "auto", "auto"],
        state_swap="v2",
        use_window=False,
        window_size=[None, None, None, None],
        trans_window_size=None,
        num_heads=[None, None, 8, 16],
        down_depthwise=False,
        down_depthwise_conv=False,
        down_depthwise_ratio=4.0,
        drop_path_rate=0.2,
        drop_path_wo_edge=True,
        scale_edge=False,
        classify=True,
        out_indices=(0, 1, 2, 3),
        with_classifier=False,
        num_classes=1000,
    ):
        """
        Inputs:
            in_channels: Number of image input channels.
            dim: Base embedding width for the first stage.
            mid_patch_dim: Intermediate width used by the patch embedding stem.
            conv_stages: Stage indices that should use pure convolutional blocks.
            depths: Total block counts for each stage.
            trans_depths: Transformer block counts per stage; "auto" means half of stage depth.
            state_swap: Auxiliary state-token swap mode ("v2" or None).
            use_window: Enable local-window token processing.
            window_size: Window configuration per stage for Mamba/attention execution.
            trans_window_size: Optional per-stage Transformer-specific window configuration.
            num_heads: Attention head counts for Transformer-enabled stages.
            drop_path_rate: Global stochastic-depth schedule maximum.
            down_depthwise: Use depthwise downsampling in hybrid stages.
            down_depthwise_conv: Use depthwise downsampling in convolutional stages.
            down_depthwise_ratio: Channel expansion ratio used in depthwise downsampling.
            drop_path_wo_edge: Exempt auxiliary edge tokens from drop-path masking.
            scale_edge: Apply edge-token scaling when drop-path is active.
            classify: Toggle between classification and dense-feature outputs.
            out_indices: Stage indices to return for downstream dense tasks.
            with_classifier: Attach and run a linear classifier head. (For FLOPs measurement only.)
            num_classes: Class count used by the classifier head.
        """
        super().__init__()

        self.num_classes = num_classes
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels, mid_patch_dim=mid_patch_dim, dim=dim
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            if i in conv_stages:
                stage = ConvStage(
                    dim=int(dim * 2**i),
                    depth=depths[i],
                    drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                    downsample=(i < len(depths) - 1),
                    down_depthwise=down_depthwise_conv,
                    down_depthwise_ratio=down_depthwise_ratio,
                    classify=classify,
                    need_last_norm=(not classify) and i in out_indices,
                )
            else:
                stage = HybridStage(
                    dim=int(dim * 2**i),
                    depth=depths[i],
                    trans_depth=depths[i] // 2
                    if trans_depths[i] == "auto"
                    else trans_depths[i],
                    num_heads=num_heads[i],
                    window_size=window_size[i],
                    trans_window_size=None
                    if trans_window_size is None
                    else trans_window_size[i],
                    drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                    downsample=(i < len(depths) - 1),
                    down_depthwise=down_depthwise,
                    down_depthwise_ratio=down_depthwise_ratio,
                    state_swap=state_swap,
                    drop_path_wo_edge=drop_path_wo_edge,
                    scale_edge=scale_edge,
                    classify=classify,
                    need_last_norm=(not classify) and i in out_indices,
                    use_window=use_window,
                )
            self.stages.append(stage)

        num_features = int(dim * 2 ** (len(depths) - 1))
        self.classify = classify
        if self.classify:
            self.norm = nn.BatchNorm2d(num_features)
            self.with_classifier = with_classifier
            if self.with_classifier:
                self.head = nn.Linear(num_features, num_classes)
        else:
            self.out_indices = out_indices

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if getattr(m, "skip_weight_init", False) or getattr(m, "_no_reinit", False):
            return
        elif isinstance(m, (nn.Linear, nn.Conv1d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, LayerNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward_classify(self, x):
        x = self.patch_embed(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)

        if self.with_classifier:
            x = self.head(x.mean(dim=(2, 3)))

        return (x,)

    def forward_dense(self, x):
        x = self.patch_embed(x)
        outputs = []
        for i, stage in enumerate(self.stages):
            x, out = stage(x)
            if i in self.out_indices:
                outputs.append(out.contiguous())
        return outputs

    def forward(self, x):
        if self.classify:
            return self.forward_classify(x)
        else:
            return self.forward_dense(x)
