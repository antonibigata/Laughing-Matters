import math
import functools
from operator import mul

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from src.utils.utils import _many
from src.models.components.modules.torch_modules import AdaLN, ExtendKwargs

repeat_many = _many(repeat)

# helper functions


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def mul_reduce(tup):
    return functools.reduce(mul, tup)


def divisible_by(numer, denom):
    return (numer % denom) == 0


mlist = nn.ModuleList

# for time conditioning


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=timesteps.device
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.theta = theta
        self.dim = dim

    def forward(self, x):
        dtype, device = x.dtype, x.device
        if dtype == torch.long:
            return timestep_embedding(x, self.dim)
        assert dtype == torch.float, "input to sinusoidal pos emb must be a float type"

        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -emb)
        emb = rearrange(x, "i -> i 1") * rearrange(emb, "j -> 1 j")
        return torch.cat((emb.sin(), emb.cos()), dim=-1).type(dtype)


# layernorm 3d


# class LayerNorm(nn.Module):
#     def __init__(self, dim, expand_to_3d=False):
#         super().__init__()
#         if expand_to_3d:
#             self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))
#         else:
#             self.g = nn.Parameter(torch.ones(dim))

#     def forward(self, x):
#         eps = 1e-5 if x.dtype == torch.float32 else 1e-3
#         var = torch.var(x, dim=1, unbiased=False, keepdim=True)
#         mean = torch.mean(x, dim=1, keepdim=True)
#         return (x - mean) * var.clamp(min=eps).rsqrt() * self.g


class ChanLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim, 1, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * var.clamp(min=eps).rsqrt() * self.g


# feedforward

# feedforward


def shift_token(t):
    t, t_shift = t.chunk(2, dim=1)
    t_shift = F.pad(t_shift, (0, 0, 0, 0, 1, -1), value=0.0)
    return torch.cat((t, t_shift), dim=1)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()

        inner_dim = int(dim * mult * 2 / 3)
        self.proj_in = nn.Sequential(nn.Conv3d(dim, inner_dim * 2, 1, bias=False), GEGLU())

        self.proj_out = nn.Sequential(ChanLayerNorm(inner_dim), nn.Conv3d(inner_dim, dim, 1, bias=False))

    def forward(self, x, enable_time=True):
        x = self.proj_in(x)

        if enable_time:
            x = shift_token(x)

        return self.proj_out(x)


# best relative positional encoding


class ContinuousPositionBias(nn.Module):
    """from https://arxiv.org/abs/2111.09883"""

    def __init__(self, *, dim, heads, num_dims=1, layers=2):
        super().__init__()
        self.num_dims = num_dims

        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(self.num_dims, dim), nn.SiLU()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), nn.SiLU()))

        self.net.append(nn.Linear(dim, heads))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, *dimensions):
        device = self.device

        shape = torch.tensor(dimensions, device=device)
        rel_pos_shape = 2 * shape - 1

        # calculate strides

        strides = torch.flip(rel_pos_shape, (0,)).cumprod(dim=-1)
        strides = torch.flip(F.pad(strides, (1, -1), value=1), (0,))

        # get all positions and calculate all the relative distances

        positions = [torch.arange(d, device=device) for d in dimensions]
        grid = torch.stack(torch.meshgrid(*positions, indexing="ij"), dim=-1)
        grid = rearrange(grid, "... c -> (...) c")
        rel_dist = rearrange(grid, "i c -> i 1 c") - rearrange(grid, "j c -> 1 j c")

        # get all relative positions across all dimensions

        rel_positions = [torch.arange(-d + 1, d, device=device) for d in dimensions]
        rel_pos_grid = torch.stack(torch.meshgrid(*rel_positions, indexing="ij"), dim=-1)
        rel_pos_grid = rearrange(rel_pos_grid, "... c -> (...) c")

        # mlp input

        bias = rel_pos_grid.float()

        for layer in self.net:
            bias = layer(bias)

        # convert relative distances to indices of the bias

        rel_dist += shape - 1  # make sure all positive
        rel_dist *= strides
        rel_dist_indices = rel_dist.sum(dim=-1)

        # now select the bias for each unique relative position combination

        bias = bias[rel_dist_indices]
        return rearrange(bias, "i j h -> h i j")


# helper classes


class Attention_old(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        nn.init.zeros_(self.to_out.weight.data)  # identity with skip connection

    def forward(self, x, rel_pos_bias=None):

        x = self.norm(x)

        q, k, v = self.to_q(x), *self.to_kv(x).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))

        q = q * self.scale

        sim = einsum("b h i d, b h j d -> b h i j", q, k)

        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class DynamicPositionBias(nn.Module):
    def __init__(self, dim, *, heads, depth):
        super().__init__()
        self.mlp = nn.ModuleList([])

        self.mlp.append(nn.Sequential(nn.Linear(1, dim), nn.LayerNorm(dim), nn.SiLU()))

        for _ in range(max(depth - 1, 0)):
            self.mlp.append(nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU()))

        self.mlp.append(nn.Linear(dim, heads))

    def forward(self, n, device, dtype):
        i = torch.arange(n, device=device)
        j = torch.arange(n, device=device)

        indices = rearrange(i, "i -> i 1") - rearrange(j, "j -> 1 j")
        indices += n - 1

        pos = torch.arange(-n + 1, n, device=device, dtype=dtype)
        pos = rearrange(pos, "... -> ... 1")

        for layer in self.mlp:
            pos = layer(pos)

        bias = pos[indices]
        bias = rearrange(bias, "i j h -> h i j")
        return bias


# attention pooling
def l2norm(t):
    return F.normalize(t, dim=-1)


# Imported from imagen modules
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        causal=False,
        context_dim=None,
        cosine_sim_attn=False,
        rel_pos_bias=False,
        rel_pos_bias_mlp_depth=2,
        # init_zero=False,
        context_type="cross_att",
        dropout=0.0,
    ):
        super().__init__()
        self.scale = dim_head**-0.5 if not cosine_sim_attn else 1.0
        self.causal = causal

        self.cosine_sim_attn = cosine_sim_attn
        self.cosine_sim_scale = 16 if cosine_sim_attn else 1

        self.rel_pos_bias = (
            DynamicPositionBias(dim=dim, heads=heads, depth=rel_pos_bias_mlp_depth) if rel_pos_bias else None
        )

        self.heads = heads
        inner_dim = dim_head * heads

        if "adln" in context_type and exists(context_dim):
            self.norm = AdaLN(dim, context_dim, normalization=nn.LayerNorm(dim))
        else:
            self.norm = ExtendKwargs(nn.LayerNorm(dim))

        self.null_attn_bias = nn.Parameter(torch.randn(heads))

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias=False)

        self.to_context = (
            nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, dim_head * 2))
            if exists(context_dim) and context_type == "cross_att"
            else None
        )

        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        if "adln" in context_type and exists(context_dim):
            self.to_out_norm = AdaLN(dim, context_dim, normalization=nn.LayerNorm(dim))
        else:
            self.to_out_norm = ExtendKwargs(nn.LayerNorm(dim))
        self.attn_dropout = nn.Dropout(dropout)
        # if init_zero:
        #     nn.init.zeros_(self.to_out[-1].g)

        nn.init.zeros_(self.to_out.weight.data)  # identity with skip connection

    def forward(self, x, context=None, mask=None, rel_pos_bias=None):
        attn_bias = rel_pos_bias  # To match with make a vid code
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x, condition=context)  # condition is used for adln
        q = self.to_q(x)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        q = q * self.scale

        # add null key / value for classifier free guidance in prior net
        nk, nv = repeat_many(self.null_kv.unbind(dim=-2), "d -> b 1 d", b=b)

        if exists(context):
            assert exists(self.to_context)
            k, v = self.to_context(context).chunk(2, dim=-1)
            # k = torch.cat((ck, k), dim=-2)
            # v = torch.cat((cv, v), dim=-2)
        else:
            k, v = self.to_kv(x).chunk(2, dim=-1)

        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        # cosine sim attention

        if self.cosine_sim_attn:
            q, k = map(l2norm, (q, k))

        # calculate query / key similarities

        sim = einsum("b h i d, b j d -> b h i j", q, k) * self.cosine_sim_scale

        # relative positional encoding (T5 style)

        if not exists(attn_bias) and exists(self.rel_pos_bias):
            attn_bias = self.rel_pos_bias(n, device=device, dtype=q.dtype)

        if exists(attn_bias):
            null_attn_bias = repeat(self.null_attn_bias, "h -> h n 1", n=n)
            attn_bias = torch.cat((null_attn_bias, attn_bias), dim=-1)
            sim = sim + attn_bias

        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, max_neg_value)

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, max_neg_value)

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out_norm(self.to_out(out), condition=context)  # condition is used for adln


# main contribution - pseudo 3d conv


class PseudoConv3d(nn.Module):
    def __init__(self, dim, dim_out=None, kernel_size=3, *, temporal_kernel_size=None, enable_time=True, **kwargs):
        super().__init__()
        dim_out = default(dim_out, dim)
        temporal_kernel_size = default(temporal_kernel_size, kernel_size)

        self.spatial_conv = nn.Conv2d(dim, dim_out, kernel_size=kernel_size, padding=kernel_size // 2)
        if enable_time:
            self.temporal_conv = (
                nn.Conv1d(dim_out, dim_out, kernel_size=temporal_kernel_size, padding=temporal_kernel_size // 2)
                if kernel_size > 1
                else None
            )
        else:
            self.temporal_conv = None

        if exists(self.temporal_conv):
            nn.init.dirac_(self.temporal_conv.weight.data)  # initialized to be identity
            nn.init.zeros_(self.temporal_conv.bias.data)

    def forward(self, x, enable_time=True):
        b, c, *_, h, w = x.shape

        is_video = x.ndim == 5
        enable_time &= is_video

        if is_video:
            x = rearrange(x, "b c f h w -> (b f) c h w")

        x = self.spatial_conv(x)

        if is_video:
            x = rearrange(x, "(b f) c h w -> b c f h w", b=b)

        if not enable_time or not exists(self.temporal_conv):
            return x

        x = rearrange(x, "b c f h w -> (b h w) c f")

        x = self.temporal_conv(x)

        x = rearrange(x, "(b h w) c f -> b c f h w", h=h, w=w)

        return x


# factorized spatial temporal attention from Ho et al.
# todo - take care of relative positional biases + rotary embeddings


class SpatioTemporalAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head=64,
        heads=8,
        enable_time=True,
        add_feed_forward=True,
        ff_mult=4,
        condition_dim=None,
        cond_type="cross_att",
    ):
        super().__init__()
        self.spatial_attn = Attention(dim=dim, dim_head=dim_head, heads=heads)
        self.spatial_rel_pos_bias = ContinuousPositionBias(dim=dim // 2, heads=heads, num_dims=2)

        self.enable_time = enable_time
        if enable_time:
            self.temporal_attn = Attention(
                dim=dim, dim_head=dim_head, heads=heads, context_dim=condition_dim, context_type=cond_type
            )
            self.temporal_rel_pos_bias = ContinuousPositionBias(dim=dim // 2, heads=heads, num_dims=1)

        self.has_feed_forward = add_feed_forward
        if add_feed_forward:
            self.ff = FeedForward(dim=dim, mult=ff_mult)

    def forward(self, x, context=None, enable_time=True):
        b, c, *_, h, w = x.shape
        is_video = x.ndim == 5
        enable_time &= is_video
        enable_time &= self.enable_time

        if is_video:
            x = rearrange(x, "b c f h w -> (b f) (h w) c")
        else:
            x = rearrange(x, "b c h w -> b (h w) c")

        space_rel_pos_bias = self.spatial_rel_pos_bias(h, w)

        x = self.spatial_attn(x, rel_pos_bias=space_rel_pos_bias) + x

        if is_video:
            x = rearrange(x, "(b f) (h w) c -> b c f h w", b=b, h=h, w=w)
        else:
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        if not enable_time:
            return x

        h, w = x.shape[-2:]
        if context is not None:
            context = repeat(context, "b f c -> (b h w) f c", h=h, w=w)
        x = rearrange(x, "b c f h w -> (b h w) f c")

        time_rel_pos_bias = self.temporal_rel_pos_bias(x.shape[1])

        x = self.temporal_attn(x, context=context, rel_pos_bias=time_rel_pos_bias) + x

        x = rearrange(x, "(b h w) f c -> b c f h w", w=w, h=h)

        if self.has_feed_forward:
            x = self.ff(x, enable_time=enable_time) + x

        return x


# resnet block


class Block(nn.Module):
    def __init__(
        self, dim, dim_out, kernel_size=3, temporal_kernel_size=None, groups=8, enable_time=True, adln_dim=None
    ):
        super().__init__()
        self.project = PseudoConv3d(dim, dim_out, 3, enable_time=enable_time)
        if adln_dim is None:
            self.norm = ExtendKwargs(nn.GroupNorm(groups, dim_out))
        else:
            self.norm = AdaLN(dim_out, adln_dim, normalization=nn.GroupNorm(groups, dim_out))
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None, audio_emb=None, enable_time=False):
        x = self.project(x, enable_time=enable_time)
        x = self.norm(x, condition=audio_emb)  # Condition used only for AdaLN

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        # if exists(audio_scale_shift):
        #     scale, shift = audio_scale_shift
        #     x = x * (scale + 1) + shift

        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        timestep_cond_dim=None,
        audio_cond_dim=None,
        groups=8,
        enable_time=True,
        cond_type="scale_shift",
    ):
        super().__init__()

        self.timestep_mlp = None

        if exists(timestep_cond_dim):
            self.timestep_mlp = nn.Sequential(nn.SiLU(), nn.Linear(timestep_cond_dim, dim_out * 2))

        self.audio_mlp = None
        if exists(audio_cond_dim) and cond_type == "scale_shift":
            self.audio_mlp = nn.Sequential(nn.SiLU(), nn.Linear(audio_cond_dim, dim_out * 2))

        self.block1 = Block(
            dim,
            dim_out,
            groups=groups,
            enable_time=enable_time,
            adln_dim=audio_cond_dim if "adln" in cond_type else None,
        )
        self.block2 = Block(dim_out, dim_out, groups=groups, enable_time=enable_time)
        self.res_conv = PseudoConv3d(dim, dim_out, 1, enable_time=enable_time) if dim != dim_out else nn.Identity()

    def forward(self, x, timestep_emb=None, audio_emb=None, enable_time=True):
        assert not (exists(timestep_emb) ^ exists(self.timestep_mlp))

        scale_shift = None
        # audio_scale_shift = None

        if exists(self.timestep_mlp) and exists(timestep_emb):
            time_emb = self.timestep_mlp(timestep_emb)
            to_einsum_eq = "b c 1 1 1" if x.ndim == 5 else "b c 1 1"
            time_emb = rearrange(time_emb, f"b c -> {to_einsum_eq}")
            scale_shift = time_emb.chunk(2, dim=1)

        if exists(self.audio_mlp):
            audio_emb = self.audio_mlp(audio_emb)
            audio_emb = rearrange(audio_emb, "b t c -> b c t 1 1")
            emb = audio_emb + time_emb
            scale_shift = emb.chunk(2, dim=1)
            # audio_scale_shift = audio_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift, audio_emb=audio_emb, enable_time=enable_time)

        h = self.block2(h, enable_time=enable_time)

        return h + self.res_conv(x)


# pixelshuffle upsamples and downsamples
# where time dimension can be configured


class Downsample(nn.Module):
    def __init__(self, dim, downsample_space=True, downsample_time=False, nonlin=False):
        super().__init__()
        assert downsample_space or downsample_time

        self.down_space = (
            nn.Sequential(
                Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
                nn.Conv2d(dim * 4, dim, 1, bias=False),
                nn.SiLU() if nonlin else nn.Identity(),
            )
            if downsample_space
            else None
        )

        self.down_time = (
            nn.Sequential(
                Rearrange("b c (f p) h w -> b (c p) f h w", p=2),
                nn.Conv3d(dim * 2, dim, 1, bias=False),
                nn.SiLU() if nonlin else nn.Identity(),
            )
            if downsample_time
            else None
        )

    def forward(self, x, enable_time=True):
        is_video = x.ndim == 5

        if is_video:
            x = rearrange(x, "b c f h w -> b f c h w")
            x, ps = pack([x], "* c h w")

        if exists(self.down_space):
            x = self.down_space(x)

        if is_video:
            (x,) = unpack(x, ps, "* c h w")
            x = rearrange(x, "b f c h w -> b c f h w")

        if not is_video or not exists(self.down_time) or not enable_time:
            return x

        x = self.down_time(x)

        return x


class Upsample(nn.Module):
    def __init__(self, dim, upsample_space=True, upsample_time=False, nonlin=False, space_scale_factor=2):
        super().__init__()
        assert upsample_space or upsample_time

        self.up_space = (
            nn.Sequential(
                nn.Conv2d(dim, dim * 4, 1),
                nn.SiLU() if nonlin else nn.Identity(),
                Rearrange("b (c p1 p2) h w -> b c (h p1) (w p2)", p1=space_scale_factor, p2=space_scale_factor),
            )
            if upsample_space
            else None
        )

        self.up_time = (
            nn.Sequential(
                nn.Conv3d(dim, dim * 2, 1),
                nn.SiLU() if nonlin else nn.Identity(),
                Rearrange("b (c p) f h w -> b c (f p) h w", p=2),
            )
            if upsample_time
            else None
        )

        self.init_()

    def init_(self):
        if exists(self.up_space):
            self.init_conv_(self.up_space[0], 4)

        if exists(self.up_time):
            self.init_conv_(self.up_time[0], 2)

    def init_conv_(self, conv, factor):
        o, *remain_dims = conv.weight.shape
        conv_weight = torch.empty(o // factor, *remain_dims)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, "o ... -> (o r) ...", r=factor)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x, enable_time=True):
        is_video = x.ndim == 5

        if is_video:
            x = rearrange(x, "b c f h w -> b f c h w")
            x, ps = pack([x], "* c h w")

        if exists(self.up_space):
            x = self.up_space(x)

        if is_video:
            (x,) = unpack(x, ps, "* c h w")
            x = rearrange(x, "b f c h w -> b c f h w")

        if not is_video or not exists(self.up_time) or not enable_time:
            return x

        x = self.up_time(x)

        return x
