import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.utils import default, exists, cast_tuple
from functools import partial
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from torch import einsum
from functools import wraps


def _many(fn):
    @wraps(fn)
    def inner(tensors, pattern, **kwargs):
        return (fn(tensor, pattern, **kwargs) for tensor in tensors)

    return inner


rearrange_many = _many(rearrange)
repeat_many = _many(repeat)

# class Upsample(nn.Module):
#     def __init__(self, scale_factor=2, dim_in=None, dim_out=None):
#         super(Upsample, self).__init__()

#         self.scale_factor = scale_factor
#         self.conv = None
#         self.up = nn.Upsample(scale_factor=scale_factor, mode="nearest")
#         if dim_in is not None:
#             self.conv = nn.Conv2d(dim_in, default(dim_out, dim_in), 3, padding=1)

#     def forward(self, x):
#         x = self.up(x)
#         if self.conv is not None:
#             x = self.conv(x)
#         return x


# class Downsample(nn.Module):
#     def __init__(self, kernel_size=2, stride=2, padding=0, downsample_type="conv", dim_in=None, dim_out=None):
#         super(Downsample, self).__init__()

#         if downsample_type == "avg_pool":
#             self.down = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
#         elif downsample_type == "conv":
#             self.down = nn.Conv2d(dim_in, default(dim_out, dim_in), kernel_size, stride, padding=padding)
#         else:
#             raise NotImplementedError("downsample_type {} not implemented".format(downsample_type))

#     def forward(self, x):
#         return self.down(x)


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


def l2norm(t):
    return F.normalize(t, dim=-1)


def Upsample(dim, dim_out=None, scale_factor=2):
    return nn.Sequential(
        nn.Upsample(scale_factor=scale_factor, mode="nearest"), nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


# Allow for unused kwargs in forward pass
class ExtendKwargs(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x)


class AdaLN(nn.Module):
    def __init__(self, dim, condition_dim, zero_init=True, normalization=None):
        super().__init__()

        self.condition_proj = nn.Linear(condition_dim, dim) if condition_dim != dim else Identity()
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim, bias=True))
        self.normalization = normalization if normalization is not None else LayerNorm(dim)
        if zero_init:
            nn.init.zeros_(self.adaLN[-1].bias)
            nn.init.zeros_(self.adaLN[-1].weight)

    def modulate(self, x, shift, scale, gate):
        x = self.normalization(x)
        x = x * (scale + 1) + shift
        return x * gate

    def forward(self, x, condition=None):
        if condition is None:
            return self.normalization(x)
        adaln_params = self.adaLN(self.condition_proj(condition))
        adaln_params = rearrange(adaln_params, "b t c -> b c t 1 1")
        shift, scale, gate = adaln_params.chunk(3, dim=1)
        return self.modulate(x, shift, scale, gate)


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization.
    improve micro-batch training significantly.
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        flattened_weights = rearrange(weight, "o ... -> o (...)")

        mean = reduce(weight, "o ... -> o 1 1 1", "mean")

        var = torch.var(flattened_weights, dim=-1, unbiased=False)
        var = rearrange(var, "o -> o 1 1 1")

        weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class WeightStandardizedConv1d(nn.Conv1d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv1d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class WeightStandardizedConv3d(nn.Conv3d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv3d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class BasicLayerNorm(nn.Module):
    def __init__(self, dim, stable=False):
        super().__init__()
        self.stable = stable
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        if self.stable:
            x = x / x.amax(dim=-1, keepdim=True).detach()

        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + self.eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm=LayerNorm):
        super().__init__()
        self.fn = fn
        self.norm = norm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


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
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, x):
        device = x.device
        if x.dtype == torch.long:
            return timestep_embedding(x, self.dim)
        half_dim = self.dim // 2
        emb = math.log(self.max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb


class LearnedSinusoidalPosEmb(nn.Module):
    """following @crowsonkb 's lead with learned sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(
            groups, dim_out
        )  # GN’s computation is independent of batch sizes, and its accuracy is stable in a wide range of batch sizes.
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None, no_act=False):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        if not no_act:
            return x
        x = self.act(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, *, context_dim=None, dim_head=64, heads=8, norm_context=False, cosine_sim_attn=False):
        super().__init__()
        self.scale = dim_head**-0.5 if not cosine_sim_attn else 1.0
        self.cosine_sim_attn = cosine_sim_attn
        self.cosine_sim_scale = 16 if cosine_sim_attn else 1

        self.heads = heads
        inner_dim = dim_head * heads

        context_dim = default(context_dim, dim)

        self.norm = BasicLayerNorm(dim)
        self.norm_context = BasicLayerNorm(context_dim) if norm_context else nn.Identity()

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias=False), BasicLayerNorm(dim))

    def forward(self, x, context, mask=None):
        b = x.shape[0]
        x = self.norm(x)
        context = self.norm_context(context)

        if len(context.shape) == 2:  # TODO: check if this is correct
            context = repeat(context, "b d -> b n d", n=1)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))

        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=self.heads)

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(self.null_kv.unbind(dim=-2), "d -> b h 1 d", h=self.heads, b=b)

        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        q = q * self.scale

        # cosine sim attention

        if self.cosine_sim_attn:
            q, k = map(l2norm, (q, k))

        # similarities

        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.cosine_sim_scale
        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, max_neg_value)

        attn = sim.softmax(dim=-1, dtype=torch.float32)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class LinearCrossAttention(CrossAttention):
    def forward(self, x, context, mask=None):
        b = x.shape[0]

        x = self.norm(x)
        context = self.norm_context(context)

        if len(context.shape) == 2:
            context = repeat(context, "b d -> b n d", n=1)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))

        q, k, v = rearrange_many((q, k, v), "b n (h d) -> (b h) n d", h=self.heads)

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(self.null_kv.unbind(dim=-2), "d -> (b h) 1 d", h=self.heads, b=b)

        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        # masking

        max_neg_value = -torch.finfo(x.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, "b n -> b n 1")
            k = k.masked_fill(~mask, max_neg_value)
            v = v.masked_fill(~mask, 0.0)

        # linear attention

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        q = q * self.scale

        context = einsum("b n d, b n e -> b d e", k, v)
        out = einsum("b n d, b d e -> b n e", q, context)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.heads)

        return self.to_out(out)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        time_emb_dim=None,
        cond_dim=None,
        groups=8,
        linear_attn=False,
        **attn_kwargs,
    ):
        super().__init__()
        self.t_emb_layers = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if exists(time_emb_dim) else None
        )

        self.cross_attn = None

        if exists(cond_dim):
            attn_klass = CrossAttention if not linear_attn else LinearCrossAttention

            self.cross_attn = EinopsToAndFrom(
                "b c h w", "b (h w) c", attn_klass(dim=dim_out, context_dim=cond_dim, **attn_kwargs)
            )

        # self.id_emb_dim = id_emb_dim
        # self.id_emb_type = id_emb_type
        # self.id_emb_layers = None
        # if id_emb_dim is not None:
        #     dim_out_id = dim_out
        #     if id_emb_type == "scale_shift":
        #         dim_out_id = dim_out * 2
        #     self.id_emb_layers = nn.Sequential(
        #         nn.SiLU(),
        #         nn.Linear(id_emb_dim, dim_out_id),
        #     )
        self.act = nn.SiLU()
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, cond=None):
        scale_shift = None
        if exists(self.t_emb_layers) and exists(time_emb):
            time_emb = self.t_emb_layers(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        # is_id_cond = exists(self.id_emb_layers) and exists(id_emb)
        # h = self.block1(x, scale_shift=scale_shift)
        h = self.block1(x)

        # # TODO: check if it's the correct way to do it
        # if is_id_cond:
        #     if self.id_emb_type == "scale":
        #         id_emb = self.id_emb_layers(id_emb)
        #         id_emb = rearrange(id_emb, "b c -> b c 1 1")
        #         h *= id_emb
        #     elif self.id_emb_type == "add":
        #         id_emb = self.id_emb_layers(id_emb)
        #         id_emb = rearrange(id_emb, "b c -> b c 1 1")
        #         h += id_emb
        #     elif self.id_emb_type == "scale_shift":
        #         id_emb = self.id_emb_layers(id_emb)
        #         id_emb = rearrange(id_emb, "b c -> b c 1 1")
        #         scale, shift = id_emb.chunk(2, dim=1)
        #         h = h * (scale + 1) + shift
        #     else:
        #         raise NotImplementedError
        #     h = self.act(h)
        if exists(self.cross_attn):
            assert exists(cond)
            h = self.cross_attn(h, context=cond) + h

        h = self.block2(h, scale_shift=scale_shift)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), LayerNorm(dim))

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)

        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)

        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class QKVAttentionLegacy(nn.Module):
    def __init__(self, n_heads):
        super(QKVAttentionLegacy, self).__init__()
        self.n_heads = n_heads

    def forward(self, qkv, dtype=torch.float32):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.type(dtype), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class CondAttentionBlock(nn.Module):
    def __init__(self, channels, cond_dim, spatial_dim, num_heads=1, num_head_channels=-1):
        super(CondAttentionBlock, self).__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.cond_emb_proj = nn.Linear(cond_dim, spatial_dim**2)
        self.norm_x = nn.GroupNorm(32, channels)
        self.norm_emb = nn.GroupNorm(32, channels)
        self.q = nn.Conv1d(channels, channels, 1)
        self.kv = nn.Conv1d(channels, channels * 2, 1)

        self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x, context):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)

        # audio_emb = emb[1]
        cond_emb = self.cond_emb_proj(context)
        cond_emb = cond_emb.unsqueeze(1).expand(-1, c, -1)

        q = self.q(self.norm_x(x))
        kv = self.kv(self.norm_emb(cond_emb))
        qkv = torch.cat([q, kv], dim=1)

        h = self.attention(qkv)

        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


# Video relatated modules


class RelativePositionBias(nn.Module):
    def __init__(self, heads=8, num_buckets=32, max_distance=128):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, "j -> 1 j") - rearrange(q_pos, "i -> i 1")
        rp_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance
        )
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, "i j h -> h i j")


def Upsample3D(dim, dim_out=None, scale_factor=2):
    # return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))
    dim_out = default(dim_out, dim)

    return nn.Sequential(
        nn.Upsample(scale_factor=(1, scale_factor, scale_factor), mode="nearest"),
        PseudoConv2d(dim, dim_out, 3, padding=1),
    )


# def Downsample3D(dim):
#     return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


# pseudo conv2d that uses conv3d but with kernel size of 1 across frames dimension
def PseudoConv2d(dim_in, dim_out, kernel, stride=1, padding=0, **kwargs):
    kernel = cast_tuple(kernel, 2)
    stride = cast_tuple(stride, 2)
    padding = cast_tuple(padding, 2)

    if len(kernel) == 2:
        kernel = (1, *kernel)

    if len(stride) == 2:
        stride = (1, *stride)

    if len(padding) == 2:
        padding = (0, *padding)

    return nn.Conv3d(dim_in, dim_out, kernel, stride=stride, padding=padding, **kwargs)


def Downsample3D(dim, dim_out=None, factor=2):
    dim_out = default(dim_out, dim)
    return nn.Sequential(
        Rearrange("b c f (h p1) (w p2) -> b (c p1 p2) f h w", p1=factor, p2=factor), PseudoConv2d(dim * 4, dim_out, 1)
    )


class Block3D(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None, audio_scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        if exists(audio_scale_shift):
            scale, shift = audio_scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class ResnetBlock3D(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, audio_cond_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if exists(time_emb_dim) else None

        self.block1 = Block3D(dim, dim_out, groups=groups)
        self.block2 = Block3D(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

        self.audio_mlp = None
        if exists(audio_cond_dim):
            self.audio_mlp = nn.Sequential(nn.SiLU(), nn.Linear(audio_cond_dim, dim_out * 2))

    def forward(
        self,
        x,
        time_emb=None,
        audio_emb=None,
    ):
        scale_shift = None
        audio_scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), "time emb must be passed in"
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        if exists(self.audio_mlp):
            audio_emb = self.audio_mlp(audio_emb)
            audio_emb = rearrange(audio_emb, "b t c -> b c t 1 1")
            audio_scale_shift = audio_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift, audio_scale_shift=audio_scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)


class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, "b c f h w -> (b f) c h w")

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = rearrange_many(qkv, "b (h c) x y -> b h c (x y)", h=self.heads)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, "(b f) c h w -> b c f h w", b=b)


# attention along space and time


class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn, drop_dims: list = None):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn
        self.drop_dims = drop_dims  # If a dimension changes, drop it
        assert isinstance(drop_dims, list) or drop_dims is None

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(" "), shape)))
        if self.drop_dims is not None:
            for dim in self.drop_dims:
                reconstitute_kwargs.pop(dim)
        x = rearrange(x, f"{self.from_einops} -> {self.to_einops}")
        x = self.fn(x, **kwargs)
        x = rearrange(x, f"{self.to_einops} -> {self.from_einops}", **reconstitute_kwargs)
        return x


class SpaceTimeAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, rotary_emb=None):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x, pos_bias=None, focus_present_mask=None):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            return self.to_out(values)

        # split out heads

        q, k, v = rearrange_many(qkv, "... n (h d) -> ... h n d", h=self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        sim = einsum("... h i d, ... h j d -> ... h i j", q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            if len(focus_present_mask.shape) == 1:
                mask = torch.where(
                    rearrange(focus_present_mask, "b -> b 1 1 1 1"),
                    rearrange(attend_self_mask, "i j -> 1 1 1 i j"),
                    rearrange(attend_all_mask, "i j -> 1 1 1 i j"),
                )
            else:
                temp_focus = repeat(focus_present_mask, "b f -> b f n", n=n)
                temp_focus = temp_focus | rearrange(temp_focus, "b f n -> b n f")
                mask = torch.where(
                    rearrange(temp_focus, "b f n-> b 1 1 f n"),
                    rearrange(attend_self_mask, "i j -> 1 1 1 i j"),
                    rearrange(attend_all_mask, "i j -> 1 1 1 i j"),
                )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum("... h i j, ... h j d -> ... h i d", attn, v)
        out = rearrange(out, "... h n d -> ... n (h d)")
        return self.to_out(out)


class LayerNorm3D(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)

        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class PreNorm3D(nn.Module):
    def __init__(self, dim, fn, norm=LayerNorm3D):
        super().__init__()
        self.fn = fn
        self.norm = norm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


import numpy as np


def variance_scaling(scale, mode, distribution, in_axis=1, out_axis=0, dtype=torch.float32, device="cpu"):
    """Ported from JAX."""

    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError("invalid mode for variance scaling initializer: {}".format(mode))
        variance = scale / denominator
        if distribution == "normal":
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        elif distribution == "uniform":
            return (torch.rand(*shape, dtype=dtype, device=device) * 2.0 - 1.0) * np.sqrt(3 * variance)
        else:
            raise ValueError("invalid distribution for variance scaling initializer")

    return init


def default_init(scale=1.0):
    """The same initialization used in DDPM."""
    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, "fan_avg", "uniform")


def conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1.0, padding=1):
    """3x3 convolution with DDPM initialization."""
    conv = nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias
    )
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv


def conv1x1(in_planes, out_planes, stride=1, bias=True, init_scale=1.0, padding=0):
    """1x1 convolution with DDPM initialization."""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=bias)
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv


def get_norm(norm, ch, affine=True):
    """Get activation functions from the opt file."""
    if norm == "none":
        return nn.Identity()
    elif norm == "batch":
        return nn.BatchNorm1d(ch, affine=affine)
    elif norm == "group":
        num_groups = min(ch // 4, 32)
        while ch % num_groups != 0:  # must find another value
            num_groups -= 1
        return nn.GroupNorm(num_groups=num_groups, num_channels=ch, eps=1e-5, affine=affine)
    elif norm == "layer":
        return nn.LayerNorm(normalized_shape=ch, eps=1e-5, elementwise_affine=affine)
    elif norm == "instance":
        return nn.InstanceNorm2d(num_features=ch, eps=1e-5, affine=affine)
    else:
        raise NotImplementedError("norm choice does not exist")


class get_act_norm(nn.Module):  # order is norm -> act
    def __init__(
        self,
        act,
        act_emb,
        norm,
        ch,
        emb_dim=None,
        spectral=False,
        is3d=False,
        n_frames=1,
        num_frames_cond=0,
        cond_ch=0,
        spade_dim=128,
        cond_conv=None,
        cond_conv1=None,
    ):
        super(get_act_norm, self).__init__()

        self.norm = norm
        self.act = act
        self.act_emb = act_emb
        self.is3d = is3d
        self.n_frames = n_frames
        self.cond_ch = cond_ch
        if emb_dim is not None:
            if self.is3d:
                out_dim = 2 * (ch // self.n_frames)
            else:
                out_dim = 2 * ch
            if spectral:
                self.Dense_0 = torch.nn.utils.spectral_norm(nn.Linear(emb_dim, out_dim))
            else:
                self.Dense_0 = nn.Linear(emb_dim, out_dim)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)
            affine = False  # We remove scale/intercept after normalization since we will learn it with [temb, yemb]
        else:
            affine = True

        self.Norm_0 = get_norm(norm, (ch // n_frames) if is3d else ch, affine)

    def forward(self, x, emb=None, cond=None):
        if emb is not None:
            # emb = torch.cat([temb, yemb], dim=1) # Combine embeddings
            emb_out = self.Dense_0(self.act_emb(emb))[:, :, None, None]  # Linear projection
            # ada-norm as in https://github.com/openai/guided-diffusion
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            if self.is3d:
                B, CN, H, W = x.shape
                N = self.n_frames
                scale = scale.reshape(B, -1, 1, 1, 1)
                shift = shift.reshape(B, -1, 1, 1, 1)
                x = x.reshape(B, -1, N, H, W)
            if self.norm == "spade":
                emb_norm = self.Norm_0(x, cond)
                emb_norm = emb_norm.reshape(B, -1, N, H, W) if self.is3d else emb_norm
            else:
                emb_norm = self.Norm_0(x)
            x = emb_norm * (1 + scale) + shift
            if self.is3d:
                x = x.reshape(B, -1, H, W)
        else:
            if self.is3d:
                B, CN, H, W = x.shape
                N = self.n_frames
                x = x.reshape(B, -1, N, H, W)
            if self.norm == "spade":
                x = self.Norm_0(x, cond)
            else:
                x = self.Norm_0(x)
                x = x.reshape(B, CN, H, W) if self.is3d else x
        x = self.act(x)
        return x


class CatOut(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return torch.cat(self.module(x), dim=1)
