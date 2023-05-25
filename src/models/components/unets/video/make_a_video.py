import torch
from torch import nn

from einops import rearrange, repeat

from src.utils.torch_utils import prob_mask_like

from src.models.components.modules.make_a_video_modules import (
    SinusoidalPosEmb,
    ResnetBlock,
    Downsample,
    SpatioTemporalAttention,
    mlist,
    Upsample,
    PseudoConv3d,
    divisible_by,
    exists,
)


# space time factorized 3d unet


class SpaceTimeUnet(nn.Module):
    def __init__(
        self,
        *,
        dim,
        channels=3,
        cond_type="cat",
        condition_dim="channels",
        dim_mult=(1, 2, 4, 8),
        self_attns=(False, False, False, True),
        temporal_compression=(False, True, True, True),
        resnet_block_depths=(2, 2, 2, 2),
        attn_dim_head=64,
        attn_heads=8,
        condition_on_timestep=True,
        enable_time=True,
        attention_add_ff=True,
    ):
        super().__init__()

        in_channels = channels
        self.cond_type = cond_type
        if condition_dim == "time":
            self.cond_type = "time"
            in_channels += 1
        if self.cond_type == "cat":  # Concatenate the conditioning vector to the input channels
            in_channels *= 2

        self.has_cond = cond_type is not None or condition_dim == "time"

        self.enable_time = enable_time  # Can overwrite the enable_time flag for the entire model
        if not self.enable_time:
            temporal_compression = (False,) * len(temporal_compression)

        assert len(dim_mult) == len(self_attns) == len(temporal_compression) == len(resnet_block_depths)
        num_layers = len(dim_mult)

        dims = [dim, *map(lambda mult: mult * dim, dim_mult)]
        dim_in_out = zip(dims[:-1], dims[1:])

        # determine the valid multiples of the image size and frames of the video

        self.frame_multiple = 2 ** sum(tuple(map(int, temporal_compression)))
        self.image_size_multiple = 2**num_layers

        # timestep conditioning for DDPM, not to be confused with the time dimension of the video

        self.to_timestep_cond = None
        timestep_cond_dim = (dim * 4) if condition_on_timestep else None

        if condition_on_timestep:
            self.to_timestep_cond = nn.Sequential(SinusoidalPosEmb(dim), nn.Linear(dim, timestep_cond_dim), nn.SiLU())

        # layers

        self.downs = mlist([])
        self.ups = mlist([])

        attn_kwargs = dict(dim_head=attn_dim_head, heads=attn_heads, add_feed_forward=attention_add_ff)

        mid_dim = dims[-1]

        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, timestep_cond_dim=timestep_cond_dim, enable_time=enable_time)
        self.mid_attn = SpatioTemporalAttention(
            dim=mid_dim, enable_time=enable_time, add_feed_forward=attention_add_ff
        )
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, timestep_cond_dim=timestep_cond_dim, enable_time=enable_time)

        for _, self_attend, (dim_in, dim_out), compress_time, resnet_block_depth in zip(
            range(num_layers), self_attns, dim_in_out, temporal_compression, resnet_block_depths
        ):
            assert resnet_block_depth >= 1

            self.downs.append(
                mlist(
                    [
                        ResnetBlock(dim_in, dim_out, timestep_cond_dim=timestep_cond_dim, enable_time=enable_time),
                        mlist(
                            [ResnetBlock(dim_out, dim_out, enable_time=enable_time) for _ in range(resnet_block_depth)]
                        ),
                        SpatioTemporalAttention(dim=dim_out, enable_time=enable_time, **attn_kwargs)
                        if self_attend
                        else None,
                        Downsample(dim_out, downsample_time=compress_time),
                    ]
                )
            )

            self.ups.append(
                mlist(
                    [
                        ResnetBlock(dim_out * 2, dim_in, timestep_cond_dim=timestep_cond_dim, enable_time=enable_time),
                        mlist(
                            [
                                ResnetBlock(dim_in + (dim_out if ind == 0 else 0), dim_in, enable_time=enable_time)
                                for ind in range(resnet_block_depth)
                            ]
                        ),
                        SpatioTemporalAttention(dim=dim_in, enable_time=enable_time, **attn_kwargs)
                        if self_attend
                        else None,
                        Upsample(dim_out, upsample_time=compress_time),
                    ]
                )
            )

        self.skip_scale = 2**-0.5  # paper shows faster convergence

        self.conv_in = PseudoConv3d(
            dim=in_channels, dim_out=dim, kernel_size=7, temporal_kernel_size=3, enable_time=enable_time
        )
        self.conv_out = PseudoConv3d(
            dim=dim, dim_out=channels, kernel_size=3, temporal_kernel_size=3, enable_time=enable_time
        )

    def make_binary_mask(self, x, condition=False):
        B, _, T, H, W = x.shape
        mask = torch.zeros((B, 1, T, H, W), dtype=x.dtype, device=x.device)
        if condition:
            mask[:, :, 0, :, :] = 1.0
        return mask

    def forward_with_cond_scale(self, *args, cond_scale=2.0, **kwargs):
        # assert cond_scale == 1.0  # For now, we don't support classifier free guidance
        logits = self.forward(*args, null_cond_prob=0.0, **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.forward(*args, null_cond_prob=1.0, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(self, x, time, enable_time=True, cond=None, null_cond_prob=0.0, **kwargs):

        # some asserts
        if not self.enable_time:
            enable_time = False

        # assert not (exists(self.to_timestep_cond) ^ exists(timestep))
        is_video = x.ndim == 5

        if enable_time and is_video:
            frames = x.shape[2]
            if self.cond_type == "time" and cond is not None:
                frames += 1
            assert divisible_by(
                frames, self.frame_multiple
            ), f"number of frames on the video ({frames}) must be divisible by the frame multiple ({self.frame_multiple})"

        height, width = x.shape[-2:]
        assert divisible_by(height, self.image_size_multiple) and divisible_by(
            width, self.image_size_multiple
        ), f"height and width of the image or video must be a multiple of {self.image_size_multiple}"

        if self.has_cond:
            batch, device = x.shape[0], x.device
            mask = prob_mask_like((batch,), null_cond_prob, device=device)
            if cond is not None:
                cond = torch.where(rearrange(mask, "b -> b 1 1 1"), torch.zeros_like(cond), cond)
            if self.cond_type == "time":
                if cond is not None:
                    cond = rearrange(cond, "b c h w -> b c 1 h w")
                    x = torch.cat((cond, x), dim=2)  # Concatenate the condition in the time dimension
                binary_mask = self.make_binary_mask(x, condition=cond is not None)
                x = torch.cat((x, binary_mask), dim=1)
            elif self.cond_type == "cat":
                cond = repeat(cond, "b c h w -> b c t h w", t=x.shape[2])
                x = torch.cat((x, cond), dim=1)
            else:
                raise NotImplementedError(f"cond_type {self.cond_type} not implemented")

        # main logic
        t = self.to_timestep_cond(rearrange(time, "... -> (...)")) if exists(time) else None

        x = self.conv_in(x, enable_time=enable_time)

        hiddens = []

        for init_block, blocks, maybe_attention, downsample in self.downs:
            x = init_block(x, t, enable_time=enable_time)

            hiddens.append(x.clone())

            for block in blocks:
                x = block(x, enable_time=enable_time)

            if exists(maybe_attention):
                x = maybe_attention(x, enable_time=enable_time)

            hiddens.append(x.clone())

            x = downsample(x, enable_time=enable_time)

        x = self.mid_block1(x, t, enable_time=enable_time)
        x = self.mid_attn(x, enable_time=enable_time)
        x = self.mid_block2(x, t, enable_time=enable_time)

        add_skip_connection = lambda x: torch.cat((x, hiddens.pop() * self.skip_scale), dim=1)  # noqa

        for init_block, blocks, maybe_attention, upsample in reversed(self.ups):
            x = upsample(x, enable_time=enable_time)

            x = add_skip_connection(x)

            x = init_block(x, t, enable_time=enable_time)

            x = add_skip_connection(x)

            for block in blocks:
                x = block(x, enable_time=enable_time)

            if exists(maybe_attention):
                x = maybe_attention(x, enable_time=enable_time)

        x = self.conv_out(x, enable_time=enable_time)

        return x
