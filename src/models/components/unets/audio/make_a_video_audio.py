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

from src.models.components.modules.torch_modules import EinopsToAndFrom, CatOut

from src.models.components.modules.DWT_IDWT_layer import DWT_2D, IDWT_2D


def identity(t):
    return t


def is_lambda(f):
    return callable(f) and f.__name__ == "<lambda>"


def default(val, d):
    if exists(val):
        return val
    return d() if is_lambda(d) else d


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
        # Audio
        audio_encoder=None,
        audio_cond_type="scale_shift",
        audio_features_type="cat",
        # Other
        init_img_transform=None,
        augment_dim=0,  # Augmentation label dimensionality, 0 = no augmentation.
    ):
        super().__init__()

        assert len(dim_mult) == len(self_attns) == len(temporal_compression) == len(resnet_block_depths)
        num_layers = len(dim_mult)

        if init_img_transform == "dwt_2d":
            init_img_transform = EinopsToAndFrom(
                "b c t h w", "(b t) c h w", CatOut(DWT_2D("haar")), drop_dims=["c", "t", "h", "w"]
            )
            final_img_itransform = IDWT_2D("haar")
            self.n_wavelets = 4
        # elif init_img_transform == "dwt_3d":
        #     init_img_transform = CatOut(DWT_3D("haar"))
        #     final_img_itransform = IDWT_3D("haar")
        #     self.n_wavelets = 8
        else:
            init_img_transform = None
            final_img_itransform = None
            self.n_wavelets = 1
        self.init_img_transform = default(init_img_transform, identity)
        self.final_img_itransform = default(final_img_itransform, identity)

        dims = [dim, *map(lambda mult: mult * dim, dim_mult)]
        dim_in_out = zip(dims[:-1], dims[1:])

        mid_dim = dims[-1]
        in_channels = channels
        self.cond_type = cond_type
        if condition_dim == "time":
            self.cond_type = "time"
            in_channels += 1
        if self.cond_type == "cat":  # Concatenate the conditioning vector to the input channels
            in_channels *= 2

        in_channels *= self.n_wavelets
        channels *= self.n_wavelets

        self.audio_encoder = audio_encoder()
        audio_code_size = self.audio_encoder.code_size
        self.audio_cond_type = audio_cond_type
        self.audio_features_type = audio_features_type
        feed_as_frames = self.audio_encoder.feed_as_frames if hasattr(self.audio_encoder, "feed_as_frames") else True
        audio_code_size *= 2 if (not feed_as_frames and audio_features_type == "cat") else 1
        audio_cond_dim = None
        self.to_audio_cond = None
        attention_condition = False
        if audio_cond_type == "scale_shift":
            audio_cond_dim = dim * 4
            self.to_audio_cond = nn.Sequential(nn.Linear(audio_code_size, audio_cond_dim), nn.SiLU())
        elif audio_cond_type == "mid_scale_shift":
            self.to_audio_cond = nn.Sequential(nn.Linear(audio_code_size, mid_dim * 2), nn.SiLU())
        elif audio_cond_type == "channel_concat":
            # Only for mel spectrogram for now
            self.to_audio_cond = nn.Sequential(nn.Linear(4, self.audio_encoder.n_mels), nn.SiLU())
            in_channels += 1
            self.audio_features_type = "transpose"
        elif audio_cond_type == "adln_resnet":
            audio_cond_dim = audio_code_size
            self.to_audio_cond = nn.Identity()
        elif audio_cond_type == "cross_att":
            attention_condition = True
            audio_cond_dim = audio_code_size
            self.to_audio_cond = nn.Identity()
        elif audio_cond_type == "adln_att":
            attention_condition = True
            audio_cond_dim = audio_code_size
            self.to_audio_cond = nn.Identity()
        elif audio_cond_type == "None" or audio_cond_type is None:
            audio_cond_dim = None
            self.audio_cond_type = "None"
            audio_cond_type = "None"
        else:
            raise ValueError("Unknown audio_cond_type")

        self.has_cond = cond_type is not None or condition_dim == "time"

        self.enable_time = enable_time  # Can overwrite the enable_time flag for the entire model
        if not self.enable_time:
            temporal_compression = (False,) * len(temporal_compression)

        # determine the valid multiples of the image size and frames of the video

        self.frame_multiple = 2 ** sum(tuple(map(int, temporal_compression)))
        self.image_size_multiple = 2**num_layers

        # timestep conditioning for DDPM, not to be confused with the time dimension of the video

        self.to_timestep_cond = None
        timestep_cond_dim = (dim * 4) if condition_on_timestep else None

        if condition_on_timestep:
            self.to_timestep_cond = nn.Sequential(SinusoidalPosEmb(dim), nn.Linear(dim, timestep_cond_dim), nn.SiLU())

        # augment conditioning
        if augment_dim:
            augment_dim = 9  # The number of augmentations we support
        self.map_augment = (
            nn.Linear(in_features=augment_dim, out_features=timestep_cond_dim, bias=False) if augment_dim else None
        )

        # layers
        self.downs = mlist([])
        self.ups = mlist([])

        attn_kwargs = dict(
            dim_head=attn_dim_head,
            heads=attn_heads,
            add_feed_forward=attention_add_ff,
            condition_dim=audio_cond_dim if attention_condition else None,
            cond_type=audio_cond_type,
        )

        self.mid_block1 = ResnetBlock(
            mid_dim,
            mid_dim,
            timestep_cond_dim=timestep_cond_dim,
            audio_cond_dim=audio_cond_dim,
            enable_time=enable_time,
            cond_type=audio_cond_type,
        )
        self.mid_attn = SpatioTemporalAttention(
            dim=mid_dim,
            enable_time=enable_time,
            add_feed_forward=attention_add_ff,
            condition_dim=audio_cond_dim if attention_condition else None,
            cond_type=audio_cond_type,
        )
        self.mid_block2 = ResnetBlock(
            mid_dim,
            mid_dim,
            timestep_cond_dim=timestep_cond_dim,
            audio_cond_dim=audio_cond_dim,
            enable_time=enable_time,
            cond_type=audio_cond_type,
        )

        for _, self_attend, (dim_in, dim_out), compress_time, resnet_block_depth in zip(
            range(num_layers), self_attns, dim_in_out, temporal_compression, resnet_block_depths
        ):
            assert resnet_block_depth >= 1

            self.downs.append(
                mlist(
                    [
                        ResnetBlock(
                            dim_in,
                            dim_out,
                            timestep_cond_dim=timestep_cond_dim,
                            audio_cond_dim=audio_cond_dim,
                            cond_type=audio_cond_type,
                            enable_time=enable_time,
                        ),
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
                        ResnetBlock(
                            dim_out * 2,
                            dim_in,
                            timestep_cond_dim=timestep_cond_dim,
                            audio_cond_dim=audio_cond_dim,
                            cond_type=audio_cond_type,
                            enable_time=enable_time,
                        ),
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

    def forward(
        self, x, time, audio=None, augment_labels=None, enable_time=True, cond=None, null_cond_prob=0.0, **kwargs
    ):
        # some asserts
        if not self.enable_time:
            enable_time = False

        # assert not (exists(self.to_timestep_cond) ^ exists(timestep))
        is_video = x.ndim == 5
        frames = 1
        if is_video:
            frames = x.shape[2]

        # conditional dropout
        if exists(audio):
            batch, device = x.shape[0], x.device
            mask = prob_mask_like((batch,), null_cond_prob, device=device)
            audio = torch.where(
                rearrange(mask, "b -> b" + " 1" * (len(audio.shape) - 1)), torch.zeros_like(audio), audio
            )

        if self.audio_cond_type == "None":
            audio = None
        else:
            audio = self.audio_encoder(audio)
            if self.audio_features_type == "cat":
                audio = rearrange(audio, "b f d c -> b f (d c)")
            elif self.audio_features_type == "mean":
                audio = audio.mean(dim=2)
            elif self.audio_features_type == "add":
                audio = audio.sum(dim=2)
            elif self.audio_features_type == "transpose":
                audio = audio.transpose(-1, -2)
            else:
                raise ValueError(f"audio_features_type {self.audio_features_type} not supported")
            if exists(self.to_audio_cond):
                audio = self.to_audio_cond(audio)

        audio_emb = None
        if self.audio_cond_type == "scale_shift" or "adln" in self.audio_cond_type:
            audio_emb = audio
        audio_emb_att = None
        if self.audio_cond_type == "cross_att":
            audio_emb_att = audio

        if enable_time and is_video:
            # frames = x.shape[2]
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

        x = self.init_img_transform(x)

        if self.audio_cond_type == "channel_concat":
            assert audio.shape[-1] == x.shape[-1], "Make sure to set n_mels to the same value as the image width"
            x = torch.cat((x, audio.unsqueeze(1)), dim=1)

        # main logic
        t = self.to_timestep_cond(rearrange(time, "... -> (...)")) if exists(time) else None
        if exists(self.map_augment) and exists(augment_labels):
            augment_labels = self.map_augment(augment_labels)
            t = t + augment_labels

        x = self.conv_in(x, enable_time=enable_time)

        hiddens = []

        for init_block, blocks, maybe_attention, downsample in self.downs:
            x = init_block(x, t, audio_emb=audio_emb, enable_time=enable_time)

            hiddens.append(x.clone())

            for block in blocks:
                x = block(x, enable_time=enable_time)

            if exists(maybe_attention):
                x = maybe_attention(x, enable_time=enable_time, context=audio_emb_att)

            hiddens.append(x.clone())

            x = downsample(x, enable_time=enable_time)

        if self.audio_cond_type == "mid_scale_shift":
            audio = rearrange(audio, "b t c -> b c t 1 1")
            scale, shift = audio.chunk(2, dim=1)
            x = x * (scale + 1) + shift

        x = self.mid_block1(x, t, audio_emb=audio_emb, enable_time=enable_time)
        x = self.mid_attn(x, enable_time=enable_time, context=audio_emb_att)
        x = self.mid_block2(x, t, audio_emb=audio_emb, enable_time=enable_time)

        add_skip_connection = lambda x: torch.cat((x, hiddens.pop() * self.skip_scale), dim=1)  # noqa

        for init_block, blocks, maybe_attention, upsample in reversed(self.ups):
            x = upsample(x, enable_time=enable_time)

            x = add_skip_connection(x)

            x = init_block(x, t, audio_emb=audio_emb, enable_time=enable_time)

            x = add_skip_connection(x)

            for block in blocks:
                x = block(x, enable_time=enable_time)

            if exists(maybe_attention):
                x = maybe_attention(x, enable_time=enable_time, context=audio_emb_att)

        x = self.conv_out(x, enable_time=enable_time)

        T = x.shape[2]

        wavelets = x.chunk(self.n_wavelets, dim=1)
        if len(wavelets) == 4:
            wavelets = [rearrange(w, "b c f h w -> (b f) c h w") for w in wavelets]
        x = self.final_img_itransform(*wavelets)
        if len(wavelets) == 4:
            x = rearrange(x, "(b f) c h w -> b c f h w", f=T)

        return x
