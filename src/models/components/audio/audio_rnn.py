import torch.nn as nn
import math
from math import ceil
import torch
from functools import partial
from einops import rearrange


def prime_factors(number):
    factor = 2
    factors = []
    while factor * factor <= number:
        if number % factor:
            factor += 1
        else:
            number //= factor
            factors.append(int(factor))
    if number > 1:
        factors.append(int(number))
    return factors


def calculate_padding(kernel_size, stride=1, in_size=0):
    out_size = ceil(float(in_size) / float(stride))
    return int((out_size - 1) * stride + kernel_size - in_size)


def calculate_output_size(in_size, kernel_size, stride, padding):
    return int((in_size + padding - kernel_size) / stride) + 1


def cut_audio_sequence_batch(seq, feature_length, overlap, rate):
    seq = seq.view(seq.shape[0], -1, 1)  # add batch dimension and reshape
    snip_length = int(feature_length * rate)
    cutting_stride = int((feature_length - overlap) * rate)
    pad_samples = snip_length - cutting_stride
    pad_left = torch.zeros(seq.shape[0], pad_samples // 2, 1, device=seq.device)  # add batch dimension
    pad_right = torch.zeros(seq.shape[0], pad_samples - pad_samples // 2, 1, device=seq.device)  # add batch dimension
    seq = torch.cat((pad_left, seq), 1)  # concatenate along time dimension
    seq = torch.cat((seq, pad_right), 1)  # concatenate along time dimension
    stacked = seq[:, :snip_length, :].unsqueeze(1)  # add time dimension
    iterations = (seq.size()[1] - snip_length) // cutting_stride + 1
    for i in range(1, iterations):
        segment = seq[:, i * cutting_stride : i * cutting_stride + snip_length, :]
        stacked = torch.cat((stacked, segment.unsqueeze(1)), 1)  # add time dimension
    return stacked


class Encoder(nn.Module):
    def __init__(
        self,
        code_size,
        rate,
        feat_length,
        init_kernel=None,
        init_stride=None,
        num_feature_maps=16,
        increasing_stride=True,
    ):
        super(Encoder, self).__init__()

        self.code_size = code_size
        self.cl = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.strides = []
        self.kernels = []

        features = feat_length * rate
        strides = prime_factors(features)
        kernels = [2 * s for s in strides]

        if init_kernel is not None and init_stride is not None:
            self.strides.append(int(init_stride * rate))
            self.kernels.append(int(init_kernel * rate))
            padding = calculate_padding(init_kernel * rate, stride=init_stride * rate, in_size=features)
            init_features = calculate_output_size(
                features, init_kernel * rate, stride=init_stride * rate, padding=padding
            )
            strides = prime_factors(init_features)
            kernels = [2 * s for s in strides]

        if not increasing_stride:
            strides.reverse()
            kernels.reverse()

        self.strides.extend(strides)
        self.kernels.extend(kernels)

        for i in range(len(self.strides) - 1):
            padding = calculate_padding(self.kernels[i], stride=self.strides[i], in_size=features)
            features = calculate_output_size(features, self.kernels[i], stride=self.strides[i], padding=padding)
            pad = int(math.ceil(padding / 2.0))

            if i == 0:
                self.cl.append(nn.Conv1d(1, num_feature_maps, self.kernels[i], stride=self.strides[i], padding=pad))
                self.activations.append(nn.Sequential(nn.BatchNorm1d(num_feature_maps), nn.ReLU(True)))
            else:
                self.cl.append(
                    nn.Conv1d(
                        num_feature_maps, 2 * num_feature_maps, self.kernels[i], stride=self.strides[i], padding=pad
                    )
                )
                self.activations.append(nn.Sequential(nn.BatchNorm1d(2 * num_feature_maps), nn.ReLU(True)))

                num_feature_maps *= 2

        self.cl.append(nn.Conv1d(num_feature_maps, self.code_size, features))
        self.activations.append(nn.Tanh())

    def forward(self, x):
        for i in range(len(self.strides)):
            x = self.cl[i](x)
            x = self.activations[i](x)

        return x.squeeze()


class AudioRNN(nn.Module):
    def __init__(
        self,
        enc_code_size,
        rnn_code_size,
        rate=16000,
        feat_length=0.2,
        overlap=0.16,
        n_layers=2,
        init_kernel=None,
        init_stride=None,
    ):
        super(AudioRNN, self).__init__()
        self.cut_audio_fn = partial(cut_audio_sequence_batch, feature_length=feat_length, overlap=overlap, rate=rate)
        self.audio_feat_samples = int(rate * feat_length)
        self.enc_code_size = enc_code_size
        self.rnn_code_size = rnn_code_size
        self.encoder = Encoder(self.enc_code_size, rate, feat_length, init_kernel=init_kernel, init_stride=init_stride)
        self.rnn = nn.GRU(self.enc_code_size, self.rnn_code_size, n_layers, batch_first=True)
        self.code_size = self.rnn_code_size

    def forward(self, x, lengths=None):
        if len(x.shape) == 4:
            return x  # already embedded
        x = rearrange(x, "b ... -> b (...)")
        x = self.cut_audio_fn(x)
        seq_length = x.size()[1]
        x = x.view(-1, 1, self.audio_feat_samples)
        x = self.encoder(x)
        x = x.view(-1, seq_length, self.enc_code_size)
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        x, _ = self.rnn(x)
        if lengths is not None:
            x, lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return rearrange(x, "b f c -> b f () c")


if __name__ == "__main__":
    model = AudioRNN(256, 128)
    x = torch.rand(2, 1, int(16000 * (16 / 25)))
    y = model(x)
    print(y.shape)
