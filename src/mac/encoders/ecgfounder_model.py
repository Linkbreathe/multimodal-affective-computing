"""ECGFounder 1D ResNet architecture.

Adapted from PKUDigitalHealth/ECGFounder ``net1d.py`` at commit 04edac7.
The upstream project is MIT licensed:

Copyright (c) 2025 PKUDigitalHealth
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyConv1dPadSame(nn.Module):
    """Conv1d with TensorFlow-style SAME padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dim = (x.shape[-1] + self.stride - 1) // self.stride
        pad = max(0, (out_dim - 1) * self.stride + self.kernel_size - x.shape[-1])
        pad_left = pad // 2
        pad_right = pad - pad_left
        return self.conv(F.pad(x, (pad_left, pad_right), "constant", 0))


class MyMaxPool1dPadSame(nn.Module):
    """MaxPool1d with SAME padding."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.max_pool = nn.MaxPool1d(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = max(0, self.kernel_size - 1)
        pad_left = pad // 2
        pad_right = pad - pad_left
        return self.max_pool(F.pad(x, (pad_left, pad_right), "constant", 0))


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class BasicBlock(nn.Module):
    """ECGFounder bottleneck residual block with squeeze-excitation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ratio: float,
        kernel_size: int,
        stride: int,
        groups: int,
        downsample: bool,
        is_first_block: bool = False,
        use_bn: bool = True,
        use_do: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.groups = groups
        self.downsample = downsample
        self.stride = stride if downsample else 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do
        self.middle_channels = int(out_channels * ratio)

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.activation1 = Swish()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(in_channels, self.middle_channels, 1, 1)

        self.bn2 = nn.BatchNorm1d(self.middle_channels)
        self.activation2 = Swish()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            self.middle_channels,
            self.middle_channels,
            kernel_size,
            self.stride,
            groups=groups,
        )

        self.bn3 = nn.BatchNorm1d(self.middle_channels)
        self.activation3 = Swish()
        self.do3 = nn.Dropout(p=0.5)
        self.conv3 = MyConv1dPadSame(self.middle_channels, out_channels, 1, 1)

        reduction = 2
        self.se_fc1 = nn.Linear(out_channels, out_channels // reduction)
        self.se_fc2 = nn.Linear(out_channels // reduction, out_channels)
        self.se_activation = Swish()

        if downsample:
            self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = x

        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.activation1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)

        if self.use_bn:
            out = self.bn2(out)
        out = self.activation2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)

        if self.use_bn:
            out = self.bn3(out)
        out = self.activation3(out)
        if self.use_do:
            out = self.do3(out)
        out = self.conv3(out)

        se = out.mean(-1)
        se = self.se_fc1(se)
        se = self.se_activation(se)
        se = self.se_fc2(se)
        se = torch.sigmoid(se)
        out = torch.einsum("abc,ab->abc", out, se)

        if self.downsample:
            identity = self.max_pool(identity)

        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1, -2)

        return out + identity


class BasicStage(nn.Module):
    """A sequence of ECGFounder residual blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ratio: float,
        kernel_size: int,
        stride: int,
        groups: int,
        i_stage: int,
        m_blocks: int,
        use_bn: bool = True,
        use_do: bool = True,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.groups = groups
        self.i_stage = i_stage
        self.m_blocks = m_blocks
        self.use_bn = use_bn
        self.use_do = use_do
        self.verbose = verbose

        self.block_list = nn.ModuleList()
        for i_block in range(m_blocks):
            is_first_block = i_stage == 0 and i_block == 0
            downsample = i_block == 0
            block_stride = stride if downsample else 1
            block_in_channels = in_channels if downsample else out_channels
            self.block_list.append(
                BasicBlock(
                    in_channels=block_in_channels,
                    out_channels=out_channels,
                    ratio=ratio,
                    kernel_size=kernel_size,
                    stride=block_stride,
                    groups=groups,
                    downsample=downsample,
                    is_first_block=is_first_block,
                    use_bn=use_bn,
                    use_do=use_do,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for i_block, block in enumerate(self.block_list):
            out = block(out)
            if self.verbose:
                print(
                    "stage: {}, block: {}, in_channels: {}, out_channels: {}, "
                    "outshape: {}".format(
                        self.i_stage,
                        i_block,
                        block.in_channels,
                        block.out_channels,
                        list(out.shape),
                    )
                )
        return out


class Net1D(nn.Module):
    """ECGFounder 1D network.

    Input: ``[B, C, T]``.
    If ``return_features=True``, returns ``(logits, deep_features)`` where
    ``deep_features`` is the global mean pooled 1024-dim representation.
    """

    def __init__(
        self,
        in_channels: int,
        base_filters: int,
        ratio: float,
        filter_list: list[int],
        m_blocks_list: list[int],
        kernel_size: int,
        stride: int,
        groups_width: int,
        n_classes: int,
        use_bn: bool = True,
        use_do: bool = True,
        return_features: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.base_filters = base_filters
        self.ratio = ratio
        self.filter_list = filter_list
        self.m_blocks_list = m_blocks_list
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups_width = groups_width
        self.n_stages = len(filter_list)
        self.n_classes = n_classes
        self.use_bn = use_bn
        self.use_do = use_do
        self.return_features = return_features
        self.verbose = verbose

        self.first_conv = MyConv1dPadSame(in_channels, base_filters, kernel_size, 2)
        self.first_bn = nn.BatchNorm1d(base_filters)
        self.first_activation = Swish()

        self.stage_list = nn.ModuleList()
        stage_in_channels = base_filters
        for i_stage, out_channels in enumerate(filter_list):
            self.stage_list.append(
                BasicStage(
                    in_channels=stage_in_channels,
                    out_channels=out_channels,
                    ratio=ratio,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=out_channels // groups_width,
                    i_stage=i_stage,
                    m_blocks=m_blocks_list[i_stage],
                    use_bn=use_bn,
                    use_do=use_do,
                    verbose=verbose,
                )
            )
            stage_in_channels = out_channels

        self.dense = nn.Linear(stage_in_channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        out = self.first_conv(x)
        if self.use_bn:
            out = self.first_bn(out)
        out = self.first_activation(out)

        for stage in self.stage_list:
            out = stage(out)

        deep_features = out.mean(-1)
        logits = self.dense(deep_features)

        if self.return_features:
            return logits, deep_features
        return logits
