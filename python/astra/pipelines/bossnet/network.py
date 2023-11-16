"""
The BOSS Net model.

This file contains the Model object.

MIT License

Copyright (c) 2023 hutchresearch

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch 
from functools import partial

from astra.pipelines.bossnet.modules.conv_block_1d import ConvolutionBlock1D
from astra.pipelines.bossnet.modules.res_block_1d import ResNetBlock1D
from astra.pipelines.bossnet.modules.linear_block import LinearBlock
from astra.pipelines.bossnet.modules.positional_encodings import PositionalEncoding1D

class BossNetModel(torch.nn.Module):
    """
    A class representing the BossNet model for predicting stellar parameters from input spectra and metadata.

    The BossNet model consists of several convolutional blocks, residual blocks, and linear layers.

    Attributes:
    - metadatanet: torch.nn.Sequential, a sequential neural network module that takes metadata as input
      and produces an output tensor of shape (batch_size, 1024).
    - pos_enc: PositionalEncoding1D, an instance of the PositionalEncoding1D class used to apply
      positional encoding to the input spectra.
    - first_conv: ConvolutionBlock1D, a convolutional block used to extract features from the input spectra.
    - res_blocks: torch.nn.ModuleList, a list of residual blocks, where each block is an instance of the
    - ResNetBlock1D class, and each block is connected to the previous block's output tensor and the metadata
      tensor produced by metadatanet.
    - adapt_pool: torch.nn.AdaptiveAvgPool1d, an adaptive average pooling layer used to reduce the spatial
      dimensions of the output tensor from res_blocks.
    - linear_sequential: torch.nn.Sequential, a sequential neural network module that applies several fully
      connected layers to the output tensor from adapt_pool, ultimately producing an output tensor of shape
      (batch_size, 3).

    Methods:
    - forward(x): Performs a forward pass of the BossNet model given input spectra x and metadata m.
      Returns the predicted stellar parameters as a tensor of shape (batch_size, 3).

    Args:
    - x: torch.Tensor, Input tensor of shape (batch_size, 1, flux_length).

    Returns:
    - torch.Tensor, Output tensor of shape (batch_size, 4).
    """
    def __init__(self) -> None:
        super(BossNet, self).__init__()

        self.pos_enc = PositionalEncoding1D(1)

        self.first_conv = ConvolutionBlock1D(
            in_channels=2,
            out_channels=4,
            kernel_size=30,
            padding=15,
        )

        res_block = partial(ResNetBlock1D,
            kernel_size=30,
            padding=15,
            metadata_emb_dim=1024,
        )

        self.res_blocks = torch.nn.ModuleList([
            res_block(
                in_channels=4,
                out_channels=8,
            ),
            res_block(
                in_channels=8,
                out_channels=16,
            ),
            res_block(
                in_channels=16,
                out_channels=32,
            ),
            res_block(
                in_channels=32,
                out_channels=64,
            ),
        ])

        self.adapt_pool = torch.nn.AdaptiveAvgPool1d(output_size=1024)

        self.linear_sequential = torch.nn.Sequential(
            LinearBlock(64 * 1024, 1024),
            LinearBlock(1024, 1024),
            LinearBlock(1024, 1024),
            LinearBlock(1024, 1024),
            LinearBlock(1024, 1024),
            LinearBlock(1024, 1024),
            torch.nn.Linear(1024, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BossNet model.

        The forward pass takes in two tensors, `x` (spectra) and `m` (metadata), and produces a tensor
        representing the predicted stellar parameters.

        Args:
        - x: torch.Tensor, Input tensor of shape (batch_size, 1, flux length).
        - m:torch.Tensor, Metadata tensor of shape (batch_size, 14).

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, 4).
        """
        pos = self.pos_enc(x)
        x = torch.hstack([x, pos])
        out = self.first_conv(x)
        for res_block in self.res_blocks:
            out = res_block(out)
        out = self.adapt_pool(out)
        linear_input = out.view(-1, 64 * 1024)
        linear_output = self.linear_sequential(linear_input)
        return linear_output

