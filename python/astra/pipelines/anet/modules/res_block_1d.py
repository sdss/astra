"""
1D Residule Neural Network for the Boss Net model

This file contains supporting code for Boss Net the model.

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

class ResNetBlock1D(torch.nn.Module):
    """
    A ResNet block for 1D convolutional neural networks.
    
    Args:
    - in_channels: int, the number of channels in the input tensor.
    - out_channels: int, the number of channels in the output tensor.
    - kernel_size: int, the size of the convolutional kernel.
    - padding: int, the number of padding pixels in the convolution.
    - metadata_emb_dim: int, the dimensionality of the metadata embedding.
    
    Attributes:
    - in_channels: int, the number of channels in the input tensor.
    - out_channels: int, the number of channels in the output tensor.
    - mlp: torch.nn.Sequential, a multi-layer perceptron that transforms metadata embeddings into learnable scales and shifts.
    - conv1: torch.nn.Conv1d, a 1D convolutional layer that convolves the input tensor.
    - bn1: torch.nn.BatchNorm1d, a batch normalization layer that normalizes the output tensor of conv1.
    - elu: torch.nn.ELU, an activation function that applies the Exponential Linear Unit function element-wise.
    - conv2: torch.nn.Conv1d, a 1D convolutional layer that convolves the output tensor of conv1.
    - bn2: torch.nn.BatchNorm1d, a batch normalization layer that normalizes the output tensor of conv2.
    
    Methods:
    - forward(x, m_emb): computes the forward pass of the ResNet block.
    
    Returns:
    - out: torch.Tensor, the output tensor of the ResNet block.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        padding: int, 
        metadata_emb_dim: int,
    ) -> None:
        super(ResNetBlock1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.bn1 = torch.nn.BatchNorm1d(out_channels)
        self.elu = torch.nn.ELU(inplace=True)
        padding_conv2 = padding - 1 if (kernel_size % 2 == 0) else padding
        self.conv2 = torch.nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding_conv2,
            bias=False,
        )

        self.bn2 = torch.nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass of the ResNet block.
        
        Args:
        - x: torch.Tensor, the input tensor to the ResNet block.
        - m_emb: torch.Tensor, the metadata embedding tensor to the ResNet block.
        
        Returns:
        - out: torch.Tensor, the output tensor of the ResNet block.
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.out_channels != self.in_channels:
            residual = residual.transpose(-1, -2)
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            residual = torch.nn.functional.pad(residual, (ch1, ch2), "constant", 0)
            residual = residual.transpose(-1, -2)

        out += residual
        out = self.elu(out)

        return out
