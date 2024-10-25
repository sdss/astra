"""
1D Convolutional Neural Network for the Boss Net model

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

class ConvolutionBlock1D(torch.nn.Module):
    """
    A 1D convolutional block, consisting of a 1D convolutional layer, batch normalization, ELU activation,
    and max pooling.

    Args:
    - in_channels: int, Number of input channels.
    - out_channels: int, Number of output channels.
    - kernel_size: int, Size of the kernel (filter) in the convolutional layer.
    - padding: int, Number of zero-padding elements added to both sides of the input.

    Attributes:
    - conv_layer: torch.nn.Conv1d, 1D convolutional layer.
    - bn: torch.nn.BatchNorm1d, Batch normalization layer.
    - elu: torch.nn.ELU, ELU activation function.
    - pool: torch.nn.MaxPool1d, Max pooling layer.

    Methods:
    - forward(x): Forward pass of the convolutional block.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        padding: int
    ) -> None:
        super(ConvolutionBlock1D, self).__init__()
        self.conv_layer = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.elu = torch.nn.ELU(inplace=True)
        self.pool = torch.nn.MaxPool1d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the convolutional block.

        Args:
        - x: torch.Tensor, Input tensor of shape (batch_size, in_channels, sequence_length).
        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, out_channels, sequence_length/2).
        """
        out = self.conv_layer(x)
        out = self.bn(out)
        out = self.elu(out)
        out = self.pool(out)
        return out