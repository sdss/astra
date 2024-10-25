"""
Linear Block for the Boss Net model

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

class LinearBlock(torch.nn.Module):
    def __init__(self, input_dim: int, linear_dim: int) -> None:
        """
        Linear block that applies a linear layer followed by a ReLU activation.

        Args:
        - input_dim: int, The dimensionality of the input tensor.
        - linear_dim: int, The output dimensionality of the linear layer.
        """
        super(LinearBlock, self).__init__()
        self.linear_layer = torch.nn.Linear(input_dim, linear_dim)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a linear transformation followed by a ReLU activation.

        Args:
        - x: torch.Tensor, The input tensor of shape (batch_size, input_dim).

        Returns:
        - torch.Tensor: The output tensor of shape (batch_size, linear_dim).
        """
        out = self.linear_layer(x)
        out = self.relu(out)
        return out