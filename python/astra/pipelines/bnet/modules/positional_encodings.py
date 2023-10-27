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
import numpy as np

def get_emb(sin_inp: torch.Tensor) -> torch.Tensor:
    """
    Compute the embedding of a sine input tensor.

    The `sin_inp` tensor is first passed through the sine and cosine functions, and
    the resulting tensors are stacked along the last dimension to create an embedding.
    The embedding is then flattened along the last two dimensions.

    Args:
    sin_inp:torch.Tensor, Input tensor of shape (batch_size, sequence_length).

    Returns:
    torch.Tensor: Embedding tensor of shape (batch_size, sequence_length, 2).
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

class PositionalEncoding1D(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        """
        Initialize a 1D positional encoding layer.

        Parameters:
        - channels: int, The number of channels in the input tensor.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply 1D positional encoding to a 3D tensor.

        The input tensor is expected to have shape (batch_size, x, ch).

        Parameters:
        - tensor: torch.Tensor Input tensor of shape (batch_size, x, ch).

        Returns:
        - positional_encoding: torch.Tensor, Positional encoding tensor of shape (batch_size, x, ch).
        """
        tensor = tensor.permute(0, 2, 1)

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc.permute(0, 2, 1)

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc.permute(0, 2, 1)
    