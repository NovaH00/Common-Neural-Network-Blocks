"""
Common Neural Network Blocks

This package contains various neural network blocks inspired by popular architectures
such as ResNet, ConvNeXt, Squeeze-and-Excitation, Bottleneck Transformer, DenseNet, 
Inception, Transformer Encoder, and CBAM (Convolutional Block Attention Module).

Each block is implemented as a PyTorch nn.Module and can be used to build more 
complex neural network models.
"""

from .architechtures import (
    ResNetBlock,
    ConvNeXtBlock,
    SEBlock,
    BottleneckTransformerBlock,
    DenseBlock,
    InceptionBlock,
    TransformerEncoderBlock,
    ChannelAttention,
    SpatialAttention,
    CBAM
)

__all__ = [
    "ResNetBlock",
    "ConvNeXtBlock", 
    "SEBlock",
    "BottleneckTransformerBlock",
    "DenseBlock",
    "InceptionBlock",
    "TransformerEncoderBlock",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM"
]

__version__ = "1.0.0"
__author__ = "Neural Network Blocks Contributors"
