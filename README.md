# Common Neural Network Blocks 

This module contains various neural network blocks inspired by popular architectures such as ResNet, ConvNeXt, Squeeze-and-Excitation, Bottleneck Transformer, DenseNet, Inception, Transformer Encoder, and CBAM (Convolutional Block Attention Module). Each block is implemented as a PyTorch `nn.Module` and can be used to build more complex neural network models.

## Prerequisites

This module should work with any version of PyTorch. Ensure you have PyTorch installed in your environment. You can install PyTorch using the following command:

```bash
pip install torch
```

## Blocks

### ResNetBlock

A residual learning unit from the ResNet architecture. It consists of three convolutional layers, each followed by Batch Normalization and ReLU activation, along with a skip connection to enable residual learning.

**Usage:**

```python
from common_neural_network_blocks import ResNetBlock
import torch

resnet_block = ResNetBlock(input_channel=64, output_channel=256)
x = torch.randn(1, 64, 32, 32)
output = resnet_block(x)
```

Number of parameters ≈ input_channel × 64 + 64² × 9 + 64 × output_channel

### ConvNeXtBlock

Inspired by the ConvNeXt architecture, this block simplifies the ResNet-style design with modern Transformer-like components. It includes depthwise convolutions, Layer Normalization, pointwise convolutions, and residual connections to enhance feature extraction.

**Usage:**

```python
from common_neural_network_blocks import ConvNeXtBlock
import torch

convnext_block = ConvNeXtBlock(input_channel=64, output_channel=96)
x = torch.randn(1, 64, 32, 32)
output = convnext_block(x)
```

Number of parameters ≈ input_channel × 49 + input_channel × 384 + 384 × output_channel

### SEBlock

A Squeeze-and-Excitation (SE) Block that recalibrates feature maps by modeling channel-wise dependencies. It consists of two main steps: Squeeze (global spatial information embedding) and Excitation (adaptive recalibration of feature maps).

**Usage:**

```python
from common_neural_network_blocks import SEBlock
import torch

se_block = SEBlock(channels=64)
x = torch.randn(1, 64, 32, 32)
output = se_block(x)
```

Number of parameters ≈ channels × (channels // reduction) × 2 (default reduction=16)

### BottleneckTransformerBlock

Combines convolutional layers with self-attention mechanisms to capture both local and global features, inspired by the Bottleneck Transformer architecture.

**Usage:**

```python
from common_neural_network_blocks import BottleneckTransformerBlock
import torch

block = BottleneckTransformerBlock(input_channel=64, hidden_dim=64, heads=4, output_channel=256)
x = torch.randn(1, 64, 32, 32)
output = block(x)
```

Number of parameters ≈ input_channel × hidden_dim + hidden_dim² × 3 + hidden_dim × output_channel

### DenseBlock

Implements a Dense Block from the DenseNet architecture, where each layer receives the concatenated feature maps from all previous layers, promoting feature reuse.

**Usage:**

```python
from common_neural_network_blocks import DenseBlock
import torch

block = DenseBlock(input_channel=64, growth_rate=32, num_layers=4)
x = torch.randn(1, 64, 32, 32)
output = block(x)
```

Number of parameters ≈ $\sum_{i=0}^{\text{num\_layers}-1} (\text{input\_channel} + i \times \text{growth\_rate}) \times \text{growth\_rate} \times 9$

### InceptionBlock

Implements an Inception Block, inspired by the Inception architecture. This block applies multiple convolutional layers with different kernel sizes in parallel and concatenates their outputs.

**Usage:**

```python
from common_neural_network_blocks import InceptionBlock
import torch

block = InceptionBlock(input_channel=64, output_channel=256)
x = torch.randn(1, 64, 32, 32)
output = block(x)
```

Number of parameters ≈ input_channel × output_channel × (1 + 9 + 25) / 4

### TransformerEncoderBlock

Implements a single Transformer Encoder Block as proposed in "Attention Is All You Need" (Vaswani et al.). This block consists of Multi-Head Self-Attention, Feedforward Neural Network (FFN), Residual connections, and Layer Normalization.

**Usage:**

```python
from common_neural_network_blocks import TransformerEncoderBlock
import torch

encoder_block = TransformerEncoderBlock(input_dim=512, num_heads=8)
x = torch.randn(32, 10, 512)  # (batch_size, sequence_length, input_dim)
output = encoder_block(x)
```

Number of parameters ≈ input_dim² × 4 + input_dim × ff_hidden_dim × 2 (default ff_hidden_dim=2048)

### ChannelAttention

Channel Attention Module (CAM) from CBAM that implements channel attention by aggregating spatial information through both average and max pooling, then using a shared MLP to compute channel-wise attention weights.

**Usage:**

```python
from common_neural_network_blocks import ChannelAttention
import torch

channel_attn = ChannelAttention(64, ratio=16)
x = torch.randn(1, 64, 32, 32)
attention_weights = channel_attn(x)  # Output shape: (1, 64, 1, 1)
```

Number of parameters ≈ in_planes × (in_planes // ratio) × 2 (default ratio=16)

### SpatialAttention

Spatial Attention Module (SAM) from CBAM that implements spatial attention by aggregating channel information through both average and max pooling along the channel dimension, then using a convolutional layer to compute spatial attention weights.

**Usage:**

```python
from common_neural_network_blocks import SpatialAttention
import torch

spatial_attn = SpatialAttention(kernel_size=7)
x = torch.randn(1, 64, 32, 32)
attention_weights = spatial_attn(x)  # Output shape: (1, 1, 32, 32)
```

Number of parameters ≈ 2 × kernel_size² (default kernel_size=7)

### CBAM

Convolutional Block Attention Module (CBAM) that combines both channel and spatial attention to improve feature representation. It sequentially applies channel attention followed by spatial attention to refine feature maps.

**Usage:**

```python
from common_neural_network_blocks import CBAM
import torch

cbam = CBAM(64, ratio=16, kernel_size=7)
x = torch.randn(1, 64, 32, 32)
output = cbam(x)  # Output shape: (1, 64, 32, 32) - same as input but with attention applied
```

Number of parameters ≈ in_planes × (in_planes // ratio) × 2 + 2 × kernel_size²

## Combining Blocks

Here is an example of how to combine multiple blocks to create a more complex neural network model:

```python
from common_neural_network_blocks import (
    ResNetBlock, ConvNeXtBlock, SEBlock, BottleneckTransformerBlock, 
    DenseBlock, InceptionBlock, TransformerEncoderBlock, CBAM
)
import torch
import torch.nn as nn

class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.resnet_block = ResNetBlock(input_channel=64, output_channel=256)
        self.cbam = CBAM(in_planes=256, ratio=16, kernel_size=7)
        self.convnext_block = ConvNeXtBlock(input_channel=256, output_channel=96)
        self.se_block = SEBlock(channels=96)
        self.bottleneck_transformer_block = BottleneckTransformerBlock(input_channel=96, hidden_dim=64, heads=4, output_channel=256)
        self.dense_block = DenseBlock(input_channel=256, growth_rate=32, num_layers=4)
        self.inception_block = InceptionBlock(input_channel=384, output_channel=256)
        self.transformer_encoder_block = TransformerEncoderBlock(input_dim=256, num_heads=8)

    def forward(self, x):
        x = self.resnet_block(x)
        x = self.cbam(x)  # Apply CBAM attention
        x = self.convnext_block(x)
        x = self.se_block(x)
        x = self.bottleneck_transformer_block(x)
        x = self.dense_block(x)
        x = self.inception_block(x)
        x = x.flatten(2).permute(0, 2, 1)  # Reshape for Transformer Encoder
        x = self.transformer_encoder_block(x)
        return x

# Example usage
model = CombinedModel()
x = torch.randn(1, 64, 32, 32)
output = model(x)
print(output.shape)
```

## Installation

You can use this package by placing it in your project directory and importing the required blocks:

```python
# Option 1: Import specific blocks
from common_neural_network_blocks import ResNetBlock, CBAM, SEBlock

# Option 2: Import all blocks
from common_neural_network_blocks import *

# Option 3: Import the module
import common_neural_network_blocks as nnb
resnet_block = nnb.ResNetBlock(64, 256)
```

## Features

- **Modular Design**: Each block is self-contained and can be used independently
- **PyTorch Integration**: All blocks inherit from `nn.Module` for seamless integration
- **Attention Mechanisms**: Includes state-of-the-art attention modules (CBAM, SE, etc.)
- **Well Documented**: Comprehensive docstrings with usage examples and shape information
- **Flexible**: Configurable parameters for different use cases

## References

- **ResNet**: Deep Residual Learning for Image Recognition (He et al., CVPR 2016)
- **ConvNeXt**: A ConvNet for the 2020s (Liu et al., CVPR 2022)
- **SE-Net**: Squeeze-and-Excitation Networks (Hu et al., CVPR 2018)
- **CBAM**: Convolutional Block Attention Module (Woo et al., ECCV 2018)
- **DenseNet**: Densely Connected Convolutional Networks (Huang et al., CVPR 2017)
- **Inception**: Going Deeper with Convolutions (Szegedy et al., CVPR 2015)
- **Transformer**: Attention Is All You Need (Vaswani et al., NeurIPS 2017)
