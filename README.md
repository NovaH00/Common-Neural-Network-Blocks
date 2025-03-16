# Block Architectures 

This module contains various neural network blocks inspired by popular architectures such as ResNet, ConvNeXt, Squeeze-and-Excitation, Bottleneck Transformer, DenseNet, Inception, and Transformer Encoder. Each block is implemented as a PyTorch `nn.Module` and can be used to build more complex neural network models.

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
from architechture import ResNetBlock
import torch

resnet_block = ResNetBlock(input_channel=64, output_channel=256)
x = torch.randn(1, 64, 32, 32)
output = resnet_block(x)
```

### ConvNeXtBlock

Inspired by the ConvNeXt architecture, this block simplifies the ResNet-style design with modern Transformer-like components. It includes depthwise convolutions, Layer Normalization, pointwise convolutions, and residual connections to enhance feature extraction.

**Usage:**

```python
from architechture import ConvNeXtBlock
import torch

convnext_block = ConvNeXtBlock(input_channel=64, output_channel=96)
x = torch.randn(1, 64, 32, 32)
output = convnext_block(x)
```

### SEBlock

A Squeeze-and-Excitation (SE) Block that recalibrates feature maps by modeling channel-wise dependencies. It consists of two main steps: Squeeze (global spatial information embedding) and Excitation (adaptive recalibration of feature maps).

**Usage:**

```python
from architechture import SEBlock
import torch

se_block = SEBlock(channels=64)
x = torch.randn(1, 64, 32, 32)
output = se_block(x)
```

### BottleneckTransformerBlock

Combines convolutional layers with self-attention mechanisms to capture both local and global features, inspired by the Bottleneck Transformer architecture.

**Usage:**

```python
from architechture import BottleneckTransformerBlock
import torch

block = BottleneckTransformerBlock(input_channel=64, hidden_dim=64, heads=4, output_channel=256)
x = torch.randn(1, 64, 32, 32)
output = block(x)
```

### DenseBlock

Implements a Dense Block from the DenseNet architecture, where each layer receives the concatenated feature maps from all previous layers, promoting feature reuse.

**Usage:**

```python
from architechture import DenseBlock
import torch

block = DenseBlock(input_channel=64, growth_rate=32, num_layers=4)
x = torch.randn(1, 64, 32, 32)
output = block(x)
```

### InceptionBlock

Implements an Inception Block, inspired by the Inception architecture. This block applies multiple convolutional layers with different kernel sizes in parallel and concatenates their outputs.

**Usage:**

```python
from architechture import InceptionBlock
import torch

block = InceptionBlock(input_channel=64, output_channel=256)
x = torch.randn(1, 64, 32, 32)
output = block(x)
```

### TransformerEncoderBlock

Implements a single Transformer Encoder Block as proposed in "Attention Is All You Need" (Vaswani et al.). This block consists of Multi-Head Self-Attention, Feedforward Neural Network (FFN), Residual connections, and Layer Normalization.

**Usage:**

```python
from architechture import TransformerEncoderBlock
import torch

encoder_block = TransformerEncoderBlock(input_dim=512, num_heads=8)
x = torch.randn(32, 10, 512)  # (batch_size, sequence_length, input_dim)
output = encoder_block(x)
```

## Combining Blocks

Here is an example of how to combine multiple blocks to create a more complex neural network model:

```python
from architechture import ResNetBlock, ConvNeXtBlock, SEBlock, BottleneckTransformerBlock, DenseBlock, InceptionBlock, TransformerEncoderBlock
import torch
import torch.nn as nn

class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.resnet_block = ResNetBlock(input_channel=64, output_channel=256)
        self.convnext_block = ConvNeXtBlock(input_channel=256, output_channel=96)
        self.se_block = SEBlock(channels=96)
        self.bottleneck_transformer_block = BottleneckTransformerBlock(input_channel=96, hidden_dim=64, heads=4, output_channel=256)
        self.dense_block = DenseBlock(input_channel=256, growth_rate=32, num_layers=4)
        self.inception_block = InceptionBlock(input_channel=384, output_channel=256)
        self.transformer_encoder_block = TransformerEncoderBlock(input_dim=256, num_heads=8)

    def forward(self, x):
        x = self.resnet_block(x)
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
