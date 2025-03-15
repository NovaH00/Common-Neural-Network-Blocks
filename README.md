# Advanced Deep Learning Blocks

This repository contains implementations of several advanced deep learning building blocks in PyTorch, inspired by popular architectures like ResNet, ConvNeXt, Squeeze-and-Excitation Networks, and Bottleneck Transformers.

## Overview

The following blocks are implemented:

- **ResNetBlock**: A residual learning unit from the ResNet architecture.
- **ConvNeXtBlock**: Inspired by ConvNeXt, combining depthwise convolutions and pointwise transformations.
- **SEBlock**: Squeeze-and-Excitation block for channel-wise attention.
- **BottleneckTransformerBlock**: Combines convolutions with self-attention mechanisms.

Each block can be used as a modular component in larger neural network architectures.

## Installation

Ensure you have PyTorch installed:
```bash
pip install torch
```

## Usage

```python
import torch
from blocks import ResNetBlock, ConvNeXtBlock, SEBlock, BottleneckTransformerBlock

# Example: ResNet Block
resnet_block = ResNetBlock(64, 256)
x = torch.randn(1, 64, 32, 32)
output = resnet_block(x)
print(output.shape)  # Expected: torch.Size([1, 256, 32, 32])
```

## Block Details

### ResNetBlock
- **Input**: (B, C_in, H, W)
- **Output**: (B, C_out, H, W)
- Implements three convolutional layers with skip connections.

### ConvNeXtBlock
- **Input**: (B, C, H, W)
- **Output**: (B, C_out, H, W)
- Includes depthwise convolutions and layer normalization.

### SEBlock
- **Input**: (B, C, H, W)
- **Output**: (B, C, H, W)
- Recalibrates feature maps using squeeze-and-excitation mechanisms.

### BottleneckTransformerBlock
- **Input**: (B, C_in, H, W)
- **Output**: (B, C_out, H, W)
- Incorporates self-attention mechanisms for global feature learning.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Contact
For questions or collaboration, feel free to reach out.

---

Let me know if you'd like to add more sections or details!

