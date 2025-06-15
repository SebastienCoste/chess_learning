# 🏗️ Detailed Architecture

---

## Overview

A comprehensive breakdown of the **EnhancedChessCNN** model architecture, including layer types, output shapes, parameter counts, and multiply-add operations.

---

## Architecture Table

| Layer (type:depth-idx) | Output Shape | Param \# | Mult-Adds |
| :-- | :-- | :-- | :-- |
| EnhancedChessCNN                          | [1, 4096]                  | --                         | --
| ├─PositionalEncoding2D: 1-1               | [1, 19, 8, 8]              | --                         | --
| ├─ModuleList: 1-2                         | --                         | --                         | --
| │    └─ResidualBlock: 2-1                 | [1, 64, 8, 8]              | --                         | --
| │    │    └─Conv2d: 3-1                   | [1, 64, 8, 8]              | 10,944                     | 700,416
| │    │    └─BatchNorm2d: 3-2              | [1, 64, 8, 8]              | 128                        | 128
| │    │    └─GELU: 3-3                     | [1, 64, 8, 8]              | --                         | --
| │    │    └─Dropout2d: 3-4                | [1, 64, 8, 8]              | --                         | --
| │    │    └─Conv2d: 3-5                   | [1, 64, 8, 8]              | 36,864                     | 2,359,296
| │    │    └─BatchNorm2d: 3-6              | [1, 64, 8, 8]              | 128                        | 128
| │    │    └─Sequential: 3-7               | [1, 64, 8, 8]              | 1,344                      | 77,952
| │    │    └─GELU: 3-8                     | [1, 64, 8, 8]              | --                         | --
| │    └─SpatialAttention: 2-2              | [1, 64, 8, 8]              | --                         | --
| │    │    └─AdaptiveAvgPool2d: 3-9        | [1, 64, 1, 1]              | --                         | --
| │    │    └─AdaptiveMaxPool2d: 3-10       | [1, 64, 1, 1]              | --                         | --
| │    │    └─Sequential: 3-11              | [1, 64]                    | 1,608                      | 1,608
| │    │    └─Sequential: 3-12              | [1, 1, 8, 8]               | 99                         | 6,336
| │    └─ResidualBlock: 2-3                 | [1, 128, 8, 8]             | --                         | --
| │    │    └─Conv2d: 3-13                  | [1, 128, 8, 8]             | 73,728                     | 4,718,592
| │    │    └─BatchNorm2d: 3-14             | [1, 128, 8, 8]             | 256                        | 256
| │    │    └─GELU: 3-15                    | [1, 128, 8, 8]             | --                         | --
| │    │    └─Dropout2d: 3-16               | [1, 128, 8, 8]             | --                         | --
| │    │    └─Conv2d: 3-17                  | [1, 128, 8, 8]             | 147,456                    | 9,437,184
| │    │    └─BatchNorm2d: 3-18             | [1, 128, 8, 8]             | 256                        | 256
| │    │    └─Sequential: 3-19              | [1, 128, 8, 8]             | 8,448                      | 524,544
| │    │    └─GELU: 3-20                    | [1, 128, 8, 8]             | --                         | --
| │    └─SpatialAttention: 2-4              | [1, 128, 8, 8]             | --                         | --
| │    │    └─AdaptiveAvgPool2d: 3-21       | [1, 128, 1, 1]             | --                         | --
| │    │    └─AdaptiveMaxPool2d: 3-22       | [1, 128, 1, 1]             | --                         | --
| │    │    └─Sequential: 3-23              | [1, 128]                   | 6,288                      | 6,288
| │    │    └─Sequential: 3-24              | [1, 1, 8, 8]               | 99                         | 6,336
| │    └─ResidualBlock: 2-5                 | [1, 256, 8, 8]             | --                         | --
| │    │    └─Conv2d: 3-25                  | [1, 256, 8, 8]             | 294,912                    | 18,874,368
| │    │    └─BatchNorm2d: 3-26             | [1, 256, 8, 8]             | 512                        | 512
| │    │    └─GELU: 3-27                    | [1, 256, 8, 8]             | --                         | --
| │    │    └─Dropout2d: 3-28               | [1, 256, 8, 8]             | --                         | --
| │    │    └─Conv2d: 3-29                  | [1, 256, 8, 8]             | 589,824                    | 37,748,736
| │    │    └─BatchNorm2d: 3-30             | [1, 256, 8, 8]             | 512                        | 512
| │    │    └─Sequential: 3-31              | [1, 256, 8, 8]             | 33,280                     | 2,097,664
| │    │    └─GELU: 3-32                    | [1, 256, 8, 8]             | --                         | --
| │    └─SpatialAttention: 2-6              | [1, 256, 8, 8]             | --                         | --
| │    │    └─AdaptiveAvgPool2d: 3-33       | [1, 256, 1, 1]             | --                         | --
| │    │    └─AdaptiveMaxPool2d: 3-34       | [1, 256, 1, 1]             | --                         | --
| │    │    └─Sequential: 3-35              | [1, 256]                   | 24,864                     | 24,864
| │    │    └─Sequential: 3-36              | [1, 1, 8, 8]               | 99                         | 6,336
| ├─ModuleList: 1-3                         | --                         | --                         | --
| │    └─ChessTransformerBlock: 2-7         | [1, 64, 256]               | --                         | --
| │    │    └─MultiheadAttention: 3-37      | [1, 64, 256]               | 263,168                    | 0
| │    │    └─LayerNorm: 3-38               | [1, 64, 256]               | 512                        | 512
| │    │    └─Sequential: 3-39              | [1, 64, 256]               | 525,568                    | 525,568
| │    │    └─LayerNorm: 3-40               | [1, 64, 256]               | 512                        | 512
| │    └─ChessTransformerBlock: 2-8         | [1, 64, 256]               | --                         | --
| │    │    └─MultiheadAttention: 3-41      | [1, 64, 256]               | 263,168                    | 0
| │    │    └─LayerNorm: 3-42               | [1, 64, 256]               | 512                        | 512
| │    │    └─Sequential: 3-43              | [1, 64, 256]               | 525,568                    | 525,568
| │    │    └─LayerNorm: 3-44               | [1, 64, 256]               | 512                        | 512
| ├─ModuleList: 1-4                         | --                         | --                         | --
| │    └─Linear: 2-9                        | [1, 512]                   | 8,389,120                  | 8,389,120
| │    └─GELU: 2-10                         | [1, 512]                   | --                         | --
| │    └─Dropout: 2-11                      | [1, 512]                   | --                         | --
| │    └─Linear: 2-12                       | [1, 256]                   | 131,328                    | 131,328
| │    └─GELU: 2-13                         | [1, 256]                   | --                         | --
| │    └─Dropout: 2-14                      | [1, 256]                   | --                         | --
| ├─Linear: 1-5                             | [1, 4096]                  | 1,052,672                  | 1,052,672


---

## Notes

- **GELU**: Gaussian Error Linear Unit activation
- **Dropout/Dropout2d**: Regularization layers
- **BatchNorm2d**: Batch normalization for 2D inputs
- **SpatialAttention**: Attention mechanism over spatial dimensions
- **MultiheadAttention**: Transformer-style attention
- **LayerNorm**: Layer normalization
- **AdaptiveAvgPool2d/AdaptiveMaxPool2d**: Adaptive pooling layers
- **Sequential**: Sequential container for layers
- **Linear**: Fully connected (dense) layers

---

This Markdown faithfully represents the detailed architecture as described in the provided file[^1].

<div style="text-align: center">⁂</div>

[^1]: architecture.txt

