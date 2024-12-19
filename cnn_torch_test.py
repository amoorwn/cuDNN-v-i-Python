import torch
import torch.nn as nn
# Tích chập 2D
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
input_tensor = torch.randn(1, 3, 32, 32)  # Batch size, Channels, Height, Width
output_tensor = conv(input_tensor)
print(output_tensor.shape)
