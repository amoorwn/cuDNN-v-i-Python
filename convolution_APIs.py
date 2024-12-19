import numpy as np
import cupy as cp
import cudnn

# Tạo context cuDNN
cudnn_context = cudnn.create()

# Định nghĩa tensor đầu vào
input_desc = cudnn.TensorDescriptor()
input_tensor = cp.random.randn(1, 3, 32, 32).astype(cp.float32)
input_desc.set(input_tensor.shape, cudnn.CUDNN_TENSOR_NCHW, cudnn.CUDNN_DATA_FLOAT)

# Định nghĩa filter
filter_desc = cudnn.FilterDescriptor()
filter_tensor = cp.random.randn(16, 3, 3, 3).astype(cp.float32)
filter_desc.set(filter_tensor.shape, cudnn.CUDNN_TENSOR_NCHW, cudnn.CUDNN_DATA_FLOAT)

# Định nghĩa convolution descriptor
conv_desc = cudnn.ConvolutionDescriptor()
conv_desc.set(
    pad=(1, 1), stride=(1, 1), dilation=(1, 1), mode=cudnn.CUDNN_CONVOLUTION
)

# Forward convolution
output_desc = cudnn.TensorDescriptor()
output_shape = cudnn.get_convolution_output_shape(input_desc, filter_desc, conv_desc)
output_tensor = cp.zeros(output_shape, dtype=cp.float32)
output_desc.set(output_tensor.shape, cudnn.CUDNN_TENSOR_NCHW, cudnn.CUDNN_DATA_FLOAT)
cudnn.convolution_forward(
    cudnn_context,
    input_desc, input_tensor,
    filter_desc, filter_tensor,
    conv_desc,
    output_desc, output_tensor
)
print("Output shape:", output_tensor.shape)
