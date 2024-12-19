import tensorflow as tf
# Tensor đầu vào
input_tensor = tf.random.normal([1, 32, 32, 3])  # Batch size, Height, Width, Channels
filter_tensor = tf.random.normal([3, 3, 3, 64])  # Filter size (3x3), Input channels (3), Output channels (64)
max_pool_output = tf.nn.max_pool2d(
    input=input_tensor,
    ksize=[1, 2, 2, 1],   # Kích thước cửa sổ
    strides=[1, 2, 2, 1], # Bước nhảy
    padding='SAME'
)
print(max_pool_output.shape)