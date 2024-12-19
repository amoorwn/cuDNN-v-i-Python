import tensorflow as tf
# Tensor đầu vào
input_tensor = tf.random.normal([1, 32, 32, 3])  # Batch size, Height, Width, Channels
filter_tensor = tf.random.normal([3, 3, 3, 64])  # Filter size (3x3), Input channels (3), Output channels (64)
# Tích chập
conv_output = tf.nn.conv2d(
    input=input_tensor,
    filters=filter_tensor,
    strides=[1, 1, 1, 1],
    padding='SAME'
)
print(conv_output.shape)
