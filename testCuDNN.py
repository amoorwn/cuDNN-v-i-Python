import tensorflow as tf
tf.debugging.set_log_device_placement(True)
# Check if TensorFlow has GPU and cuDNN detection
print(tf.config.list_physical_devices('GPU'))