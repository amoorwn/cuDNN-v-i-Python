import tensorflow as tf

# Kiểm tra xem GPU có hỗ trợ CUDA không
assert tf.test.is_gpu_available(cuda_only=True), "GPU with CUDA support is required for cuDNN."

# Cài đặt môi trường cuDNN
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:$LD_LIBRARY_PATH'

# Khởi tạo mô hình YOLOv3 với weights đã tải sẵn
yolo_model = tf.keras.applications.YOLOv3(weights='yolov3.h5')

# Sử dụng GPU với tf.device('/GPU:0')
with tf.device('/GPU:0'):
    yolo_model.build(input_shape=(None, None, 3))  # Thêm kích thước đầu vào cho ảnh
