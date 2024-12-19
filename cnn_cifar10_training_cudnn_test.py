import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import time

# Load dataset CIFAR-10
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Định nghĩa mô hình CNN đơn giản
def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Bật/tắt cuDNN (giả lập không dùng cuDNN bằng cách sử dụng CPU)
def train_model(use_cudnn=True):
    if not use_cudnn:
        tf.config.experimental.set_visible_devices([], 'GPU')  # Tắt GPU

    model = create_cnn_model()
    start_time = time.time()
    model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=2)
    end_time = time.time()
    print(f"Thời gian huấn luyện (use_cudnn={use_cudnn}): {end_time - start_time:.2f} giây")

# Thực nghiệm
print("Thực nghiệm với cuDNN:")
train_model(use_cudnn=True)

print("\nThực nghiệm không dùng cuDNN:")
train_model(use_cudnn=False)
