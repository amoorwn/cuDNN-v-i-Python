import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample text data
texts = ["I love this product", "This is a bad experience", "Amazing quality!", "Terrible service."]
labels = [1, 0, 1, 0]

# Tokenization and padding
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=10)

# Build LSTM model optimized with cuDNN
model = Sequential([
    Embedding(input_dim=1000, output_dim=64, input_length=10),
    LSTM(128, activation='tanh', recurrent_activation='sigmoid', use_bias=True),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, labels, epochs=10, batch_size=2)

# Example: Predict sentiment
test_texts = ["Great product", "Worst experience ever"]
test_sequences = tokenizer.texts_to_sequences(test_texts)
test_padded = pad_sequences(test_sequences, maxlen=10)
predictions = model.predict(test_padded)
print(["Positive" if p > 0.5 else "Negative" for p in predictions])
