import torch
import torch.nn as nn
import torch.optim as optim
import time

# Generate synthetic sequence data
def generate_sequence_data(seq_len, num_samples):
    x = torch.randn(num_samples, seq_len, 10)  # Random sequence data
    y = torch.randint(0, 2, (num_samples,))   # Random binary labels
    return x, y

# Define a simple RNN model
class SimpleRNN(nn.Module):
    def __init__(self):
        super(SimpleRNN, self).__init__()
        self.lstm = nn.LSTM(input_size=10, hidden_size=20, batch_first=True)
        self.fc = nn.Linear(20, 2)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out

# Experiment with and without cuDNN
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for use_cudnn in [False, True]:
    torch.backends.cudnn.enabled = use_cudnn
    model = SimpleRNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    x_train, y_train = generate_sequence_data(50, 1000)
    x_train, y_train = x_train.to(device), y_train.to(device)

    start_time = time.time()
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    duration = time.time() - start_time

    print(f"Time taken {'with' if use_cudnn else 'without'} cuDNN: {duration:.2f} seconds")
