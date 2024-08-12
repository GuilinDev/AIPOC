import torch
import torch.nn as nn


# Define the SimpleLSTM class as provided earlier
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Only take the last output in the sequence
        return out


# Create an instance of the SimpleLSTM model
input_size = 10  # Number of features in the input
hidden_size = 20  # Number of features in the hidden state
output_size = 1  # Number of features in the output

model = SimpleLSTM(input_size, hidden_size, output_size)

# Generate some sample input data
# Example input: batch of 5 sequences, each sequence of length 3, each step having 10 features
sample_input = torch.randn(5, 3, input_size)  # Batch size 5, sequence length 3

# Pass the input data through the model
output = model(sample_input)

# Print the output
print("Model output:\n", output)
