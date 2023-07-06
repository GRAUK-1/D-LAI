import torch
from constants import OUTPUT_SIZE
import torch.nn as nn
import numpy as np


class SimpleModel(nn.Module):
    def __init__(self, input_shape):  # Remove the default value
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=5, stride=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=2), nn.BatchNorm2d(64), nn.ReLU(),
        )

        # Create a dummy input with the correct input shape
        dummy_input = torch.zeros(1, *input_shape)

        # Pass the dummy input through the conv layers to get the output size
        conv_output = self.conv(dummy_input)

        # Calculate the number of features from the output size
        num_features = int(np.prod(conv_output.size()))

        self.fc = nn.Sequential(
            nn.Linear(num_features, 512), nn.ReLU(),
            nn.Linear(512, OUTPUT_SIZE)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        return self.fc(x)


class CuriosityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.inverse_model = nn.Sequential(
            nn.Linear(4480*2, 512), nn.ReLU(),
            nn.Linear(512, OUTPUT_SIZE)
        )
        self.forward_model = nn.Sequential(
            nn.Linear(4480+OUTPUT_SIZE, 512), nn.ReLU(),
            nn.Linear(512, 4480)
        )

    def forward(self, state, action, next_state):
        state_action = torch.cat([state, action], 1)
        predicted_next_state = self.forward_model(state_action)

        state_next_state = torch.cat([state, next_state], 1)
        predicted_action = self.inverse_model(state_next_state)

        return predicted_action, predicted_next_state
