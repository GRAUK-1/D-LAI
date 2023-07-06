import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define the directory to save the screenshots
screenshot_dir = 'screenshots/'

# Define the data transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((80, 80)),
    transforms.ToTensor()
])

# Load the data
data = datasets.ImageFolder(screenshot_dir, transform=transform)
data_loader = DataLoader(data, batch_size=32, shuffle=True)

# Define the model
model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(64 * 20 * 20, 128),
    nn.ReLU(),
    nn.Linear(128, 2)
)

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Train the model
for epoch in range(10):
    for imgs, labels in data_loader:
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch}, Loss: {loss.item()}')
