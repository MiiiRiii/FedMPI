import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.activation = torch.nn.ReLU(inplace=False)

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(
            5, 5), padding=1, stride=1, bias=False)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(
            5, 5), padding=1, stride=1, bias=False)

        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.flatten = torch.nn.Flatten()

        self.fc1 = torch.nn.Linear(
            in_features=64*(7*7), out_features=512, bias=False)
        self.fc2 = torch.nn.Linear(
            in_features=512, out_features=10, bias=False)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.maxpool1(x)

        x = self.activation(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten(x)

        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return x
