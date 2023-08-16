import torch

class Model(torch.nn.Module):
    """
    #[참고] : https://team00csduck.tistory.com/232
    def __init__(self):
        super(Model, self).__init__()
        self.activation = torch.nn.ReLU(inplace=False)

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        
        self.flatten = torch.nn.Flatten()

        self.fc1 = torch.nn.Linear(in_features=32*(4*4), out_features=10, bias=False)

        
    def forward(self, x):
        x=self.conv1(x)
        x=self.activation(x)
        x=self.maxpool1(x)

        x=self.conv2(x)
        x=self.activation(x)
        x=self.maxpool2(x)
        x=self.flatten(x)
        
        x=self.fc1(x)

        return x
    """

    # [참고] TiFL
    def __init__(self):
        super(Model, self).__init__()
        self.activation = torch.nn.ReLU(inplace=False)

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2)

        self.fc1 = torch.nn.Linear(in_features=64*(12*12), out_features=128, bias=False)
        self.fc2 = torch.nn.Linear(in_features=128, out_features=10)

        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.5)

        self.flatten = torch.nn.Flatten()


    def forward(self, x):
        x=self.conv1(x)
        x=self.activation(x)

        x=self.conv2(x)
        x=self.activation(x)
        x=self.maxpool1(x)
        x=self.dropout1(x)

        x = x.view(-1, 64*12*12)

        x=self.fc1(x)
        x=self.activation(x)
        x=self.dropout2(x)

        x=self.fc2(x)
        

        return x
