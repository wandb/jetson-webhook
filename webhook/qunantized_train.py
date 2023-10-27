import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.quantization
import wandb
import os
wandb.login(key=os.environ['WANDB_API_KEY'],force=True,relogin=True)

run = wandb.init(project='jetson_training',
                 entity='demonstrations')

artifact = wandb.Artifact(name='CIFAR_10',type='CIFAR_DATA')
artifact.add_dir('./data')

run.log_artifact(artifact)

# Define a smaller and quantization-aware CNN model
class SmallQuantizedCNN(nn.Module):
    def __init__(self):
        super(SmallQuantizedCNN, self).__init__()
        self.quant = torch.quantization.QuantStub()  # Quantization stub for inputs
        self.conv1 = nn.Conv2d(3, 4, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.fc1 = nn.Linear(8 * 6 * 6, 60)
        self.fc2 = nn.Linear(60, 40)
        self.fc3 = nn.Linear(40, 10)
        self.dequant = torch.quantization.DeQuantStub()  # De-quantization stub for outputs

    def forward(self, x):
        x = self.quant(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 8 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.dequant(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

# Initialize the smaller, quantization-aware model
model = SmallQuantizedCNN().to(device)
run.watch(model)
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')  # Configure for QAT
torch.quantization.prepare_qat(model, inplace=True)  # Prepare the model for QAT

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model for 10 epochs
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        run.log({'loss':loss})

        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# Convert to a quantized model
model.eval()
torch.quantization.convert(model, inplace=True)
