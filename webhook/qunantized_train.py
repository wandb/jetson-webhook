import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.quantization
import wandb
import os
from copy import deepcopy

wandb.login(key=os.environ['WANDB_API_KEY'], force=True, relogin=True)

run = wandb.init(project='quantized edge training', entity='tiny-ml')


artifact = wandb.Artifact(name='CIFAR_10', type='CIFAR_DATA')
artifact.add_dir('./data')
run.log_artifact(artifact)


class SmallQuantizedCNN(nn.Module):
    def __init__(self):
        super(SmallQuantizedCNN, self).__init__()
        self.quant = torch.quantization.QuantStub()  
        self.conv1 = nn.Conv2d(3, 4, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.fc1 = nn.Linear(8 * 6 * 6, 60)
        self.fc2 = nn.Linear(60, 40)
        self.fc3 = nn.Linear(40, 10)
        self.dequant = torch.quantization.DeQuantStub()  

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


def mean_rgb(image):
    return image.mean(dim=[1, 2]).tolist()  

def compute_accuracy(model, dataloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

model = SmallQuantizedCNN().to(device)
run.watch(model)
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')  
torch.quantization.prepare_qat(model, inplace=True)  

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(30):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 500 == 0:  # Choose a suitable frequency
            table = wandb.Table(columns=["Image", "Label", "Mean R", "Mean G", "Mean B"])
            for j in range(len(inputs)):
                image = inputs[j].cpu().numpy().transpose(1, 2, 0)  # Move to cpu and convert CHW to HWC
                label = labels[j].item()
                r_mean, g_mean, b_mean = mean_rgb(inputs[j])
                table.add_data(wandb.Image(image), label, r_mean, g_mean, b_mean)
            run.log({"Image Table": table}, commit=False)  # Log table

        if i % 2000 == 1999:
            accuracy = compute_accuracy(model, testloader, device)
            run.log({'epoch': epoch, 'loss': loss.item(), 'accuracy': accuracy})
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            weights_to_save = deepcopy(model.cpu().state_dict())
            torch.save(weights_to_save, 'quantized_model.pth')
            print('Model Saved')
            model.to(device)
            artifact = wandb.Artifact(name='quantized_model', type='model')
            artifact.add_file('quantized_model.pth')
            run.log_artifact(artifact)
            
    print(f"Epoch {epoch+1}, Accuracy: {accuracy}%")





