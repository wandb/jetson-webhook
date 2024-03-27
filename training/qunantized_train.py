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

class TrainModel:
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=20, shuffle=True)
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=4, shuffle=False)
        self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')  
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        torch.quantization.prepare_qat(model, inplace=True)  
        wandb.login(key=os.environ['WANDB_API_KEY'], force=True, relogin=True)

    def mean_rgb(self,image):
        return image.mean(dim=[1, 2]).tolist()  
    
    def wandb_login(self):
        wandb.login(key=os.environ['WANDB_API_KEY'], force=True, relogin=True)
    
    def wandb_init(self):
        self.run = wandb.init(project='quantized edge training', entity='tiny-ml')
    
    def create_wandb_artifact(self):
        self.artifact = wandb.Artifact(name='CIFAR_10', type='CIFAR_DATA')
        self.artifact.add_dir('./data')
        self.run.log_artifact(self.artifact)

    def compute_accuracy(self,model, dataloader, device):
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

    def train(self):
        self.run.watch(self.model)
        self.model.to(self.device)
        for epoch in range(30):
            if hasattr(self.model, 'quant'):
                self.model.quant.to(self.device)
            if hasattr(self.model, 'dequant'):
                self.model.dequant.to(self.device)
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()


                running_loss += loss.item()
                acc =  self.compute_accuracy(self.model, self.testloader, self.device)
                self.run.log({'step_accuracy': acc})
                if i % 500 == 0:  # Choose a suitable frequency
                    table = wandb.Table(columns=["Image", "Label", "Mean R", "Mean G", "Mean B"])
                    for j in range(len(inputs)):
                        image = inputs[j].cpu().numpy().transpose(1, 2, 0)  # Move to cpu and convert CHW to HWC
                        label = labels[j].item()
                        r_mean, g_mean, b_mean = self.mean_rgb(inputs[j])
                        table.add_data(wandb.Image(image), label, r_mean, g_mean, b_mean)
                    self.run.log({"Image Table": table}, commit=False)  # Log table

                if i % 2000 == 1999:
                    accuracy = self.compute_accuracy(self.model, self.testloader, self.device)
                    self.run.log({'epoch': epoch, 'loss': loss.item(), 'accuracy': accuracy})
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
                    weights_to_save = deepcopy(self.model.cpu().state_dict())
                    torch.save(weights_to_save, 'quantized_model.pth')
                    print('Model Saved')
                    self.model.to(self.device)
                    artifact = wandb.Artifact(name='quantized_model', type='model')
                    artifact.add_file('quantized_model.pth')
                    self.run.log_artifact(artifact)
                    
            print(f"Epoch {epoch+1}, Accuracy: {accuracy}%")





