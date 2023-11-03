import os
import numpy as np
import shutil
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
   

def split_and_save_dataset(root='./data', 
                           new_root='./cifar_2', 
                           split_ratio=0.2,
                           transform=None):
    dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    num_samples = len(dataset)
    indices = list(range(num_samples))
    
    # Randomly splitting indices:
    split = int(np.floor(split_ratio * num_samples))
    np.random.shuffle(indices)
    
    new_indices, train_indices = indices[:split], indices[split:]
    
    # Creating new directory to save the subset of the dataset
    if not os.path.exists(new_root):
        os.makedirs(new_root)
    
    # Moving the 20% of the data to the new directory
    for idx in new_indices:
        src_path = os.path.join(root, f"{idx}")
        dst_path = os.path.join(new_root, f"{idx}")
        
        shutil.move(src_path, dst_path)  # Moving the files
    return ImageFolder(root=root, transform=transform)



