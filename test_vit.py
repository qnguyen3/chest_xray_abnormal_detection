import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
import random
import os
import numpy as np
from tqdm import tqdm
import torch.optim as optim

from train_vit import read_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform_test = transforms.Compose([
    transforms.RandomResizedCrop(224), 
    transforms.ToTensor(),
    ])

def test(model, test_loader, criterion):
    model.to(device)
    test_accuracy = 0
    test_loss = 0
    for data, label in tqdm(test_loader):
        model.eval()
        data = data.to(device)
        label = label.float().to(device)
        output = model(data)
        output = output.float()
        output_size = output.size(0)
        output = output.float().reshape((output_size))
        test_loss = criterion(output, label)

        pred = torch.round(output.detach().cpu())
        target = torch.round(label.detach().cpu())
        acc = (pred == target).sum().float()
        test_accuracy += (acc / output_size) / len(test_loader)
        test_loss += test_loss / len(test_loader)

    print(f"test-loss : {test_loss:.4f} - test-acc: {test_accuracy:.4f}\n")

if __name__ == "__main__":
    #read in data
    default_data_path = "./chest_xray"
    test_data = read_data(default_data_path, 'test')
    test_loader = DataLoader(dataset=test_data, batch_size = 16)

    model = torch.load('best-mode.pt')
    criterion = nn.BCELoss()
    criterion.to(device)

    test(model, test_loader, criterion)