import torch

def accuracy(outputs, labels):
    _, predicted = torch.max(outputs, dim=1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy