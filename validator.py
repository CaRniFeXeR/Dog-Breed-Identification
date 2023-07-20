from pathlib import Path
from metrics import accuracy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models



class Validator():

    def __init__(self, val_folder : Path, val_interval : int = 2) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.val_folder = val_folder
        self.val_interval = val_interval
        self._prepare_data()


        
    def _prepare_data(self):
        # Define transformations to apply to the images
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load the dataset
        self.dataset = datasets.ImageFolder(self.val_folder, transform=transform)

    def validate_model(self, model, criterion):
        
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=32, shuffle=True)

        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                batch_accuracy = accuracy(outputs, labels)
            
                total_accuracy += batch_accuracy
                total_batches += 1

                running_loss += loss.item()

                if (i + 1) % 10 == 0:
                    print(f"Step [{i + 1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}")
                    running_loss = 0.0

            print(f"Validation accuracy: {total_accuracy / total_batches:.4f}")

