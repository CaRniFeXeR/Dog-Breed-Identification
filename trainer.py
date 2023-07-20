from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from validator import Validator


class Trainer():

    def __init__(self, train_folder : Path, save_folder : Path, augmetation_transforms) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_folder = train_folder
        self.save_folder = save_folder
        self.augmetation_transforms = augmetation_transforms
        self._prepare_data()
        self._prepare_model()

    def _prepare_data(self):
        # Define transformations to apply to the images
        transform_steps = [
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ]

        transform_steps.extend(self.augmetation_transforms)

        transform_steps.append(transforms.ToTensor())
        transform_steps.append( transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        
        transform = transforms.Compose(transform_steps)

        # Load the dataset
        self.dataset = datasets.ImageFolder(self.train_folder, transform=transform)

    def _prepare_model(self):
         # Load pre-trained ResNet model
        self.model = models.resnet50(pretrained=True)
        num_classes = len(self.dataset.classes)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model = self.model.to(self.device)

    def load_model_from_file(self, file : Path):
        self.model.load_state_dict(torch.load(file))
        self.model = self.model.to(self.device)

    def train(self, validator : Validator):
        
        
        # Create data loader
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=32, shuffle=True)
    

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        # Training loop
        num_epochs = 10
        best_val_score = 0.0
        for epoch in range(num_epochs):
            running_loss = 0.0
            self.model.train()
            for i, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if (i + 1) % 10 == 0:
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}")
                    running_loss = 0.0
            
            if (epoch + 1) % validator.val_interval == 0:
                val_score = validator.validate_model(self.model, criterion)

                if val_score > best_val_score:
                    best_val_score = val_score
                    torch.save(self.model.state_dict(), self.save_folder / "best_model.pt")
                    print("Best model saved!")

        print("Training complete!")
