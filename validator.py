from pathlib import Path

import numpy as np
from metrics import accuracy
import torch
from PIL import Image
from torchvision import datasets, transforms, models



class Validator():

    def __init__(self, val_folder : Path, val_interval : int = 2) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.val_folder = val_folder
        self.val_interval = val_interval
        self._prepare_data()


        
    def _prepare_data(self):
        # Define transformations to apply to the images
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load the dataset
        self.dataset = datasets.ImageFolder(self.val_folder, transform=self.transform)

    def validate_model(self, model, criterion):
        """
        Validate the model on the validation set.
        """
        
        assert model is not None, "Model is not defined"

        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=32, shuffle=True)

        model.eval()
        total_accuracy = 0.0
        running_loss = 0.0
        with torch.no_grad():
            for batch_i, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                batch_accuracy = accuracy(outputs, labels)
                total_accuracy += batch_accuracy

                running_loss += loss.item()

                if (batch_i + 1) % 10 == 0:
                    print(f"Val Step [{batch_i + 1}/{len(dataloader)}], Loss: {running_loss / 10:.4f} Accuracy: {total_accuracy / batch_i:.4f}")
                    running_loss = 0.0

            print(f"Validation accuracy: {total_accuracy / batch_i:.4f}")

    def test_on_folder(self, model : torch.nn.Module, test_folder : Path, output_file : Path):
        """
        Test the model on a folder of images and save the predictions to a csv file.
        """

        print(f"Testing model on {test_folder}")

        csv_output = "id," + ",".join(self._get_class_names()) + "\n"
        for img_path in test_folder.iterdir():
            if img_path.is_file():
                img_data = Image.open(img_path)
                img_tensor = self.transform(img_data)
                img_tensor = img_tensor.unsqueeze(0)
                img_tensor = img_tensor.to(self.device)
                outputs = model(img_tensor).squeeze().cpu().detach().numpy()
                csv_output += f"{img_path.stem},{self._prediction_to_csv_str(outputs)}\n"
        
        with open(output_file, "w") as f:
                f.write(csv_output)
        
        print(f"Saved predictions to {output_file}")

    def _get_class_names(self):
        return self.dataset.classes
    
    def _prediction_to_csv_str(self, predicted_prod : np.array, sep = ",") -> str:
        """
        turns array of prediction into a comma seperated string
        """
        result_str= ""
        for predicted_prob in predicted_prod:
            result_str += f"{predicted_prob:.4f}{sep}"
        return result_str[:-1] # remove last comma


