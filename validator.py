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
                    print(f"Val Step [{batch_i + 1}/{len(dataloader)}], Loss: {running_loss / 10:.4f} Accuracy: {total_accuracy / (batch_i +1):.4f}")
                    running_loss = 0.0

            val_metric = total_accuracy / (batch_i + 1)
            print(f"Validation accuracy: {val_metric:.4f}")

            return val_metric

    def test_on_folder(self, model : torch.nn.Module, test_folder : Path, output_file : Path, batch_size : int = 64):
        """
        Test the model on a folder of images and save the predictions to a csv file.
        """

        if output_file.exists():
            print(f"Output file {output_file} already exists. Exiting...")
            return

        print(f"Testing model on {test_folder}")

        csv_output = "id," + ",".join(self._get_class_names()) + "\n"
        batch_img_tensors = []
        batch_img_paths = []
        batch_count = 0
        total_count = 0
        n_test_imgs = len(list(test_folder.iterdir()))
        for img_path in test_folder.iterdir():
            if img_path.is_file():
                img_data = Image.open(img_path)
                img_tensor = self.transform(img_data)
                img_tensor = img_tensor.unsqueeze(0)
                batch_img_tensors.append(img_tensor)
                batch_img_paths.append(img_path)

                if batch_count == batch_size or total_count == n_test_imgs - 1:
                    batch_count = 0
                    img_tensor = torch.cat(batch_img_tensors)
                    
                    img_tensor = img_tensor.to(self.device)
                    outputs =  torch.nn.functional.sigmoid(model(img_tensor)).cpu().detach().numpy()

                    for output, path in zip(outputs, batch_img_paths):
                        csv_line = f"{path.stem},{self._prediction_to_csv_str(output)}\n"
                        csv_output += csv_line
                   
                    batch_img_tensors = []
                    batch_img_paths = []

                batch_count += 1
                total_count += 1

        with open(output_file, "w") as f:
                f.write(csv_output)
        
        print(f"Saved predictions to {output_file}")

    def _get_class_names(self):
        return self.dataset.classes
    
    def _prediction_to_csv_str(self, pred_probs : np.array, sep = ",") -> str:
        """
        turns array of prediction into a comma seperated string
        """
        result_str= ""
        for class_prob in pred_probs:
            if class_prob < 0:
                class_prob *= -1
            result_str += f"{class_prob:.4f}{sep}"
        return result_str[:-1] # remove last comma


