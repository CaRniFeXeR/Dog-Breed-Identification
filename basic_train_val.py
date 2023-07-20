from trainer import Trainer
from validator import Validator
from pathlib import Path

from torchvision import transforms

data_augementation_steps = [
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
]

trainer = Trainer(train_folder=Path("./data/train_valid_test/train"), save_folder=Path("./models"))
validator = Validator(val_folder=Path("./data/train_valid_test/valid"), val_interval=1)



print("start training ...")
trainer.train(validator)
trainer.load_model_from_file(Path("./models/best_model.pt"))
validator.test_on_folder(trainer.model, test_folder=Path("./data/test"), output_file=Path("test_results.csv"))


