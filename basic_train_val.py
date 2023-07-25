from trainer import Trainer
from validator import Validator
from pathlib import Path

from torchvision import transforms

data_augementation_steps = [
    transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    transforms.GaussianBlur(3),
    transforms.RandomPerspective(distortion_scale=0.3),
    transforms.RandomAffine(degrees=30, translate=(0.2, 0.2)),
    transforms.ColorJitter(brightness=0.5, contrast=0.4, saturation=0.4, hue=0.1)
]

trainer = Trainer(train_folder=Path("./data/train_valid_test/train"),
                 save_folder=Path("./models"),
                    n_epochs=25,
                    batch_size=32,
                    augmetation_transforms=data_augementation_steps
                 )
validator = Validator(val_folder=Path("./data/train_valid_test/valid"), val_interval=1)



print("start training ...")
trainer.train(validator)
trainer.load_model_from_file(Path("./models/best_model.pt"))
validator.test_on_folder(trainer.model, test_folder=Path("./data/test"), output_file=Path("test_results.csv"), batch_size=32)


