from trainer import Trainer
from validator import Validator


trainer = Trainer(train_folder="./data/train_valid_test/train")
validator = Validator(val_folder="./data/train_valid_test/valid", val_interval=2)

print("start training ...")
trainer.train(validator)

