import math
import collections
import shutil
import os
from typing import List

def mkdir_if_not_exist(path_list : List[str]):
    """Make a directory if it does not exist."""
    path = os.path.join(*path_list)
    if not os.path.exists(path):
        os.makedirs(path)

def reorg_train_valid(data_dir, train_dir, input_dir, valid_ratio, idx_label):
  # The number of examples of the least represented breed in the training set.
  min_n_train_per_label = (
      collections.Counter(idx_label.values()).most_common()[:-2:-1][0][1])
  
  # The number of examples of each breed in the validation set.
  n_valid_per_label = math.floor(min_n_train_per_label * valid_ratio)
  label_count = {}
  for train_file in os.listdir(os.path.join(data_dir, train_dir)):
    idx = train_file.split('.')[0]
    label = idx_label[idx]

    mkdir_if_not_exist([data_dir, input_dir, 'train_valid', label])
    
    shutil.copy(os.path.join(data_dir, train_dir, train_file),
                os.path.join(data_dir, input_dir, 'train_valid', label))
    
    if label not in label_count or label_count[label] < n_valid_per_label:
      mkdir_if_not_exist([data_dir, input_dir, 'valid', label])
      shutil.copy(os.path.join(data_dir, train_dir, train_file),
                  os.path.join(data_dir, input_dir, 'valid', label))
      label_count[label] = label_count.get(label, 0) + 1
      
    else:
      mkdir_if_not_exist([data_dir, input_dir, 'train', label])
      shutil.copy(os.path.join(data_dir, train_dir, train_file),
                  os.path.join(data_dir, input_dir, 'train', label))
      
def reorg_dog_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio):
  # Read the training data labels.
  with open(os.path.join(data_dir, label_file), 'r') as f:
    # Skip the file header line (column name).
    lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    idx_label = dict(((idx, label) for idx, label in tokens))
  
  reorg_train_valid(data_dir, train_dir, input_dir, valid_ratio, idx_label)

  # Organize the training set.
  mkdir_if_not_exist([data_dir, input_dir, 'test', 'unknown'])
  for test_file in os.listdir(os.path.join(data_dir, test_dir)):
    shutil.copy(os.path.join(data_dir, test_dir, test_file),
                os.path.join(data_dir, input_dir, 'test', 'unknown'))
    
if __name__ == '__main__':
  """
  we define the reorg_train_valid function to split the validation set from the original Kaggle competition training set. 
  The parameter valid_ratio in this function is the ratio of the number of examples of each dog breeds in the validation set to the number of examples of the
  breed with the least examples (66) in the original training set. 
  After organizing the data, images of the same breed will be placed in the same folder so that we can read them later.
  """
  data_dir, label_file, train_dir, test_dir = './data', 'labels.csv', 'train', 'test'
  input_dir, batch_size, valid_ratio = 'train_valid_test', 128, 0.1
  reorg_dog_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio)