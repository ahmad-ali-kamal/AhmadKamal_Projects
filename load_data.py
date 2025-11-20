import os
import pandas as pd

train_pos_path = 'aclIMDB/train/pos'
train_neg_path = 'aclIMDB/train/neg'
test_pos_path = 'aclIMDB/test/pos'
test_neg_path = 'aclIMDB/test/neg'

def load_dataset(folder):
    """
    Reads all text files in the specified folder and returns their content as a list.
    Args:
        folder (str): Path to the folder containing text files.
    Returns:
        list: A list of strings, where each string is the content of a text file.
    """
    data = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) and filename.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                data.append(f.read())
    return data

train_pos = load_dataset(train_pos_path)
train_neg = load_dataset(train_neg_path)
test_pos = load_dataset(test_pos_path)
test_neg = load_dataset(test_neg_path)

train_df = pd.DataFrame({
    'review': train_pos + train_neg,
    'label': [1] * len(train_pos) + [0] * len(train_neg)
})

test_df = pd.DataFrame({
    'review': test_pos + test_neg,
    'label': [1] * len(test_pos) + [0] * len(test_neg)
})

assert train_df.shape[0] == 25000, "Training dataset does not have 25000 rows."
assert test_df.shape[0] == 25000, "Testing dataset does not have 25000 rows."
assert train_df.shape[1] == 2, "Training dataset does not have exactly 2 columns."
assert test_df.shape[1] == 2, "Testing dataset does not have exactly 2 columns."
