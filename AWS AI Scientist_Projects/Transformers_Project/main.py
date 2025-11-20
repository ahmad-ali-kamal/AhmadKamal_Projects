import torch
from load_data import train_df, test_df
from explore_data import explore_data
from dataset import IMDBDataset, MAX_LENGTH
from config import tokenizer, config
from model import DemoGPT
from train import train_model
from test_model import test_model
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

explore_data(train_df)

train_size = int(0.9 * len(train_df))
shuffled_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
train_data = shuffled_df.iloc[:train_size]
val_data = shuffled_df.iloc[train_size:]

train_dataset = IMDBDataset(train_data, tokenizer)
val_dataset = IMDBDataset(val_data, tokenizer)
test_dataset = IMDBDataset(test_df, tokenizer)

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = DemoGPT(config).to(device)

train_model(model, train_loader, val_loader, device, epochs=3)

test_model(model, test_loader, device)
