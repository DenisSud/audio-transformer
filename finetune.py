import torch
import argparse
import yaml
from utils.model import ClipTransformer
from utils.data import AudioClipDataset
from torch.utils.data import DataLoader
import os

def load_config():
    with open("config.yml", "r") as f:
        return yaml.safe_load(f)

def finetune(model_path, train_data, valid_data, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained model
    model = ClipTransformer(num_classes=config["model"]["num_classes"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Prepare datasets
    train_dataset = AudioClipDataset(train_data["files"], train_data["labels"])
    valid_dataset = AudioClipDataset(valid_data["files"], valid_data["labels"])

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config["training"]["batch_size"])

    # Fine-tuning setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"] * 0.1)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(config["training"]["epochs"]):
        model.train()
        # ... (similar to train.py but with fewer epochs and lower learning rate)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to pre-trained model")
    parser.add_argument("--train_data", required=True, help="Path to training data")
    parser.add_argument("--valid_data", required=True, help="Path to validation data")
    args = parser.parse_args()

    config = load_config()
    finetune(args.model, args.train_data, args.valid_data, config)

if __name__ == "__main__":
    main()
