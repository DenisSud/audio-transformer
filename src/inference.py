import torch
import librosa
import argparse
from utils.model import ClipTransformer, AudioEmbedder
import yaml
import os

def load_config():
    with open("config.yml", "r") as f:
        return yaml.safe_load(f)

def predict(audio_path, model_path, config):
    # Initialize model and load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClipTransformer(num_classes=config["model"]["num_classes"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load and process audio
    audio, _ = librosa.load(audio_path, sr=config["audio"]["sr"])
    embedder = AudioEmbedder()
    embeddings = embedder.get_embeddings(audio)

    # Make prediction
    with torch.no_grad():
        output = model(embeddings.unsqueeze(0))
        prediction = torch.argmax(output, dim=1)

    return prediction.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--model", required=True, help="Path to model weights")
    args = parser.parse_args()

    config = load_config()
    prediction = predict(args.audio, args.model, config)
    print(f"Predicted class: {prediction}")

if __name__ == "__main__":
    main()
