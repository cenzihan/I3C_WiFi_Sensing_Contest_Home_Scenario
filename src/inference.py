import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

from model import CSITransformer
from utils import load_config, CSIDataset
from torch.utils.data import DataLoader

def plot_confusion_matrix(cm, class_names, room_name, output_path):
    """Plots a confusion matrix and saves it to a file."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {room_name}')
    plt.savefig(output_path)
    plt.close()

def inference(config):
    """Runs inference on the test set and saves the results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # -- Load Data --
    # For inference, we can just use a portion of the dataset as a "test set"
    dataset = CSIDataset(
        data_path=config['data']['path'],
        scenario=config['data']['scenario'],
        sequence_length=config['data']['sequence_length']
    )
    # We'll just use the whole dataset as a test set for this example
    test_loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # -- Model --
    model_config = config['model']
    model = CSITransformer(
        input_features=model_config['input_features'],
        d_model=model_config['d_model'],
        n_heads=model_config['n_heads'],
        n_layers=model_config['n_layers'],
        dim_feedforward=model_config['dim_feedforward'],
        num_rooms=model_config['num_rooms'],
        num_classes_per_room=model_config['num_classes_per_room'],
        dropout=model_config['dropout']
    )
    
    model.load_state_dict(torch.load(config['inference']['model_path'], map_location=device))
    model.to(device)
    model.eval()

    all_preds = [[] for _ in range(model_config['num_rooms'])]
    all_labels = [[] for _ in range(model_config['num_rooms'])]

    with torch.no_grad():
        for csi_data, labels in test_loader:
            csi_data = csi_data.to(device)
            outputs = model(csi_data)
            
            for i in range(model_config['num_rooms']):
                preds = torch.argmax(outputs[i], dim=1).cpu().numpy()
                all_preds[i].extend(preds)
                all_labels[i].extend(labels[:, i].cpu().numpy())

    # -- Evaluate and Visualize --
    os.makedirs(config['inference']['visualization_dir'], exist_ok=True)
    room_names = ['Room_A', 'Room_B', 'Parlor']
    class_names = [str(i) for i in range(model_config['num_classes_per_room'])]

    for i in range(model_config['num_rooms']):
        accuracy = accuracy_score(all_labels[i], all_preds[i])
        cm = confusion_matrix(all_labels[i], all_preds[i])
        
        print(f"Accuracy for {room_names[i]}: {accuracy:.4f}")
        
        cm_path = os.path.join(config['inference']['visualization_dir'], f"cm_{room_names[i]}.png")
        plot_confusion_matrix(cm, class_names, room_names[i], cm_path)
        print(f"Saved confusion matrix for {room_names[i]} to {cm_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for WiFi Sensing model.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()
    
    config = load_config(args.config)
    inference(config) 