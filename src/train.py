import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
# No longer needed for single GPU:
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler
# import torch.distributed as dist

from model import CSITransformer
from utils import load_config, CSIDataset
from losses import MultiTaskCrossEntropyLoss

def train(config, args):
    """The main training function for single GPU."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n--- [PHASE 1: DATA LOADING & PREPROCESSING] ---")

    # -- Load data --
    dataset = CSIDataset(
        data_path=config['data']['path'],
        scenario=config['data']['scenario'],
        sequence_length=config['data']['sequence_length']
    )
    
    # Split dataset into training and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Dataset split complete: {train_size} training samples, {val_size} validation samples.")

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    print("DataLoaders created.")
    print("--- [PHASE 1 COMPLETE] ---")

    # -- Model --
    print("\n--- [PHASE 2: MODEL & OPTIMIZER INITIALIZATION] ---")
    print(f"Initializing model: {config['model']['name']}")

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
    ).to(device)
    
    print("Model initialized and moved to device.")

    # -- Loss and Optimizer --
    criterion = MultiTaskCrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    
    print("Loss function and optimizer created.")
    print("--- [PHASE 2 COMPLETE] ---")
    print("\n===== Initialization Complete. Starting Training Loop... =====\n")

    # -- Training Loop --
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        for csi_data, labels in train_pbar:
            csi_data, labels = csi_data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(csi_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # -- Validation --
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
            for csi_data, labels in val_pbar:
                csi_data, labels = csi_data.to(device), labels.to(device)
                outputs = model(csi_data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_pbar.set_postfix(val_loss=f"{loss.item():.4f}")
        
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{config['training']['epochs']}] Summary: Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")
        
        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            model_path = os.path.join(config['training']['model_dir'], f"{config['model']['name']}_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Saved model to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer model for WiFi Sensing.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    train(config, args) 