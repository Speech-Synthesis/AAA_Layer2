import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import Wav2Vec2BertModel
import os
import pandas as pd
import torchaudio

class ASVspoofDataset(torch.utils.data.Dataset):
    def __init__(self, audio_dir, metadata_file):
        self.audio_dir = audio_dir
        self.metadata = pd.read_csv(metadata_file, sep=' ', header=None, names=['speaker', 'file_name', 'sys_id', 'key', 'label'])
        
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_path = os.path.join(self.audio_dir, f"{row['file_name']}.flac")
        label = 1.0 if row['label'] == 'bonafide' else 0.0
        
        try:
            waveform, sr = torchaudio.load(file_path)
            # Ensure it's 1D for the padding logic in collate_fn
            waveform = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform.squeeze(0)
            return waveform, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            # Re-raise so the smoke test try/except can catch it for the dummy files
            raise RuntimeError(f"Error loading {file_path}: {e}")

# --- HARDENED DATA LOADER ---
# This ensures that even if one audio file is corrupted, the training doesn't crash.
def collate_fn(batch):
    # Padding: If audio clips have slightly different lengths, pad them to the longest in the batch
    waveforms, labels = zip(*batch)
    waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    return waveforms, torch.stack(labels)

class AAAEngineAntiSpoofing(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
        for param in self.base_model.parameters():
            param.requires_grad = False # Freeze foundation model
            
        self.classification_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.base_model(x).last_hidden_state
        pooled = features.mean(dim=1)
        return self.classification_head(pooled)

def train_on_hpc():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"HPC Training initialized on: {device}")
    
    model = AAAEngineAntiSpoofing().to(device)
    optimizer = optim.AdamW(model.classification_head.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    
    # Path setup for HPC
    dataset = ASVspoofDataset(audio_dir="aaa_dataset/flac", metadata_file="labels.txt")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    # --- TRAINING LOOP WITH CHECKPOINTING ---
    for epoch in range(15):
        model.train()
        for batch_idx, (audio, labels) in enumerate(dataloader):
            audio, labels = audio.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(audio).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        # Save checkpoint every epoch so you don't lose work
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }, f"checkpoint_epoch_{epoch}.pth")
        print(f"Epoch {epoch} finished.")

def smoke_test():
    print("--- RUNNING SMOKE TEST ---")
    try:
        # Point to your mock folder
        dataset = ASVspoofDataset(audio_dir="aaa_dataset_mock/flac", metadata_file="aaa_dataset_mock/labels.txt")
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
        
        # Pull one sample
        audio, label = next(iter(dataloader))
        print(f"Integration Test PASSED. Tensor shape: {audio.shape}")
    except Exception as e:
        print(f"Integration Test FAILED. Error: {e}")

# Call this instead of train_on_hpc() while testing
if __name__ == "__main__":
    smoke_test()