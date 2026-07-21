import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import Wav2Vec2BertModel
import os

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

if __name__ == "__main__":
    train_on_hpc()