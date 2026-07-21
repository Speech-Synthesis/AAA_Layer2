import os
import torchaudio
from pathlib import Path
from tqdm import tqdm

def validate_flac_directory(directory_path):
    print(f"--- Starting Validation of FLAC files in {directory_path} ---")
    
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return

    flac_files = list(Path(directory_path).rglob("*.flac"))
    print(f"Found {len(flac_files)} .flac files. Checking for corruption...\n")
    
    corrupted_files = []
    
    # Try loading each file to see if it's healthy
    for file_path in tqdm(flac_files, desc="Validating"):
        try:
            # We only load the metadata to check if the file header is valid
            # This is much faster than loading the entire waveform
            _ = torchaudio.info(str(file_path))
        except Exception as e:
            corrupted_files.append((file_path, str(e)))
            
    print("\n--- Validation Complete ---")
    if not corrupted_files:
        print("Success! All .flac files are healthy and ready for training.")
    else:
        print(f"Warning: Found {len(corrupted_files)} corrupted files!")
        for file, error in corrupted_files:
            print(f"Corrupt: {file}\n  Error: {error}")
            
if __name__ == "__main__":
    # Update this path to your downloaded ASVspoof 2019 Dev set folder
    TARGET_DIR = "path/to/your/downloaded/ASVspoof2019_LA_dev/flac"
    
    # For now, let's test it on the mock directory we created earlier!
    # TARGET_DIR = "aaa_dataset_mock/flac"
    
    validate_flac_directory(TARGET_DIR)
