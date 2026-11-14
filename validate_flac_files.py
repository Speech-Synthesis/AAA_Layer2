import os
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm


def validate_flac_directory(directory_path):
    print(f"--- Starting Validation of FLAC files in {directory_path} ---")

    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return

    flac_files = sorted(
        path for path in Path(directory_path).rglob("*")
        if path.is_file() and path.suffix.lower() == ".flac"
    )
    print(f"Found {len(flac_files)} .flac files. Checking metadata and decoding...\n")

    metadata_failures = []
    decoding_failures = []
    valid_files = []

    for file_path in tqdm(flac_files, desc="Validating"):
        metadata_error = None
        try:
            info = sf.info(str(file_path))
        except Exception as exc:
            metadata_error = exc
            metadata_failures.append((file_path, str(exc)))

        try:
            audio, sample_rate = sf.read(str(file_path), dtype="float32", always_2d=False)

            if audio.size == 0:
                raise ValueError("audio is empty")
            if audio.ndim == 0:
                raise ValueError("audio is scalar, not a valid waveform")
            if sample_rate <= 0:
                raise ValueError(f"invalid sample rate: {sample_rate}")
            if not np.isfinite(audio).all():
                raise ValueError("audio contains NaN or Inf values")

            channels = 1 if audio.ndim == 1 else audio.shape[1]
            duration = len(audio) / sample_rate

            if metadata_error is None:
                valid_files.append((file_path, sample_rate, channels, duration))
                print(
                    f"OK: {file_path} | sample rate={sample_rate} | "
                    f"channels={channels} | duration={duration:.3f}s"
                )
        except Exception as exc:
            decoding_failures.append((file_path, str(exc)))

    print("\n--- Validation Complete ---")

    if not flac_files:
        print("No FLAC files were found in the target directory.")
        return

    if valid_files:
        print(f"Validated {len(valid_files)} files successfully.")

    if metadata_failures:
        print(f"\nMetadata failures: {len(metadata_failures)}")
        for file_path, error in metadata_failures:
            print(f"Metadata failure: {file_path}\n  Error: {error}")

    if decoding_failures:
        print(f"\nDecoding failures: {len(decoding_failures)}")
        for file_path, error in decoding_failures:
            print(f"Decoding failure: {file_path}\n  Error: {error}")

    if not metadata_failures and not decoding_failures:
        print("Success! All FLAC files passed metadata and decoding checks.")
    else:
        print(
            f"Summary: {len(valid_files)} valid, {len(metadata_failures)} metadata failures, "
            f"{len(decoding_failures)} decoding failures."
        )


if __name__ == "__main__":
    TARGET_DIR = "aaa_dataset_mock/flac"
    validate_flac_directory(TARGET_DIR)
