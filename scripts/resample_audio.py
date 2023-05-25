"""
Script to resample audio files to specified sample rate.
"""

import argparse
import torchaudio
from pathlib import Path
from tqdm import tqdm


def resample_audio(input_path, output_path, sample_rate):
    """
    Resamples audio files to specified sample rate.
    """
    # Load audio file
    waveform, sr = torchaudio.load(input_path)
    # Resample audio file
    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=sample_rate)
    # Save audio file
    torchaudio.save(output_path, waveform, sample_rate)


def main():
    parser = argparse.ArgumentParser(description="Resample audio files to specified sample rate.")
    parser.add_argument("--filelist", type=str, required=True, help="Filelist to the paths to input audio file.")
    # parser.add_argument("--output_folder", type=str, required=True, help="Path to output audio file.")
    parser.add_argument("--audio_folder", type=str, default="Audio", help="Path to audio folder.")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate to resample audio file to.")
    args = parser.parse_args()

    with open(args.filelist, "r") as f:
        for line in tqdm(f.readlines()):
            # Get input and output paths
            line = line.strip()
            input_path = line
            output_path = line.replace(
                args.audio_folder, "{}_{}kHz".format(args.audio_folder, args.sample_rate // 1000)
            )
            # Create output path
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            # Resample audio file
            resample_audio(input_path, output_path, args.sample_rate)


if __name__ == "__main__":
    main()
