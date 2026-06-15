import os
import librosa
import soundfile as sf

def split_audio(input_path, output_dir, chunk_duration=10):
    # Load audio
    y, sr = librosa.load(input_path, sr=None)  # keep original sample rate

    # Calculate samples per chunk
    chunk_size = int(chunk_duration * sr)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Split and save chunks
    num_chunks = (len(y) + chunk_size - 1) // chunk_size  # ceiling division

    split_name = os.path.basename(input_path).split('.')
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = y[start:end]

        output_path = os.path.join(output_dir, f"{".".join(split_name[:-1])}_{i:04d}.{split_name[-1]}")
        sf.write(output_path, chunk, sr)

        print(f"Saved: {output_path}")

if __name__ == "__main__":
    with os.scandir("../songs/") as files:
        for file in files:
            if file.is_file():
                split_audio(file.path, "./", chunk_duration=10)
