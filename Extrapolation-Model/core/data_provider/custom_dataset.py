import os
import numpy as np
from PIL import Image

class InputHandle:
    """Handles loading and batching data for training and testing."""
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.current_position = 0

    def begin(self, do_shuffle=True):
        """Shuffle data at the start of each epoch."""
        if do_shuffle:
            np.random.shuffle(self.data)
        self.current_position = 0

    def get_batch(self):
        """Return a batch of sequential frames."""
        if self.current_position + self.batch_size > len(self.data):
            self.current_position = 0  # Reset if batch exceeds data size
        batch = self.data[self.current_position:self.current_position + self.batch_size]
        self.current_position += self.batch_size
        return np.array(batch)

    def no_batch_left(self):
        """Check if all batches have been processed."""
        return self.current_position >= len(self.data)

class DataProcess:
    """Processes disaster dataset for training and testing PredRNN++."""
    def __init__(self, input_param):
        print(f"Initializing DataProcess with params: {input_param}")

        self.paths = input_param.get('train_data_paths', input_param.get('valid_data_paths'))
        self.batch_size = input_param['batch_size']
        self.img_width = input_param['image_width']
        self.seq_length = input_param['seq_length']
        self.data = self.load_data()

        print(f"✅ Loaded {len(self.data)} sequences from {self.paths}")

    def load_data(self):
        """Loads images as NumPy arrays and ensures sequences of seq_length."""
        if not os.path.exists(self.paths):
            raise FileNotFoundError(f"Dataset path not found: {self.paths}")

        seq_folders = sorted(os.listdir(self.paths))  # Get list of sequence folders
        if len(seq_folders) == 0:
            raise ValueError(f"No sequences found in dataset path: {self.paths}")

        data = []
        for seq_folder in seq_folders:
            seq_path = os.path.join(self.paths, seq_folder)
            frames = sorted(os.listdir(seq_path))  # List images in order
            if len(frames) < self.seq_length:
                print(f"⚠️ Skipping {seq_folder}: Not enough frames ({len(frames)})")
                continue  # Skip sequences that are too short

            sequence = []
            for i in range(self.seq_length):
                img_path = os.path.join(seq_path, frames[i])
                img = Image.open(img_path).resize((self.img_width, self.img_width))
                img = np.array(img)

                if len(img.shape) == 2:  # Grayscale image
                    img = np.expand_dims(img, axis=-1)  # Convert to (H, W, 1)

                img = np.transpose(img, (0, 1, 2))  # Convert to (C, H, W)
                sequence.append(img)

            data.append(sequence)

        data = np.array(data)  # Convert to NumPy array
        print(f"✅ Final dataset shape: {data.shape}")  # Debug print
        return data  # Shape should now be (num_sequences, seq_length, C, H, W)

    def get_train_input_handle(self):
        print("Returning train input handle...")  # Debug print
        return InputHandle(self.data, self.batch_size)

    def get_test_input_handle(self):
        print("Returning test input handle...")  # Debug print
        return InputHandle(self.data, self.batch_size)
