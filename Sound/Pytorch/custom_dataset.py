# use the URBANSOUND8K dataset as an example

from torch.util.data import Dataset
import torchaudio
import pandas as pd
import os

class UrbanSoundDataset(Dataset):

    def __init__(self, annotations_file, audio_dir): # initial constructor
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir

    def __len__(self): # how to use len()
        return len(self.annotations)

    def __getitem__(self, index): # how to use indexing, e.g. a_list[0] -> a_list.__getitem__(0)
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        return (signal, label)

    def _get_audio_sample_path(self, index): # complicated(more than 2 lines) -> put in a separate function
        fold = f"fold{self.annotations.iloc[index, 5]}" # use iloc to specify the row and column
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])
        return path

    def _get_label(self, index):
        return self.annotations.iloc[index, 6]

def main():
    ANNOTATION_FILE = "UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "UrbanSound8K/audio"
    usd = UrbanSoundDataset(ANNOTATION_FILE, AUDIO_DIR)
    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = usd[0]


if __name__ == "__main__":
    main()

