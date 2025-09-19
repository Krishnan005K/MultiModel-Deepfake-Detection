import librosa, torch, numpy as np
from torch.utils.data import Dataset

class RealFakeAudioDataset(Dataset):
    def __init__(self, real_dir, fake_dir, sr=16000, n_mfcc=20, max_len=200):
        self.files, self.labels = [], []
        self.sr, self.n_mfcc, self.max_len = sr, n_mfcc, max_len

        for f in librosa.util.find_files(real_dir, ext=["wav"]):
            self.files.append(f)
            self.labels.append(0)
        for f in librosa.util.find_files(fake_dir, ext=["wav"]):
            self.files.append(f)
            self.labels.append(1)

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        y, sr = librosa.load(self.files[idx], sr=self.sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        delta = librosa.feature.delta(mfcc)
        feat = np.vstack([mfcc, delta])  # (40, time)

        # Pad/Trim to max_len frames
        if feat.shape[1] < self.max_len:
            feat = np.pad(feat, ((0,0),(0, self.max_len - feat.shape[1])))
        else:
            feat = feat[:, :self.max_len]

        return torch.tensor(feat, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)
