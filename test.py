import torch
import numpy as np
import librosa
from model import FakeAudioDetector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load trained model ---
model = FakeAudioDetector().to(device)
model.load_state_dict(torch.load("fake_audio_detector.pth", map_location=device))
model.eval()  # important for inference

# --- Preprocess single audio ---
def preprocess_audio(file_path, n_mfcc=20, max_len=200):
    y, sr = librosa.load(file_path, sr=None, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    feat = np.vstack([mfcc, delta])  # shape (40, time)

    # pad or trim to max_len
    if feat.shape[1] < max_len:
        pad_width = max_len - feat.shape[1]
        feat = np.pad(feat, ((0,0),(0,pad_width)), mode='constant')
    else:
        feat = feat[:, :max_len]

    # convert to tensor
    feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)  # (1,40,max_len)
    return feat_tensor

# --- Predict single file ---
def predict_audio(file_path):
    feat = preprocess_audio(file_path)
    with torch.no_grad():
        output = model(feat)
        pred = torch.argmax(output, dim=1).item()
    label = "REAL" if pred == 0 else "FAKE"
    print(f"Prediction for {file_path}: {label}")
    return label

# --- Example usage ---
test_file = "fakesample.wav"   # change this to your audio
predict_audio(test_file)
