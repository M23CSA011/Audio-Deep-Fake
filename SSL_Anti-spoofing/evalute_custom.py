
import os
import torch
import librosa
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from model import Model
import torch.nn as nn

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def preprocess_audio(audio_path):
    _, ext = os.path.splitext(audio_path)
    if ext.lower() not in ('.mp3', '.wav'):
        return None

    audio, sr = librosa.load(audio_path, sr=None)
    audio = pad(audio)
    return audio

real_audio_dir = r"Dataset_Speech_Assignment/Dataset_Speech_Assignment/Real"
fake_audio_dir = r"Dataset_Speech_Assignment/Dataset_Speech_Assignment/Fake"

real_audio_paths = [os.path.join(real_audio_dir, filename) for filename in os.listdir(real_audio_dir)]
fake_audio_paths = [os.path.join(fake_audio_dir, filename) for filename in os.listdir(fake_audio_dir)]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Model(None, device=device)
model = nn.DataParallel(model).to(device)
model.load_state_dict(torch.load('best_la_model.pth'))

model.eval()

predictions = []
ground_truth = []

for audio_path in tqdm(real_audio_paths + fake_audio_paths):
    try:
        processed_audio = preprocess_audio(audio_path)

        if processed_audio is None:
            continue

        processed_audio_tensor = torch.tensor(processed_audio, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = model(processed_audio_tensor)
            output_cpu = output.cpu()
            prediction = output_cpu[0][1].item()

        predictions.append(prediction)

        if audio_path in real_audio_paths:
            ground_truth.append(0)
        else:
            ground_truth.append(1)

    except Exception as e:
        continue

fpr, tpr, _ = roc_curve(ground_truth, predictions)
auc_score = roc_auc_score(ground_truth, predictions)

eer = 1.0
for i in range(len(fpr)):
    if fpr[i] >= 1 - tpr[i]:
        eer = fpr[i]
        break

print("EER:", eer)
print("AUC:", auc_score)
