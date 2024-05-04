import os
import torch
import librosa
import numpy as np

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchaudio import load
import torch.optim as optim

from tqdm import tqdm
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_auc_score, roc_curve

from model import Model
import wandb

wandb.init(project="Speech Assignment 3", name = "task 4")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AudioDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.classes, self.class_to_idx = self._find_classes()
        self.samples = self._make_dataset()

    def _find_classes(self):
        classes = [d.name for d in os.scandir(self.root) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self):
        samples = []
        for target_class in self.classes:
            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(self.root, target_class)
            for root_dir, _, file_names in os.walk(target_dir):
                for file_name in file_names:
                    if file_name.endswith('.wav') or file_name.endswith('.mp3') or file_name.endswith('.ogg'):
                        file_path = os.path.join(root_dir, file_name)
                        samples.append((file_path, class_index))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, class_index = self.samples[idx]
        waveform, _ = load(audio_path)
        waveform = self._preprocess_audio(waveform)
        return waveform, class_index

    def _preprocess_audio(self, waveform):
        waveform = waveform.numpy()[0]
        max_len = 64600
        if waveform.shape[0] >= max_len:
            return waveform[:max_len]
        else:
            num_repeats = int(max_len / waveform.shape[0]) + 1
            padded_waveform = np.tile(waveform, (1, num_repeats))[:, :max_len][0]
            return padded_waveform

root = "for-2seconds"

train_dataset = AudioDataset(root=os.path.join(root, "training"))
test_dataset = AudioDataset(root=os.path.join(root, "testing"))
validation_dataset = AudioDataset(root=os.path.join(root, "validation"))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=6)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False, num_workers=6)

model = Model(None, device=device)
model = nn.DataParallel(model).to(device)
model.load_state_dict(torch.load('best_la_model.pth'))

def train_one_epoch(train_loader, neural_net, learning_rate, optimizer, device):
    total_loss = 0.0
    total_samples = 0.0
    num_batches = len(train_loader)

    neural_net.train()

    progress_bar = tqdm(train_loader, desc='Training', leave=False)

    for input_batch, target_batch in progress_bar:
        batch_size = input_batch.size(0)
        total_samples += batch_size

        input_batch = input_batch.to(device)
        target_batch = target_batch.view(-1).type(torch.int64).to(device)

        optimizer.zero_grad()

        output_batch = neural_net(input_batch)

        batch_loss = compute_loss(output_batch, target_batch)

        total_loss += batch_loss.item() * batch_size

        batch_loss.backward()
        optimizer.step()

        progress_bar.set_postfix(loss=total_loss / num_batches)

    progress_bar.close()

    total_loss /= len(train_loader.dataset)

    # Log training loss
    wandb.log({"train_loss": total_loss})

    return total_loss

def compute_loss(output, target):

    loss_func = torch.nn.CrossEntropyLoss()
    return loss_func(output, target)

def validate_model(validation_loader, neural_net, device):
    total_loss = 0.0
    total_samples = 0.0
    num_batches = len(validation_loader)

    neural_net.eval()

    progress_bar = tqdm(validation_loader, desc='Validation', leave=False)

    with torch.no_grad():
        for input_batch, target_batch in progress_bar:
            batch_size = input_batch.size(0)
            total_samples += batch_size

            input_batch = input_batch.to(device)
            target_batch = target_batch.view(-1).type(torch.int64).to(device)

            output_batch = neural_net(input_batch)

            batch_loss = compute_loss(output_batch, target_batch)

            total_loss += batch_loss.item() * batch_size

            progress_bar.set_postfix(loss=total_loss / num_batches)

    progress_bar.close()

    total_loss /= len(validation_loader.dataset)

    wandb.log({"val_loss": total_loss})

    return total_loss

lr = 5e-5

optimizer = optim.Adam(model.parameters(), lr=lr)

num_epochs = 5

for epoch in range(num_epochs):
    train_loss = train_one_epoch(train_loader, model, lr, optimizer, device)
    val_loss = validate_model(validation_loader, model, device)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

torch.save(model.state_dict(), "final_model.pth")

def evaluate_model(test_loader, model):
    true_labels = []
    predicted_scores = []

    model.eval()

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            output = model(data)
            predicted_scores.extend(output[:, 1].cpu().numpy())
            true_labels.extend(target.cpu().numpy())

    true_labels = np.array(true_labels)
    predicted_scores = np.array(predicted_scores)

    auc_score = roc_auc_score(true_labels, predicted_scores)

    fpr, tpr, thresholds = roc_curve(true_labels, predicted_scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    threshold = interp1d(fpr, thresholds)(eer)

    print("AUC:", auc_score)
    print("EER:", eer)
    print("Threshold at EER:", threshold)

    return auc_score, eer, threshold

evaluate_model(test_loader, model)
