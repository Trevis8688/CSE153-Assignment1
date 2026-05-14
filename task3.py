"""Task 3: Multi-label audio tagging.

Bigger CNN with BatchNorm + global average pooling, SpecAugment augmentation,
trained with BCEWithLogitsLoss on Apple MPS.
"""
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
from torch.utils.data import Dataset, DataLoader, random_split
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from sklearn.metrics import average_precision_score
from tqdm import tqdm

DATAROOT = "student_files/task3_audio_classification"
SAMPLE_RATE = 22050
N_MELS = 96
N_CLASSES = 10
AUDIO_DURATION = 10
BATCH_SIZE = 32
TAGS = ['rock', 'oldies', 'jazz', 'pop', 'dance',  'blues',  'punk', 'chill', 'electronic', 'country']


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def extract_waveform(path):
    waveform, sr = librosa.load(os.path.join(DATAROOT, path), sr=SAMPLE_RATE)
    waveform = torch.FloatTensor(np.array([waveform]))
    if sr != SAMPLE_RATE:
        resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resample(waveform)
    target_len = SAMPLE_RATE * AUDIO_DURATION
    if waveform.shape[1] < target_len:
        pad_len = target_len - waveform.shape[1]
        waveform = F.pad(waveform, (0, pad_len))
    else:
        waveform = waveform[:, :target_len]
    return waveform


class AudioDataset(Dataset):
    def __init__(self, meta, augment=False):
        self.meta = meta
        self.ids = list(meta.keys())
        self.augment = augment
        self.mel = MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=1024, hop_length=256, f_min=20, f_max=SAMPLE_RATE // 2)
        self.db = AmplitudeToDB(top_db=80)
        self.feats = {}
        for path in tqdm(self.ids, desc="Preloading mels"):
            w = extract_waveform(path)
            m = self.db(self.mel(w)).squeeze(0)
            # Normalize per-clip
            m = (m - m.mean()) / (m.std() + 1e-6)
            self.feats[path] = m

    def __len__(self):
        return len(self.ids)

    def _spec_augment(self, m):
        # m shape: (mel, time)
        if random.random() < 0.5:
            # Frequency mask
            f = random.randint(0, 16)
            f0 = random.randint(0, max(1, N_MELS - f))
            m[f0:f0+f, :] = 0
        if random.random() < 0.5:
            # Time mask
            t = random.randint(0, 40)
            t0 = random.randint(0, max(1, m.shape[1] - t))
            m[:, t0:t0+t] = 0
        return m

    def __getitem__(self, idx):
        path = self.ids[idx]
        tags = self.meta[path]
        bin_label = torch.tensor([1 if t in tags else 0 for t in TAGS], dtype=torch.float32)
        m = self.feats[path].clone()
        if self.augment:
            m = self._spec_augment(m)
        return m.unsqueeze(0), bin_label, path


class CNNClassifier(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        def block(ic, oc, pool=(2, 4)):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1, bias=False),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
                nn.Conv2d(oc, oc, 3, padding=1, bias=False),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool),
            )
        self.b1 = block(1, 32, pool=(2, 4))
        self.b2 = block(32, 64, pool=(2, 4))
        self.b3 = block(64, 128, pool=(2, 4))
        self.b4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)  # logits


def evaluate(model, loader, device, return_paths=False):
    model.eval()
    all_logits, all_targets, all_paths = [], [], []
    with torch.no_grad():
        for x, y, ps in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            all_logits.append(logits.cpu())
            all_targets.append(y.cpu())
            all_paths += list(ps)
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    probs = torch.sigmoid(logits).numpy()
    targets_np = targets.numpy()
    # mAP only meaningful when targets exist (i.e., not test set)
    mAP = None
    if targets_np.sum() > 0:
        try:
            mAP = average_precision_score(targets_np, probs, average='macro')
        except Exception:
            mAP = None
    if return_paths:
        return probs, all_paths, mAP
    return probs, mAP


def train_predict_save():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    device = get_device()
    print("Device:", device)

    train_meta = eval(open(os.path.join(DATAROOT, "train.json")).read())
    test_list = eval(open(os.path.join(DATAROOT, "test.json")).read())
    # Test labels are unknown; use empty placeholders
    test_meta = {k: [] for k in test_list}

    print(f"Train clips: {len(train_meta)}, Test clips: {len(test_meta)}")

    all_train = AudioDataset(train_meta, augment=False)
    # split into train/valid
    n = len(all_train)
    n_train = int(n * 0.9)
    n_valid = n - n_train
    g = torch.Generator().manual_seed(0)
    train_set, valid_set = random_split(all_train, [n_train, n_valid], generator=g)

    # Build augment view by sharing the underlying tensors but switching the augment flag on a copy
    # Simplest: subclass to toggle augment per-batch; we'll just enable augment globally and turn off during val
    # We achieve that by giving train_set/valid_set a custom collate that knows the split type
    train_aug = AudSubset(train_set, augment=True)
    valid_clean = AudSubset(valid_set, augment=False)

    test_set = AudioDataset(test_meta, augment=False)

    loader_train = DataLoader(train_aug, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    loader_valid = DataLoader(valid_clean, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    loader_test = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = CNNClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    epochs = 25
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_map = -1.0
    best_state = None
    for ep in range(epochs):
        model.train()
        running = 0.0
        nb = 0
        for x, y, _ in tqdm(loader_train, desc=f"Epoch {ep+1}/{epochs}"):
            x = x.to(device); y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item()
            nb += 1
        scheduler.step()
        _, val_map = evaluate(model, loader_valid, device)
        print(f"[Epoch {ep+1}] loss={running/nb:.4f} val_mAP={val_map:.4f}")
        if val_map is not None and val_map > best_map:
            best_map = val_map
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"Best val mAP: {best_map:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)

    # Predict on test
    probs, paths, _ = evaluate(model, loader_test, device, return_paths=True)

    predictions = {}
    for i, p in enumerate(paths):
        # autograder expects normalized paths (drop leading "./")
        key = p[2:] if p.startswith("./") else p
        predictions[key] = {TAGS[j]: float(probs[i][j]) for j in range(N_CLASSES)}

    with open("predictions3.json", "w") as f:
        f.write(repr(predictions) + "\n")
    print(f"Wrote predictions3.json ({len(predictions)} entries)")


class AudSubset(Dataset):
    """Wrap a Subset and apply per-item augment flag."""
    def __init__(self, subset, augment):
        self.subset = subset
        self.augment = augment
        # The underlying AudioDataset already has a _spec_augment method; we
        # call it manually here, because Subset always uses the dataset's own __getitem__
        self.dataset = subset.dataset
        self.indices = subset.indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        path = self.dataset.ids[real_idx]
        tags = self.dataset.meta[path]
        bin_label = torch.tensor([1 if t in tags else 0 for t in TAGS], dtype=torch.float32)
        m = self.dataset.feats[path].clone()
        if self.augment:
            m = self.dataset._spec_augment(m)
        return m.unsqueeze(0), bin_label, path


if __name__ == "__main__":
    train_predict_save()
