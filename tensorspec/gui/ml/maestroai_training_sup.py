import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from PySide6.QtCore import QThread, Signal

# Import ONLY the Supervised model
from tensorspec.core.ml.models import SupervisedCNN

class SupTrainWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(object)

    def __init__(self, X, Y, num_classes):
        super().__init__()
        self.X, self.Y, self.num_classes = X, Y, num_classes

    def run(self):
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        model = SupervisedCNN(self.num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        X_ten = torch.tensor(self.X, dtype=torch.float32).unsqueeze(1) 
        X_ten = F.interpolate(X_ten, size=(128, 128), mode='bilinear', align_corners=False)
        
        orig_min, orig_max = X_ten.min(), X_ten.max()
        X_ten = (X_ten - orig_min) / (orig_max - orig_min + 1e-8)
        Y_ten = torch.tensor(self.Y, dtype=torch.long)

        loader = DataLoader(TensorDataset(X_ten, Y_ten), batch_size=min(16, len(self.X)), shuffle=True)

        epochs = 60
        model.train()
        for ep in range(epochs):
            for b_x, b_y in loader:
                b_x, b_y = b_x.to(device), b_y.to(device)
                optimizer.zero_grad()
                out = model(b_x)
                loss = criterion(out, b_y)
                loss.backward()
                optimizer.step()
            if ep % 5 == 0:
                self.progress.emit(int((ep/epochs)*100), f"Training Few-Shot... Epoch {ep}")

        model.eval()
        model.data_min = orig_min.item()
        model.data_max = orig_max.item()
        self.progress.emit(100, "Training Done")
        self.finished.emit(model.cpu())


class SupTestWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(np.ndarray)

    def __init__(self, model, val_array):
        super().__init__()
        self.model = model
        self.val_array = val_array

    def run(self):
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()

        dim_E, dim_A, nY, nX = self.val_array.shape
        flat_val = self.val_array.transpose(2, 3, 0, 1).reshape(nY*nX, 1, dim_E, dim_A)

        t_orig = torch.tensor(flat_val, dtype=torch.float32)
        t_orig = F.interpolate(t_orig, size=(128, 128), mode='bilinear', align_corners=False)
        t_orig = (t_orig - self.model.data_min) / (self.model.data_max - self.model.data_min + 1e-8)
        
        loader = DataLoader(TensorDataset(t_orig), batch_size=256, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for i, (b_x,) in enumerate(loader):
                b_x = b_x.to(device)
                logits = self.model(b_x)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
                if i % 10 == 0:
                    self.progress.emit(int((i/len(loader))*100), "Running Inference on Full Map...")

        prob_array = np.concatenate(all_probs, axis=0) 
        prob_map = prob_array.reshape((nY, nX, -1))
        
        self.model.cpu()
        self.progress.emit(100, "Done!")
        self.finished.emit(prob_map)