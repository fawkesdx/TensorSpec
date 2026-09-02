import numpy as np
from PySide6.QtCore import QThread, Signal

class ActiveLearningWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, list)
    error = Signal(str)

    def __init__(self, x_arr, y_arr, labels_2d, algo):
        super().__init__()
        self.x_arr, self.y_arr = x_arr, y_arr
        self.labels_2d = labels_2d
        self.algo = algo 

    def run(self):
        try:
            self.progress.emit(10, "Prepping coordinates...")
            X_grid, Y_grid = np.meshgrid(self.x_arr, self.y_arr)
            coords = np.column_stack((X_grid.flatten(), Y_grid.flatten()))
            labels = self.labels_2d.flatten()

            valid_mask = labels != -1
            coords_valid = coords[valid_mask]
            labels_valid = labels[valid_mask]

            x_margin = (self.x_arr.max() - self.x_arr.min()) * 0.2
            y_margin = (self.y_arr.max() - self.y_arr.min()) * 0.2
            new_x = np.linspace(self.x_arr.min() - x_margin, self.x_arr.max() + x_margin, int(len(self.x_arr)*1.4))
            new_y = np.linspace(self.y_arr.min() - y_margin, self.y_arr.max() + y_margin, int(len(self.y_arr)*1.4))
            New_X_grid, New_Y_grid = np.meshgrid(new_x, new_y)
            new_coords = np.column_stack((New_X_grid.flatten(), New_Y_grid.flatten()))
            bounds = [self.x_arr.min(), self.x_arr.max(), self.y_arr.min(), self.y_arr.max()]

            unique_labels = np.unique(labels_valid)
            num_classes = len(unique_labels)
            label_map = {old: new for new, old in enumerate(unique_labels)}
            mapped_labels = np.array([label_map[l] for l in labels_valid])
            reverse_map = {new: old for new, old in enumerate(unique_labels)}

            if "CPU" in self.algo:
                from scipy.stats import entropy
                import warnings
                from sklearn.exceptions import ConvergenceWarning
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                
                x_c, x_s = coords_valid[:, 0].mean(), coords_valid[:, 0].std() + 1e-8
                y_c, y_s = coords_valid[:, 1].mean(), coords_valid[:, 1].std() + 1e-8
                
                norm_coords_valid = np.zeros_like(coords_valid, dtype=float)
                norm_coords_valid[:, 0] = (coords_valid[:, 0] - x_c) / x_s
                norm_coords_valid[:, 1] = (coords_valid[:, 1] - y_c) / y_s
                
                norm_new_coords = np.zeros_like(new_coords, dtype=float)
                norm_new_coords[:, 0] = (new_coords[:, 0] - x_c) / x_s
                norm_new_coords[:, 1] = (new_coords[:, 1] - y_c) / y_s

                if "Gaussian Process" in self.algo:
                    max_pts = 600
                    if len(norm_coords_valid) > max_pts:
                        idx = np.random.choice(len(norm_coords_valid), max_pts, replace=False)
                        X_train, y_train = norm_coords_valid[idx], mapped_labels[idx]
                    else:
                        X_train, y_train = norm_coords_valid, mapped_labels
                        
                    from sklearn.gaussian_process import GaussianProcessClassifier
                    from sklearn.gaussian_process.kernels import RBF
                    self.progress.emit(30, "Training Gaussian Process (Locked to 1 Core for macOS Safety)...")
                    model = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0), random_state=42, n_jobs=1)
                    
                else:
                    X_train, y_train = norm_coords_valid, mapped_labels
                    from sklearn.ensemble import RandomForestClassifier
                    self.progress.emit(30, "Training Random Forest (Using All CPU Cores)...")
                    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
                
                model.fit(X_train, y_train)

                self.progress.emit(60, "Predicting Grid & Uncertainties...")
                chunk_size = 2000
                probs = np.zeros((len(norm_new_coords), num_classes))
                pred_idx = np.zeros(len(norm_new_coords), dtype=int)
                
                for i in range(0, len(norm_new_coords), chunk_size):
                    end_idx = min(i + chunk_size, len(norm_new_coords))
                    probs[i:end_idx] = model.predict_proba(norm_new_coords[i:end_idx])
                    pred_idx[i:end_idx] = model.predict(norm_new_coords[i:end_idx])
                    self.progress.emit(60 + int(35 * (end_idx / len(norm_new_coords))), "Predicting...")
                
                uncert = entropy(probs.T)

            elif "GPU" in self.algo:
                import torch
                import torch.nn as nn
                import torch.optim as optim
                import torch.nn.functional as F
                from scipy.stats import entropy

                device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
                
                x_c, x_s = coords_valid[:, 0].mean(), coords_valid[:, 0].std() + 1e-8
                y_c, y_s = coords_valid[:, 1].mean(), coords_valid[:, 1].std() + 1e-8
                norm_coords = np.zeros_like(coords_valid)
                norm_coords[:, 0], norm_coords[:, 1] = (coords_valid[:, 0] - x_c) / x_s, (coords_valid[:, 1] - y_c) / y_s
                X_ten = torch.tensor(norm_coords, dtype=torch.float32).to(device)
                Y_ten = torch.tensor(mapped_labels, dtype=torch.long).to(device)

                norm_new = np.zeros_like(new_coords)
                norm_new[:, 0], norm_new[:, 1] = (new_coords[:, 0] - x_c) / x_s, (new_coords[:, 1] - y_c) / y_s
                New_X_ten = torch.tensor(norm_new, dtype=torch.float32).to(device)

                class SimpleNet(nn.Module):
                    def __init__(self, use_dropout=False):
                        super().__init__()
                        self.net = nn.Sequential(
                            nn.Linear(2, 64), nn.ReLU(), nn.Dropout(0.2 if use_dropout else 0.0),
                            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.2 if use_dropout else 0.0),
                            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2 if use_dropout else 0.0),
                            nn.Linear(64, num_classes)
                        )
                    def forward(self, x): return self.net(x)

                if "Bayesian Network" in self.algo:
                    model = SimpleNet(use_dropout=True).to(device)
                    optimizer, criterion = optim.Adam(model.parameters(), lr=0.01), nn.CrossEntropyLoss()
                    self.progress.emit(30, "Training Bayesian Network (GPU)...")
                    model.train()
                    for epoch in range(300):
                        optimizer.zero_grad()
                        loss = criterion(model(X_ten), Y_ten)
                        loss.backward(); optimizer.step()
                        if epoch % 50 == 0: self.progress.emit(30 + int(30 * (epoch/300)), "Training...")

                    self.progress.emit(60, "Monte Carlo Sampling...")
                    mc_samples = 50
                    all_probs = torch.zeros(mc_samples, len(new_coords), num_classes).to(device)
                    with torch.no_grad():
                        for i in range(mc_samples):
                            all_probs[i] = F.softmax(model(New_X_ten), dim=1)
                            if i % 10 == 0: self.progress.emit(60 + int(35 * (i/mc_samples)), "Sampling...")
                    mean_probs = all_probs.mean(dim=0).cpu().numpy()
                    uncert = entropy(mean_probs.T)
                    pred_idx = np.argmax(mean_probs, axis=1)

                elif "Deep Ensembles" in self.algo:
                    num_models = 5
                    models = [SimpleNet(use_dropout=False).to(device) for _ in range(num_models)]
                    self.progress.emit(30, f"Training {num_models} Independent Networks (GPU)...")
                    
                    for m_idx, model in enumerate(models):
                        optimizer, criterion = optim.Adam(model.parameters(), lr=0.01), nn.CrossEntropyLoss()
                        model.train()
                        for epoch in range(250):
                            optimizer.zero_grad()
                            loss = criterion(model(X_ten), Y_ten)
                            loss.backward(); optimizer.step()
                        self.progress.emit(30 + int(30 * ((m_idx+1)/num_models)), f"Trained Model {m_idx+1}/{num_models}...")

                    self.progress.emit(60, "Ensemble Voting...")
                    all_probs = torch.zeros(num_models, len(new_coords), num_classes).to(device)
                    with torch.no_grad():
                        for m_idx, model in enumerate(models):
                            model.eval()
                            all_probs[m_idx] = F.softmax(model(New_X_ten), dim=1)
                    mean_probs = all_probs.mean(dim=0).cpu().numpy()
                    uncert = entropy(mean_probs.T)
                    pred_idx = np.argmax(mean_probs, axis=1)

                elif "Evidential" in self.algo:
                    model = SimpleNet(use_dropout=False).to(device)
                    optimizer = optim.Adam(model.parameters(), lr=0.01)
                    self.progress.emit(30, "Training Evidential Deep Learning Model (GPU)...")
                    
                    y_onehot = F.one_hot(Y_ten, num_classes=num_classes).float()
                    model.train()
                    for epoch in range(300):
                        optimizer.zero_grad()
                        evidence = F.softplus(model(X_ten))
                        alpha = evidence + 1
                        S = torch.sum(alpha, dim=1, keepdim=True)
                        p = alpha / S
                        loss = torch.mean(torch.sum((y_onehot - p)**2, dim=1) + torch.sum(p*(1-p)/(S+1), dim=1))
                        loss.backward(); optimizer.step()
                        if epoch % 50 == 0: self.progress.emit(30 + int(30 * (epoch/300)), "Training...")

                    self.progress.emit(60, "Calculating Dirichlet Uncertainty...")
                    model.eval()
                    with torch.no_grad():
                        evidence = F.softplus(model(New_X_ten))
                        alpha = evidence + 1
                        S = torch.sum(alpha, dim=1, keepdim=True)
                        uncert = (num_classes / S).squeeze().cpu().numpy()
                        pred_idx = torch.argmax(alpha, dim=1).cpu().numpy()

            pred_flat = np.array([reverse_map[idx] for idx in pred_idx])
            uncert_map = uncert.reshape(New_X_grid.shape)
            pred_map = pred_flat.reshape(New_X_grid.shape)

            self.progress.emit(100, "Done!")
            self.finished.emit(pred_map, uncert_map, new_x, new_y, bounds)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class SimulateALWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(np.ndarray, np.ndarray, int)
    error = Signal(str)

    def __init__(self, x_arr, y_arr, labels_2d, algo, measured_mask):
        super().__init__()
        self.x_arr, self.y_arr = x_arr, y_arr
        self.labels_2d = labels_2d
        self.algo = algo
        self.measured_mask = measured_mask 

    def run(self):
        try:
            self.progress.emit(10, "Prepping simulation grid...")
            X_grid, Y_grid = np.meshgrid(self.x_arr, self.y_arr)
            coords = np.column_stack((X_grid.flatten(), Y_grid.flatten()))
            labels = self.labels_2d.flatten()

            valid_mask = labels != -1
            coords_valid = coords[valid_mask]
            labels_valid = labels[valid_mask]
            measured_valid = self.measured_mask[valid_mask]

            X_train = coords_valid[measured_valid]
            y_train = labels_valid[measured_valid]

            unique_train_labels = np.unique(y_train)
            num_classes_total = len(np.unique(labels_valid))

            if len(unique_train_labels) < 2:
                self.progress.emit(100, "Need more classes to train. Picking random point...")
                unmeasured_idx = np.where(~measured_valid)[0]
                next_valid_idx = np.random.choice(unmeasured_idx)
                next_idx = np.where(valid_mask)[0][next_valid_idx]
                self.finished.emit(np.zeros(X_grid.shape), np.zeros(X_grid.shape), int(next_idx))
                return

            x_c, x_s = coords_valid[:, 0].mean(), coords_valid[:, 0].std() + 1e-8
            y_c, y_s = coords_valid[:, 1].mean(), coords_valid[:, 1].std() + 1e-8
            
            norm_valid = np.zeros_like(coords_valid, dtype=float)
            norm_valid[:, 0] = (coords_valid[:, 0] - x_c) / x_s
            norm_valid[:, 1] = (coords_valid[:, 1] - y_c) / y_s
            
            X_train_norm = norm_valid[measured_valid]

            unique_labels = np.unique(labels_valid)
            label_map = {old: new for new, old in enumerate(unique_labels)}
            y_train_mapped = np.array([label_map[l] for l in y_train])
            reverse_map = {new: old for new, old in enumerate(unique_labels)}

            self.progress.emit(30, f"Training {self.algo}...")

            if "CPU" in self.algo:
                from scipy.stats import entropy
                import warnings
                from sklearn.exceptions import ConvergenceWarning
                warnings.filterwarnings("ignore", category=ConvergenceWarning)

                if "Gaussian Process" in self.algo:
                    from sklearn.gaussian_process import GaussianProcessClassifier
                    from sklearn.gaussian_process.kernels import RBF
                    model = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0), random_state=42, n_jobs=1)
                else:
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
                
                model.fit(X_train_norm, y_train_mapped)
                
                self.progress.emit(70, "Predicting Full Grid...")
                probs = model.predict_proba(norm_valid)
                pred_idx = model.predict(norm_valid)
                
                full_probs = np.zeros((len(norm_valid), num_classes_total))
                for i, c in enumerate(np.unique(y_train_mapped)):
                    full_probs[:, c] = probs[:, i]
                uncert = entropy(full_probs.T)

            elif "GPU" in self.algo:
                import torch
                import torch.nn as nn
                import torch.optim as optim
                import torch.nn.functional as F
                from scipy.stats import entropy

                device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
                X_ten = torch.tensor(X_train_norm, dtype=torch.float32).to(device)
                Y_ten = torch.tensor(y_train_mapped, dtype=torch.long).to(device)
                New_X_ten = torch.tensor(norm_valid, dtype=torch.float32).to(device)

                class SimpleNet(nn.Module):
                    def __init__(self, use_dropout=False):
                        super().__init__()
                        self.net = nn.Sequential(
                            nn.Linear(2, 64), nn.ReLU(), nn.Dropout(0.2 if use_dropout else 0.0),
                            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.2 if use_dropout else 0.0),
                            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2 if use_dropout else 0.0),
                            nn.Linear(64, num_classes_total)
                        )
                    def forward(self, x): return self.net(x)

                if "Bayesian" in self.algo:
                    model = SimpleNet(use_dropout=True).to(device)
                    opt, crit = optim.Adam(model.parameters(), lr=0.01), nn.CrossEntropyLoss()
                    model.train()
                    for _ in range(300):
                        opt.zero_grad(); crit(model(X_ten), Y_ten).backward(); opt.step()

                    self.progress.emit(70, "Monte Carlo Sampling...")
                    all_probs = torch.zeros(50, len(norm_valid), num_classes_total).to(device)
                    with torch.no_grad():
                        for i in range(50): all_probs[i] = F.softmax(model(New_X_ten), dim=1)
                    mean_probs = all_probs.mean(dim=0).cpu().numpy()
                    uncert = entropy(mean_probs.T)
                    pred_idx = np.argmax(mean_probs, axis=1)

                elif "Evidential" in self.algo:
                    model = SimpleNet(use_dropout=False).to(device)
                    opt = optim.Adam(model.parameters(), lr=0.01)
                    y_onehot = F.one_hot(Y_ten, num_classes=num_classes_total).float()
                    model.train()
                    for _ in range(300):
                        opt.zero_grad()
                        ev = F.softplus(model(X_ten))
                        alpha = ev + 1
                        S = torch.sum(alpha, dim=1, keepdim=True)
                        p = alpha / S
                        loss = torch.mean(torch.sum((y_onehot - p)**2, dim=1) + torch.sum(p*(1-p)/(S+1), dim=1))
                        loss.backward(); opt.step()

                    self.progress.emit(70, "Calculating Dirichlet Uncertainty...")
                    model.eval()
                    with torch.no_grad():
                        ev = F.softplus(model(New_X_ten))
                        alpha = ev + 1
                        S = torch.sum(alpha, dim=1, keepdim=True)
                        uncert = (num_classes_total / S).squeeze().cpu().numpy()
                        pred_idx = torch.argmax(alpha, dim=1).cpu().numpy()

                elif "Deep Ensembles" in self.algo:
                    models = [SimpleNet(use_dropout=False).to(device) for _ in range(5)]
                    for m in models:
                        opt, crit = optim.Adam(m.parameters(), lr=0.01), nn.CrossEntropyLoss()
                        m.train()
                        for _ in range(200):
                            opt.zero_grad(); crit(m(X_ten), Y_ten).backward(); opt.step()
                    
                    self.progress.emit(70, "Ensemble Voting...")
                    all_probs = torch.zeros(5, len(norm_valid), num_classes_total).to(device)
                    with torch.no_grad():
                        for idx, m in enumerate(models):
                            m.eval(); all_probs[idx] = F.softmax(m(New_X_ten), dim=1)
                    mean_probs = all_probs.mean(dim=0).cpu().numpy()
                    uncert = entropy(mean_probs.T)
                    pred_idx = np.argmax(mean_probs, axis=1)

            pred_flat = np.array([reverse_map[idx] for idx in pred_idx])
            
            uncert_map_flat = np.zeros(len(coords), dtype=float)
            uncert_map_flat[valid_mask] = uncert
            
            pred_map_flat = np.full(len(coords), -1, dtype=int)
            pred_map_flat[valid_mask] = pred_flat

            unmeasured_valid = ~measured_valid
            if not np.any(unmeasured_valid):
                next_idx = -1
            else:
                uncert_unmeasured = uncert[unmeasured_valid]
                local_max_idx = np.argmax(uncert_unmeasured)
                global_valid_idx = np.where(unmeasured_valid)[0][local_max_idx]
                next_idx = np.where(valid_mask)[0][global_valid_idx]

            self.progress.emit(100, "Done!")
            self.finished.emit(pred_map_flat.reshape(X_grid.shape), uncert_map_flat.reshape(X_grid.shape), int(next_idx))

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))