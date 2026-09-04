import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as T

from PySide6.QtCore import QThread, Signal

# Import ONLY the SSL models
from .maestroai_models import (SimpleCAE, SimCLRModel, ContrastiveDataset, NTXentLoss, 
                              BetaVAE, CNNMaskedAutoencoder, MoCoModel, BYOLModel, 
                              ViTMAE, SwAVModel)

class TrainWorker(QThread):
    progress = Signal(int, float) 
    model_changed = Signal(str)
    finished = Signal(dict) 
    
    def __init__(self, data_array, epochs, lr, selected_models):
        super().__init__()
        self.data_array = data_array
        self.epochs = epochs
        self.lr = lr
        self.selected_models = selected_models

    def run(self):
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        dim_E, dim_A, nY, nX = self.data_array.shape
        n_samples = nY * nX
        
        flat_data = self.data_array.transpose(2, 3, 0, 1).reshape(n_samples, dim_A, dim_E)
        t_64 = torch.tensor(flat_data, dtype=torch.float32).unsqueeze(1)
        t_64 = F.interpolate(t_64, size=(64, 64), mode='bilinear', align_corners=False)
        means, stds = t_64.mean(dim=(2,3), keepdim=True), t_64.std(dim=(2,3), keepdim=True) + 1e-6
        t_64 = (t_64 - means) / stds

        flat_orig = self.data_array.transpose(2, 3, 0, 1).reshape(n_samples, 1, dim_A, dim_E)
        orig_min, orig_max = np.min(flat_orig), np.max(flat_orig)
        t_orig = torch.tensor((flat_orig - orig_min) / (orig_max - orig_min + 1e-8), dtype=torch.float32)

        results = {}

        for model_name in self.selected_models:
            self.model_changed.emit(model_name)
            
            if model_name == "Autoencoder":
                loader = DataLoader(TensorDataset(t_64), batch_size=64, shuffle=True)
                model = SimpleCAE(latent_dim=32).to(device)
                optimizer, criterion = torch.optim.Adam(model.parameters(), lr=self.lr), nn.MSELoss()
                for epoch in range(self.epochs):
                    total_loss = 0
                    model.train()
                    for batch in loader:
                        imgs = batch[0].to(device)
                        optimizer.zero_grad()
                        outputs = model(imgs)
                        loss = criterion(outputs, imgs)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                    self.progress.emit(epoch + 1, total_loss / len(loader))
                model.eval()
                embeds = [model.encoder(b[0].to(device)).cpu().detach().numpy() for b in DataLoader(TensorDataset(t_64), batch_size=128)]
                results['embeddings_autoencoder'] = np.concatenate(embeds, axis=0)

            elif model_name == "SimCLR":
                simclr_transform = T.Compose([
                    T.RandomResizedCrop(size=(64, 64), scale=(0.8, 1.0), antialias=True),
                    T.RandomHorizontalFlip(p=0.5), T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                ])
                loader = DataLoader(
                    ContrastiveDataset(t_64, simclr_transform), 
                    batch_size=min(64, len(t_64)), 
                    shuffle=True
                )
                
                model = SimCLRModel(base_dim=64, proj_dim=32).to(device)
                optimizer, criterion = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4), NTXentLoss(0.5)
                for epoch in range(self.epochs):
                    total_loss = 0
                    model.train()
                    for x_i, x_j in loader:
                        x_i, x_j = x_i.to(device), x_j.to(device)
                        optimizer.zero_grad()
                        _, z_i = model(x_i)
                        _, z_j = model(x_j)
                        loss = criterion(z_i, z_j)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                    self.progress.emit(epoch + 1, total_loss / max(1, len(loader)))
                model.eval()
                embeds = [model(b[0].to(device))[0].cpu().detach().numpy() for b in DataLoader(TensorDataset(t_64), batch_size=128)]
                results['embeddings_SimCLR'] = np.concatenate(embeds, axis=0)

            elif model_name == "Beta-VAE":
                loader = DataLoader(TensorDataset(t_orig), batch_size=64, shuffle=True)
                model = BetaVAE(latent_dim=32)
                model.initialize_linears(input_shape=(1, dim_A, dim_E)); model.to(device)
                optimizer, beta = torch.optim.Adam(model.parameters(), lr=self.lr), 4.0
                for epoch in range(self.epochs):
                    total_loss = 0
                    model.train()
                    for batch in loader:
                        imgs = batch[0].to(device)
                        optimizer.zero_grad()
                        recon, mu, logvar = model(imgs)
                        loss = nn.functional.mse_loss(recon, imgs, reduction='sum') - 0.5 * beta * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                    self.progress.emit(epoch + 1, total_loss / len(loader))
                model.eval()
                embeds = [model.encode(b[0].to(device))[0].cpu().detach().numpy() for b in DataLoader(TensorDataset(t_orig), batch_size=64)]
                results['embeddings_betavae'] = np.concatenate(embeds, axis=0)

            elif model_name == "MAE":
                batch_size = 256
                t_orig_device = t_orig.to(device) 
                loader = DataLoader(TensorDataset(t_orig_device), batch_size=batch_size, shuffle=True)
                
                model = CNNMaskedAutoencoder(latent_dim=64)
                model.initialize_linears(input_shape=(1, dim_A, dim_E)); model.to(device)
                
                scaled_lr = self.lr * (batch_size / 64)
                optimizer, criterion = torch.optim.Adam(model.parameters(), lr=scaled_lr), nn.MSELoss()
                
                for epoch in range(self.epochs):
                    total_loss = 0
                    model.train()
                    for batch in loader:
                        imgs = batch[0]
                        mask = (torch.rand(imgs.shape, device=device) > 0.75).float()
                        masked_imgs = imgs * mask
                        
                        optimizer.zero_grad()
                        recon, _ = model(masked_imgs)
                        loss = criterion(recon, imgs)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                    self.progress.emit(epoch + 1, total_loss / len(loader))
                
                model.eval()
                embeds = [model(b[0])[1].cpu().detach().numpy() for b in DataLoader(TensorDataset(t_orig_device), batch_size=512)]
                results['embeddings_mae'] = np.concatenate(embeds, axis=0)

            elif model_name == "MoCo":
                moco_transform = T.Compose([
                    T.RandomResizedCrop(size=(64, 64), scale=(0.8, 1.0), antialias=True),
                    T.RandomHorizontalFlip(p=0.5), T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                ])
                loader = DataLoader(ContrastiveDataset(t_64, moco_transform), batch_size=min(64, len(t_64)), shuffle=True)
                
                model = MoCoModel(base_dim=64, proj_dim=32, K=1024, m=0.99).to(device)
                optimizer = torch.optim.Adam(model.encoder_q.parameters(), lr=self.lr)
                criterion = nn.CrossEntropyLoss()
                
                for epoch in range(self.epochs):
                    total_loss = 0
                    model.train()
                    for x_q, x_k in loader:
                        x_q, x_k = x_q.to(device), x_k.to(device)
                        
                        q = model.encoder_q(x_q)
                        q = F.normalize(q, dim=1)
                        
                        with torch.no_grad():
                            model._momentum_update_key_encoder() 
                            k = model.encoder_k(x_k)
                            k = F.normalize(k, dim=1)
                            
                        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
                        l_neg = torch.einsum('nc,ck->nk', [q, model.queue.clone().detach()])
                        
                        logits = torch.cat([l_pos, l_neg], dim=1)
                        logits /= 0.07 
                        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
                        
                        loss = criterion(logits, labels)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        batch_size = q.shape[0]
                        ptr = int(model.queue_ptr)
                        if ptr + batch_size <= model.K:
                            model.queue[:, ptr:ptr + batch_size] = k.T
                            model.queue_ptr[0] = (ptr + batch_size) % model.K
                            
                        total_loss += loss.item()
                    self.progress.emit(epoch + 1, total_loss / max(1, len(loader)))
                    
                model.eval()
                embeds = [model.encoder_q[:-2](b[0].to(device)).cpu().detach().numpy() for b in DataLoader(TensorDataset(t_64), batch_size=128)]
                results['embeddings_moco'] = np.concatenate(embeds, axis=0)

            elif model_name == "BYOL":
                byol_transform = T.Compose([
                    T.RandomResizedCrop(size=(64, 64), scale=(0.8, 1.0), antialias=True),
                    T.RandomHorizontalFlip(p=0.5), T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                ])
                loader = DataLoader(ContrastiveDataset(t_64, byol_transform), batch_size=min(64, len(t_64)), shuffle=True)
                model = BYOLModel(base_dim=64, proj_dim=32).to(device)
                
                optimizer = torch.optim.Adam(list(model.online_encoder.parameters()) + 
                                             list(model.online_projector.parameters()) + 
                                             list(model.online_predictor.parameters()), lr=self.lr)
                
                def byol_loss_fn(p, z):
                    p = F.normalize(p, dim=1)
                    z = F.normalize(z, dim=1)
                    return 2 - 2 * (p * z).sum(dim=-1).mean()
                
                m = 0.99
                for epoch in range(self.epochs):
                    total_loss = 0
                    model.train()
                    for x1, x2 in loader:
                        x1, x2 = x1.to(device), x2.to(device)
                        
                        p1 = model.online_predictor(model.online_projector(model.online_encoder(x1)))
                        p2 = model.online_predictor(model.online_projector(model.online_encoder(x2)))
                        
                        with torch.no_grad():
                            for param_o, param_t in zip(model.online_encoder.parameters(), model.target_encoder.parameters()):
                                param_t.data = param_t.data * m + param_o.data * (1. - m)
                            for param_o, param_t in zip(model.online_projector.parameters(), model.target_projector.parameters()):
                                param_t.data = param_t.data * m + param_o.data * (1. - m)
                                
                            z1 = model.target_projector(model.target_encoder(x1))
                            z2 = model.target_projector(model.target_encoder(x2))
                            
                        loss = byol_loss_fn(p1, z2.detach()) + byol_loss_fn(p2, z1.detach())
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                    self.progress.emit(epoch + 1, total_loss / max(1, len(loader)))
                    
                model.eval()
                embeds = [model.online_encoder(b[0].to(device)).cpu().detach().numpy() for b in DataLoader(TensorDataset(t_64), batch_size=128)]
                results['embeddings_byol'] = np.concatenate(embeds, axis=0)

            elif model_name == "ViT-MAE":
                loader = DataLoader(TensorDataset(t_64), batch_size=128, shuffle=True)
                model = ViTMAE(img_size=64, patch_size=8, embed_dim=128, mask_ratio=0.75).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=self.lr * 2) 
                criterion = nn.MSELoss()

                for epoch in range(self.epochs):
                    total_loss = 0
                    model.train()
                    for batch in loader:
                        imgs = batch[0].to(device)
                        B = imgs.shape[0]
                        target = imgs.reshape(B, 1, 8, 8, 8, 8).permute(0, 2, 4, 3, 5, 1).reshape(B, 64, 64)

                        optimizer.zero_grad()
                        pred, _ = model(imgs)

                        loss = criterion(pred, target)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                    self.progress.emit(epoch + 1, total_loss / len(loader))

                model.eval()
                embeds = []
                with torch.no_grad():
                    for batch in DataLoader(TensorDataset(t_64), batch_size=256):
                        _, latent = model(batch[0].to(device))
                        embeds.append(latent.cpu().numpy())
                results['embeddings_vit_mae'] = np.concatenate(embeds, axis=0)
            
            elif model_name == "SwAV":
                swav_transform = T.Compose([
                    T.RandomResizedCrop(size=(64, 64), scale=(0.8, 1.0), antialias=True),
                    T.RandomHorizontalFlip(p=0.5), T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                ])
                loader = DataLoader(ContrastiveDataset(t_64, swav_transform), batch_size=min(64, len(t_64)), shuffle=True)
                model = SwAVModel(base_dim=64, proj_dim=32, num_prototypes=64).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                
                @torch.no_grad()
                def sinkhorn(scores, eps=0.05, iters=3):
                    Q = torch.exp(scores / eps).t()
                    B, K = Q.shape[1], Q.shape[0]
                    Q /= torch.sum(Q)
                    for _ in range(iters):
                        Q /= torch.sum(Q, dim=1, keepdim=True); Q /= K
                        Q /= torch.sum(Q, dim=0, keepdim=True); Q /= B
                    return (Q * B).t()

                tau = 0.1 
                for epoch in range(self.epochs):
                    total_loss = 0
                    model.train()
                    with torch.no_grad():
                        w = model.prototypes.weight.data.clone()
                        w = F.normalize(w, dim=1, p=2)
                        model.prototypes.weight.copy_(w)

                    for x1, x2 in loader:
                        x1, x2 = x1.to(device), x2.to(device)
                        
                        _, _, scores1 = model(x1)
                        _, _, scores2 = model(x2)
                        
                        q1 = sinkhorn(scores1)
                        q2 = sinkhorn(scores2)
                        
                        p1 = F.log_softmax(scores1 / tau, dim=1)
                        p2 = F.log_softmax(scores2 / tau, dim=1)
                        
                        loss = - (torch.sum(q1 * p2, dim=1).mean() + torch.sum(q2 * p1, dim=1).mean()) / 2.0
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                    self.progress.emit(epoch + 1, total_loss / max(1, len(loader)))
                    
                model.eval()
                embeds = [model(b[0].to(device))[0].cpu().detach().numpy() for b in DataLoader(TensorDataset(t_64), batch_size=128)]
                results['embeddings_swav'] = np.concatenate(embeds, axis=0)

        self.finished.emit(results)