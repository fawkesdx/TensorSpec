import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

# =============================================================================
# NEURAL NETWORK MODELS
# =============================================================================

class SimpleCAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, latent_dim) 
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 8 * 8), nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)
        )
    def forward(self, x): return self.decoder(self.encoder(x))

class SimCLRModel(nn.Module):
    def __init__(self, base_dim=64, proj_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 8 * 8, base_dim) 
        )
        self.projector = nn.Sequential(nn.Linear(base_dim, base_dim), nn.ReLU(), nn.Linear(base_dim, proj_dim))
    def forward(self, x):
        h = self.encoder(x)
        return h, self.projector(h)

class ContrastiveDataset(Dataset):
    def __init__(self, data, transform):
        self.data, self.transform = data, transform
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.transform(self.data[idx]), self.transform(self.data[idx])

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temp, self.criterion = temperature, nn.CrossEntropyLoss()
    def forward(self, z_i, z_j):
        z_i, z_j = F.normalize(z_i, dim=1), F.normalize(z_j, dim=1)
        bs = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)
        sim_matrix = torch.matmul(z, z.T) / self.temp
        labels = torch.cat([torch.arange(bs, 2*bs), torch.arange(bs)], dim=0).to(z_i.device)
        sim_matrix.masked_fill_(torch.eye(2*bs, dtype=torch.bool).to(z_i.device), -9e15)
        return self.criterion(sim_matrix, labels)

class BetaVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(), nn.Flatten()
        )
    def initialize_linears(self, input_shape, latent_dim=32):
        batch_size = 1
        input_tensor = torch.autograd.Variable(torch.rand(batch_size, *input_shape))
        conv_out = self.encoder[:-1](input_tensor)
        self.unflatten_shape = conv_out.shape[1:] 
        self._to_linear = int(np.prod(self.encoder(input_tensor).size()))
        self.fc_mu, self.fc_logvar = nn.Linear(self._to_linear, latent_dim), nn.Linear(self._to_linear, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, self._to_linear)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        h = self.decoder_input(z).view(-1, *self.unflatten_shape)
        out = F.interpolate(self.decoder(h), size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        return out, mu, logvar

class CNNMaskedAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(), nn.Flatten()
        )
    def initialize_linears(self, input_shape, latent_dim=64):
        input_tensor = torch.autograd.Variable(torch.rand(1, *input_shape))
        self.unflatten_shape = self.encoder[:-1](input_tensor).shape[1:] 
        self._to_linear = int(np.prod(self.encoder(input_tensor).size()))
        self.fc_encode, self.fc_decode = nn.Linear(self._to_linear, latent_dim), nn.Linear(latent_dim, self._to_linear)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )
    def forward(self, x):
        latent = self.fc_encode(self.encoder(x))
        h_dec = self.fc_decode(latent).view(-1, *self.unflatten_shape) 
        out = F.interpolate(self.decoder(h_dec), size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        return out, latent

class SupervisedCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

class MoCoModel(nn.Module):
    def __init__(self, base_dim=64, proj_dim=32, K=1024, m=0.99):
        super().__init__()
        self.K = K
        self.m = m
        
        def create_encoder():
            return nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.Flatten(), nn.Linear(64 * 8 * 8, base_dim),
                nn.ReLU(), nn.Linear(base_dim, proj_dim)
            )
            
        self.encoder_q = create_encoder()
        self.encoder_k = create_encoder()
        
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False 
            
        self.register_buffer("queue", torch.randn(proj_dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

class BYOLModel(nn.Module):
    def __init__(self, base_dim=64, proj_dim=32):
        super().__init__()
        
        def create_net():
            return nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.Flatten(), nn.Linear(64 * 8 * 8, base_dim)
            )
            
        self.online_encoder = create_net()
        self.target_encoder = create_net()
        
        self.online_projector = nn.Sequential(nn.Linear(base_dim, base_dim), nn.ReLU(), nn.Linear(base_dim, proj_dim))
        self.target_projector = nn.Sequential(nn.Linear(base_dim, base_dim), nn.ReLU(), nn.Linear(base_dim, proj_dim))
        
        self.online_predictor = nn.Sequential(nn.Linear(proj_dim, base_dim), nn.ReLU(), nn.Linear(base_dim, proj_dim))
        
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False
        for param_o, param_t in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False

class ViTMAE(nn.Module):
    def __init__(self, img_size=64, patch_size=8, embed_dim=128, mask_ratio=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=256, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.dec_proj = nn.Linear(embed_dim, embed_dim // 2)
        decoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim // 2, nhead=4, dim_feedforward=128, activation='gelu', batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=2)

        self.pred = nn.Linear(embed_dim // 2, patch_size * patch_size) 

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2) 
        x = x + self.pos_embed

        N = self.num_patches
        len_keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, self.embed_dim))

        x_kept = self.encoder(x_kept)

        mask_tokens = self.mask_token.repeat(B, N - len_keep, 1)
        x_full = torch.cat([x_kept, mask_tokens], dim=1)
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, self.embed_dim))

        x_full = self.dec_proj(x_full)
        decoded = self.decoder(x_full)
        pred = self.pred(decoded) 

        return pred, x_kept.mean(dim=1)

class SwAVModel(nn.Module):
    def __init__(self, base_dim=64, proj_dim=32, num_prototypes=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 8 * 8, base_dim)
        )
        self.projector = nn.Sequential(
            nn.Linear(base_dim, base_dim), nn.ReLU(),
            nn.Linear(base_dim, proj_dim)
        )
        self.prototypes = nn.Linear(proj_dim, num_prototypes, bias=False)

    def forward(self, x):
        h = self.encoder(x)
        z = F.normalize(self.projector(h), dim=1)
        scores = self.prototypes(z) 
        return h, z, scores