import torch
import torch.nn as nn
import torch.nn.functional as F


class Reparameterize(nn.Module):
    def forward(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std


class StateVAE(nn.Module):
    def __init__(self, input_mode: str, latent_dim: int, beta: float, state_dim: int = None):
        super().__init__()
        self.input_mode = input_mode  # "image" or "state"
        self.latent_dim = latent_dim
        self.beta = beta

        if self.input_mode == "image":
            # Reuse LIBERO's ImageEncoder dynamically at runtime to avoid import cycles
            from libero.lifelong.models.modules.rgb_modules import ImageEncoder

            # 64 keypoints * 2 dims = 128-dim feature
            self.image_encoder = ImageEncoder(
                feature_dimension=128,
                backbone_type="ResNet18",
                pool_type="SpatialSoftmax",
                num_kp=64,
            )
            feat_dim = 128
        else:
            assert (
                state_dim is not None
            ), "state_dim is required when input_mode == 'state'"
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
            )
            feat_dim = 512

        self.fc_mean = nn.Linear(feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(feat_dim, latent_dim)
        self.reparam = Reparameterize()

        # Lightweight decoder: decode to feature space for reconstruction loss
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, feat_dim),
            nn.ReLU(),
        )

    def encode_features(self, obs: dict) -> torch.Tensor:
        if self.input_mode == "image":
            # Expect a dict of images; use first image stream by convention
            # Normalize to [-1, 1]
            # obs should contain preprocessed tensors shaped [B, C, H, W]
            # If multiple cams present, concatenate features
            if isinstance(obs, dict):
                feats = []
                for _, img in obs.items():
                    x = img * 2.0 - 1.0
                    feats.append(self.image_encoder(x))
                feat = torch.cat(feats, dim=-1)
            else:
                x = obs * 2.0 - 1.0
                feat = self.image_encoder(x)
            return feat
        else:
            return self.state_encoder(obs)

    def forward(self, obs):
        feat = self.encode_features(obs)
        mean = self.fc_mean(feat)
        logvar = self.fc_logvar(feat)
        z = self.reparam(mean, logvar)
        recon = self.decoder(z)
        return {
            "z": z,
            "mean": mean,
            "logvar": logvar,
            "feat": feat.detach(),
            "recon": recon,
        }

    def loss(self, out: dict, recon_weight: float = 1.0) -> torch.Tensor:
        # Feature-space reconstruction
        recon_loss = F.mse_loss(out["recon"], out["feat"], reduction="mean")
        # KL divergence term
        kl = -0.5 * torch.mean(1 + out["logvar"] - out["mean"].pow(2) - out["logvar"].exp())
        return recon_weight * recon_loss + self.beta * kl


class ActionVAE(nn.Module):
    def __init__(self, action_dim: int, latent_dim: int, beta: float):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = nn.Sequential(
            nn.Linear(action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        self.reparam = Reparameterize()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, actions: torch.Tensor):
        h = self.encoder(actions)
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        z = self.reparam(mean, logvar)
        recon = self.decoder(z)
        return {
            "z": z,
            "mean": mean,
            "logvar": logvar,
            "recon": recon,
        }

    def loss(self, out: dict) -> torch.Tensor:
        recon_loss = F.mse_loss(out["recon"], out["recon"].detach(), reduction="mean")
        # Note: action reconstruction target should be the input actions; pass externally if needed
        kl = -0.5 * torch.mean(1 + out["logvar"] - out["mean"].pow(2) - out["logvar"].exp())
        return recon_loss + self.beta * kl


class JointVAE(nn.Module):
    def __init__(
        self,
        state_input_mode: str,
        state_latent_dim: int,
        state_beta: float,
        action_dim: int,
        action_latent_dim: int,
        action_beta: float,
        state_dim: int = None,
    ):
        super().__init__()
        self.state_vae = StateVAE(
            input_mode=state_input_mode,
            latent_dim=state_latent_dim,
            beta=state_beta,
            state_dim=state_dim,
        )
        self.action_vae = ActionVAE(
            action_dim=action_dim,
            latent_dim=action_latent_dim,
            beta=action_beta,
        )

    @torch.no_grad()
    def encode(self, obs, actions):
        s = self.state_vae(obs)
        a = self.action_vae(actions)
        return s["z"], a["z"]


