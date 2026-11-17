#!/usr/bin/env python3
"""Train and INT8-Quantize M13: Supervised Contrastive Circular CNN-2D (SupCon).

Implements Passo 3 of the accuracy improvement roadmap:
  - Stage 1 (Contrastive Pre-training): Trains the `CompactCircularSemgCnn2D`
    backbone + an MLP projection head (`112 -> 128 -> 64`) using a Class-Balanced
    Supervised Contrastive Loss (`SupConLoss`, Khosla et al. 2020) over two
    augmented sEMG views per window. Weights anchor losses inversely by batch
    class frequency so REST (36% of NinaPro DB2) does not drown out the 49 active
    gesture classes.
  - Stage 2 (Prototype Initialization & Fine-tuning): Discards the projection head,
    computes L2-normalized class prototypes in the SupCon embedding space to
    warm-start `fc_conv`, initializes `fc_aux` with Canonical LDA, and fine-tunes
    end-to-end with label-smoothed Cross-Entropy.
  - Export / Deployment: Performs BatchNorm folding and exact symmetric INT8/Q24
    quantization into `QuantizedCnn2DCircularArtifacts` — resulting in ZERO extra
    parameters, ZERO extra SRAM, and ZERO extra inference cycles compared to M12!
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semg_baseline.lda import CanonicalLdaTrainer, PriorType  # noqa: E402
from semg_baseline.scaling import FeatureScaler  # noqa: E402
from src.models.train_cnn2d_circular import (  # noqa: E402
    CompactCircularSemgCnn2D,
    QuantizedCnn2DCircularArtifacts,
    evaluate_and_quantize_cnn2d_circular,
    set_deterministic_seed,
)


class SupConProjectionHead(nn.Module):
    """Training-only MLP projection head discarded prior to INT8 deployment."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(h), dim=1)


class SupConLoss(nn.Module):
    """Class-Balanced Supervised Contrastive Loss for imbalanced sEMG streams.

    Args:
        temperature: Softmax temperature scaling cosine similarities.
        balance_classes: If True, weight each anchor inversely by its frequency in
            the current mini-batch so majority classes (e.g., REST) do not dominate
            the representation geometry.
    """

    def __init__(self, temperature: float = 0.10, balance_classes: bool = True) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = float(temperature)
        self.balance_classes = balance_classes

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute Supervised Contrastive Loss.

        Args:
            features: L2-normalized tensor of shape (B, n_views, D).
            labels: Integer class labels of shape (B,).
        """
        if features.ndim != 3:
            raise ValueError(f"Expected 3D tensor (B, V, D), got shape {tuple(features.shape)}")

        device = features.device
        batch_size, n_views, _ = features.shape
        if batch_size <= 1:
            return torch.zeros((), device=device, dtype=features.dtype)

        labels_col = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels_col, labels_col.T).float().to(device)  # (B, B)

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # (B*V, D)
        anchor_feature = contrast_feature
        anchor_count = n_views

        # Cosine similarity scaled by temperature
        anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T) / self.temperature
        # Numerical stability via max subtraction
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile positive mask across views and remove self-contrasts
        mask = mask.repeat(anchor_count, n_views)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size * n_views, device=device)
        mask = mask * logits_mask

        # Compute log_prob over non-self samples
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Compute mean of log-likelihood over positives for anchors that have >= 1 positive
        mask_pos_pairs = mask.sum(1)
        valid_anchors = mask_pos_pairs > 0
        if not torch.any(valid_anchors):
            return torch.zeros((), device=device, dtype=features.dtype)

        mean_log_prob_pos = (mask[valid_anchors] * log_prob[valid_anchors]).sum(1) / mask_pos_pairs[valid_anchors]
        anchor_losses = -mean_log_prob_pos

        if self.balance_classes:
            # Inverse class frequency weighting per anchor within batch
            labels_rep = labels.repeat(anchor_count)[valid_anchors]
            uniq, inv_idx, counts = torch.unique(labels_rep, return_inverse=True, return_counts=True)
            weights = 1.0 / counts.float()[inv_idx]
            weights = weights / (weights.mean() + 1e-8)
            return (anchor_losses * weights).mean()

        return anchor_losses.mean()


def augment_semg_view(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Apply physiologically plausible sEMG augmentations to form a contrastive view.

    Transformations:
      1. Channel-wise amplitude jitter U(0.80, 1.20) for electrode impedance drift.
      2. Random subtle temporal shift within +/- 4 samples (+/- 2 ms @ 2kHz).
      3. Additive white Gaussian noise (sigma = 0.04).
    """
    bsz, ch, _ = x.shape
    amp_scale = 0.80 + 0.40 * torch.rand((bsz, ch, 1), generator=gen, device=x.device)
    out = x * amp_scale

    # Random small circular time shift for the batch chunk
    shift = int(torch.randint(-4, 5, (1,), generator=gen).item())
    if shift != 0:
        out = torch.roll(out, shifts=shift, dims=2)

    noise = 0.04 * torch.randn(out.shape, generator=gen, device=x.device)
    return out + noise


def train_and_quantize_supcon_cnn2d(
    dataset_name: str,
    subject_id: str,
    train_win: np.ndarray,
    train_lbl: np.ndarray,
    test_win: np.ndarray,
    test_lbl: np.ndarray,
    X_tr_aux: np.ndarray,
    X_te_aux: np.ndarray,
    num_classes: int,
    supcon_epochs: int = 12,
    finetune_epochs: int = 15,
    batch_size: int = 512,
    temperature: float = 0.10,
    seed: int = 42,
) -> tuple[QuantizedCnn2DCircularArtifacts, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train M13 (SupCon + Circular CNN-2D), discard projection head, and quantize to INT8."""
    set_deterministic_seed(seed)
    torch.set_num_threads(4)

    in_channels = int(train_win.shape[2])
    num_aux = int(X_tr_aux.shape[1])

    # Training-only input standardization
    in_mean = np.mean(train_win, axis=(0, 1), dtype=np.float64).astype(np.float32)
    in_std = np.std(train_win, axis=(0, 1), dtype=np.float64).astype(np.float32)
    in_std = np.where(in_std < 1e-8, 1.0, in_std).astype(np.float32)
    in_inv_std = (1.0 / in_std).astype(np.float32)

    scaler = FeatureScaler()
    X_tr_aux_s = scaler.fit_transform(X_tr_aux).astype(np.float32)
    X_te_aux_s = scaler.transform(X_te_aux).astype(np.float32)
    sdict = scaler.to_dict()
    aux_mean = np.asarray(sdict["mean"], dtype=np.float32)
    aux_scale = np.asarray(sdict["scale"], dtype=np.float32)
    aux_inv_std = np.where(aux_scale != 0.0, 1.0 / aux_scale, 0.0).astype(np.float32)
    for z_idx in sdict["zero_variance_indices"]:
        if 0 <= z_idx < num_aux:
            aux_inv_std[z_idx] = 0.0

    # Canonical LDA warm-start for auxiliary linear branch
    lda_trainer = CanonicalLdaTrainer(prior_type=PriorType.UNIFORM, num_classes=num_classes)
    lda_trainer.fit(X_tr_aux_s, train_lbl)
    lda_art = lda_trainer.get_model_artifacts()

    X_tr_cnn = np.transpose((train_win - in_mean[None, None, :]) * in_inv_std[None, None, :], (0, 2, 1)).astype(np.float32)
    X_te_cnn = np.transpose((test_win - in_mean[None, None, :]) * in_inv_std[None, None, :], (0, 2, 1)).astype(np.float32)

    t_x_tr = torch.from_numpy(X_tr_cnn)
    t_aux_tr = torch.from_numpy(X_tr_aux_s)
    t_y_tr = torch.from_numpy(train_lbl.astype(np.int64))

    model = CompactCircularSemgCnn2D(in_channels=in_channels, num_classes=num_classes, num_aux=num_aux, dropout_p=0.15)
    proj_head = SupConProjectionHead(in_dim=model.total_conv_feat_dim, hidden_dim=128, out_dim=64)

    gen = torch.Generator().manual_seed(seed)
    n_tr = t_x_tr.shape[0]

    # =========================================================================
    # STAGE 1: Supervised Contrastive Representation Pre-Training
    # =========================================================================
    if supcon_epochs > 0:
        backbone_params = [p for n, p in model.named_parameters() if "fc_" not in n]
        opt_sc = torch.optim.AdamW(
            backbone_params + list(proj_head.parameters()),
            lr=3.5e-3,
            weight_decay=1e-3,
        )
        sched_sc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_sc, T_max=supcon_epochs)
        supcon_crit = SupConLoss(temperature=temperature, balance_classes=True)

        model.train()
        proj_head.train()
        for ep in range(1, supcon_epochs + 1):
            perm = torch.randperm(n_tr, generator=gen)
            ep_loss = 0.0
            n_batches = 0
            for start in range(0, n_tr, batch_size):
                b_idx = perm[start : start + batch_size]
                if len(b_idx) < 4:
                    continue
                bx = t_x_tr[b_idx]
                by = t_y_tr[b_idx]

                v1 = augment_semg_view(bx, gen)
                v2 = augment_semg_view(bx, gen)

                h1 = model.forward_features(v1)["combined"]
                h2 = model.forward_features(v2)["combined"]
                z1 = proj_head(h1)
                z2 = proj_head(h2)
                feats = torch.stack([z1, z2], dim=1)  # (B, 2, 64)

                loss = supcon_crit(feats, by)
                opt_sc.zero_grad()
                loss.backward()
                opt_sc.step()

                ep_loss += float(loss.item())
                n_batches += 1
            sched_sc.step()
            if ep % 3 == 0 or ep == supcon_epochs:
                print(
                    f"    [{dataset_name}][Stage1-SupCon] Epoch {ep:02d}/{supcon_epochs} | loss={ep_loss / max(1, n_batches):.4f}",
                    flush=True,
                )

    # =========================================================================
    # STAGE 2: Prototype Initialization & End-to-End Fine-Tuning
    # =========================================================================
    model.eval()
    with torch.no_grad():
        # Compute L2-normalized class prototypes in SupCon embedding space
        emb_list = []
        for start in range(0, n_tr, 1024):
            emb_list.append(model.forward_features(t_x_tr[start : start + 1024])["combined"])
        all_emb = torch.cat(emb_list, dim=0)  # (N_tr, D)

        prototypes = torch.zeros((num_classes, model.total_conv_feat_dim), dtype=torch.float32)
        for cls_id in range(num_classes):
            cls_mask = t_y_tr == cls_id
            if torch.any(cls_mask):
                proto = all_emb[cls_mask].mean(dim=0)
                prototypes[cls_id] = F.normalize(proto, dim=0) * 0.25

        model.fc_conv.weight.copy_(prototypes)
        model.fc_conv.bias.copy_(torch.from_numpy(lda_art.intercept_.astype(np.float32)))
        model.fc_aux.weight.copy_(torch.from_numpy(lda_art.coef_.astype(np.float32)))

    opt_ft = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if "fc_" not in n], "lr": 1.2e-3, "weight_decay": 1e-3},
            {"params": model.fc_conv.parameters(), "lr": 2.5e-3, "weight_decay": 1e-2},
            {"params": model.fc_aux.parameters(), "lr": 3.0e-4, "weight_decay": 1e-2},
        ]
    )
    sched_ft = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ft, T_max=max(1, finetune_epochs))
    ce_crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    model.train()
    for ep in range(1, finetune_epochs + 1):
        perm = torch.randperm(n_tr, generator=gen)
        ep_loss = 0.0
        n_batches = 0
        for start in range(0, n_tr, batch_size):
            b_idx = perm[start : start + batch_size]
            bx = t_x_tr[b_idx]
            baux = t_aux_tr[b_idx]
            by = t_y_tr[b_idx]

            # Mild jitter during fine-tuning
            amp_scale = 0.90 + 0.20 * torch.rand((bx.shape[0], bx.shape[1], 1), generator=gen)
            noise = 0.02 * torch.randn(bx.shape, generator=gen)
            bx_aug = bx * amp_scale + noise

            opt_ft.zero_grad()
            logits = model(bx_aug, baux)
            loss = ce_crit(logits, by)
            loss.backward()
            opt_ft.step()

            ep_loss += float(loss.item())
            n_batches += 1
        sched_ft.step()
        if ep % 5 == 0 or ep == finetune_epochs:
            print(
                f"    [{dataset_name}][Stage2-FineTune] Epoch {ep:02d}/{finetune_epochs} | loss={ep_loss / max(1, n_batches):.4f}",
                flush=True,
            )

    return evaluate_and_quantize_cnn2d_circular(
        model=model,
        dataset_name=dataset_name,
        subject_id=subject_id,
        X_tr_cnn=X_tr_cnn,
        X_tr_aux_s=X_tr_aux_s,
        test_win=test_win,
        test_lbl=test_lbl,
        X_te_cnn=X_te_cnn,
        X_te_aux=X_te_aux,
        X_te_aux_s=X_te_aux_s,
        in_mean=in_mean,
        in_inv_std=in_inv_std,
        aux_mean=aux_mean,
        aux_inv_std=aux_inv_std,
    )
