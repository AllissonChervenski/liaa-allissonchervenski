"""Enhanced Multi-Domain Neuromuscular Feature Extractor (v3) for High-Class sEMG.

Expands the baseline 5-feature Hudgins TD representation to a comprehensive
multi-domain neuromuscular vector combining:
  Per-Channel Features (42 features / channel):
    1. Nonlinear & Log-Amplitude TD (8): MAV, LOG_RMS, WL, MFL (Max Fractal Length),
       DASDV, WAMP, FORM_FACTOR (RMS / MAV), CREST_FACTOR (log(Peak / RMS)).
    2. Higher-Order Statistics & Morphological TD (6): ZC, SSC, HJORTH_MOB,
       HJORTH_COMP, SKEW, LOG_KURT.
    3. 4th-Order Auto-Regressive Poles (4): AR1..AR4 via regularized Yule-Walker.
    4. Multi-Band Spectral PSD, Moments & sEMG Cepstral Coeffs (11):
       - Log power in 5 sEMG bands ([20-60], [60-120], [120-200], [200-320], [320-500] Hz)
       - Normalized Mean Frequency (MNF) and Spectral Entropy
       - 4 sEMG Cepstral Coefficients (EMGCC1..4 = DCT of the 5 log-band energies)
    5. 4-Level Haar Wavelet Log-Energies (5): cD1, cD2, cD3, cD4, cA4.
    6. Sub-window Temporal Envelope Dynamics (8): Log-MAV (4) and Log-WL (4) across
       four contiguous 50 ms sub-windows (0-50, 50-100, 100-150, 150-200 ms).

  Cross-Channel Muscle Synergy & Spatial Co-Activation Matrix:
    - All C*(C-1)/2 upper-triangle pairwise Pearson cross-correlations between
      electrodes (66 for 12CH, 6 for 4CH, 1 for 2CH).
    - All C*(C-1)/2 upper-triangle pairwise log-RMS energy differences between
      electrodes (66 for 12CH, 6 for 4CH, 1 for 2CH), invariant to global force level.

Total feature dimension:
  - 12CH (NinaPro DB2): 12 * 42 + 66 + 66 = 636 features
  - 4CH (Ozdemir):      4 * 42 + 6 + 6    = 180 features
  - 2CH (Minimal):      2 * 42 + 1 + 1    = 86 features
"""

from __future__ import annotations

from typing import Sequence
import numpy as np


def _build_dct_basis(num_bands: int = 5, num_ceps: int = 4) -> np.ndarray:
    """Build Type-II DCT projection matrix of shape (num_bands, num_ceps)."""
    k = np.arange(1, num_ceps + 1, dtype=np.float64)[None, :]  # (1, num_ceps)
    n = np.arange(num_bands, dtype=np.float64)[:, None]  # (num_bands, 1)
    basis = np.cos(np.pi / float(num_bands) * (n + 0.5) * k) * np.sqrt(2.0 / float(num_bands))
    return basis  # (5, 4)


class EnhancedSemgFeatureExtractor:
    """Deterministic multi-domain sEMG feature extractor for 50-class TinyML decoding."""

    PER_CHANNEL_DIM = 42

    def __init__(
        self,
        epsilon: np.ndarray | Sequence[float],
        num_channels: int,
        fs: float = 2000.0,
    ) -> None:
        eps_arr = np.asarray(epsilon, dtype=np.float64)
        if eps_arr.shape != (num_channels,):
            raise ValueError(f"Expected epsilon shape ({num_channels},), got {eps_arr.shape}")
        self.epsilon = eps_arr
        self.num_channels = int(num_channels)
        self.fs = float(fs)

        # Pairwise spatial synergy dimensions
        self.pair_dim = (self.num_channels * (self.num_channels - 1)) // 2
        self._triu_r, self._triu_c = np.triu_indices(self.num_channels, k=1)
        self.spatial_dim = 2 * self.pair_dim  # correlations + pairwise log-RMS ratios
        self.total_dim = (self.PER_CHANNEL_DIM * self.num_channels) + self.spatial_dim
        self._dct_basis = _build_dct_basis(num_bands=5, num_ceps=4)

    @staticmethod
    def _solve_ar4_batch(windows: np.ndarray) -> np.ndarray:
        """Compute 4th-order AR coefficients via vectorized regularized Yule-Walker."""
        n, t_len, c = windows.shape
        x = windows.astype(np.float64) - np.mean(windows, axis=1, keepdims=True, dtype=np.float64)

        r = np.empty((n, c, 5), dtype=np.float64)
        r[:, :, 0] = np.sum(x * x, axis=1) + 1e-10
        for lag in range(1, 5):
            r[:, :, lag] = np.sum(x[:, lag:, :] * x[:, : t_len - lag, :], axis=1)

        rho = r[:, :, 1:5] / r[:, :, 0:1]
        r_mat = np.zeros((n, c, 4, 4), dtype=np.float64)
        for i in range(4):
            for j in range(4):
                diff = abs(i - j)
                r_mat[:, :, i, j] = 1.001 if diff == 0 else rho[:, :, diff - 1]

        ar = np.linalg.solve(r_mat, rho[..., None])[..., 0]
        return np.nan_to_num(ar, nan=0.0, posinf=0.0, neginf=0.0)

    def extract(self, windows: np.ndarray, chunk_size: int = 2048) -> np.ndarray:
        """Extract multi-domain features from sEMG windows of shape (N, T, C)."""
        if windows.ndim != 3:
            raise ValueError(f"Expected 3D windows (N, T, C), got {windows.shape}")
        n_total, t_len, c = windows.shape
        if c != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {c}")
        if n_total == 0:
            return np.empty((0, self.total_dim), dtype=np.float32)

        out = np.empty((n_total, self.total_dim), dtype=np.float32)
        freqs = np.fft.rfftfreq(t_len, d=1.0 / self.fs)
        hann = np.hanning(t_len)[None, :, None]

        # 5 sEMG spectral bands
        m_b1 = (freqs >= 20.0) & (freqs < 60.0)
        m_b2 = (freqs >= 60.0) & (freqs < 120.0)
        m_b3 = (freqs >= 120.0) & (freqs < 200.0)
        m_b4 = (freqs >= 200.0) & (freqs < 320.0)
        m_b5 = (freqs >= 320.0) & (freqs <= 500.0)
        m_valid = (freqs >= 20.0) & (freqs <= 500.0)
        valid_freqs = freqs[m_valid]

        eps_b = self.epsilon[None, None, :]

        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            w = windows[start:end].astype(np.float64)
            n_b = len(w)

            # 1. Nonlinear & Log-Amplitude TD (8)
            abs_w = np.abs(w)
            mav = np.mean(abs_w, axis=1)
            rms = np.sqrt(np.mean(w * w, axis=1) + 1e-12)
            log_rms = np.log(rms + 1e-6)
            form_factor = rms / (mav + 1e-6)
            crest_factor = np.log(np.max(abs_w, axis=1) / (rms + 1e-12) + 1e-6)

            dx = w[:, 1:, :] - w[:, :-1, :]
            abs_dx = np.abs(dx)
            wl = np.sum(abs_dx, axis=1)
            mfl = np.log(wl + 1e-6)
            dasdv = np.sqrt(np.mean(dx * dx, axis=1) + 1e-12)
            wamp = np.sum(abs_dx >= eps_b, axis=1).astype(np.float64)

            # 2. Morphological, Hjorth & Higher-Order Stats (6)
            zc_cond = (w[:, :-1, :] * w[:, 1:, :] < 0.0) & (abs_dx >= eps_b)
            zc = np.sum(zc_cond, axis=1).astype(np.float64)

            d1 = w[:, 1:-1, :] - w[:, :-2, :]
            d2 = w[:, 1:-1, :] - w[:, 2:, :]
            ssc_cond = (d1 * d2 > 0.0) & ((np.abs(d1) >= eps_b) | (np.abs(d2) >= eps_b))
            ssc = np.sum(ssc_cond, axis=1).astype(np.float64)

            w_centered = w - np.mean(w, axis=1, keepdims=True)
            std_w = np.sqrt(np.mean(w_centered * w_centered, axis=1) + 1e-12)
            z_w = w_centered / std_w[:, None, :]
            skew = np.mean(z_w ** 3, axis=1)
            kurt_log = np.log(np.mean(z_w ** 4, axis=1) + 1.0)

            var0 = (std_w ** 2) + 1e-12
            var1 = np.var(dx, axis=1) + 1e-12
            ddx = dx[:, 1:, :] - dx[:, :-1, :]
            var2 = np.var(ddx, axis=1) + 1e-12
            mob = np.sqrt(var1 / var0)
            comp = np.sqrt(var2 / var1) / (mob + 1e-8)

            # 3. AR-4 Coefficients (4)
            ar4 = self._solve_ar4_batch(w)

            # 4. 5-Band Spectral PSD, Moments & EMGCC Cepstrum (11)
            spec = np.fft.rfft(w_centered * hann, axis=1)
            psd = (spec.real ** 2 + spec.imag ** 2) + 1e-12

            bp1 = np.log(np.sum(psd[:, m_b1, :], axis=1) + 1e-8)
            bp2 = np.log(np.sum(psd[:, m_b2, :], axis=1) + 1e-8)
            bp3 = np.log(np.sum(psd[:, m_b3, :], axis=1) + 1e-8)
            bp4 = np.log(np.sum(psd[:, m_b4, :], axis=1) + 1e-8)
            bp5 = np.log(np.sum(psd[:, m_b5, :], axis=1) + 1e-8)

            log_bands = np.stack([bp1, bp2, bp3, bp4, bp5], axis=2)  # (B, C, 5)
            emgcc = log_bands @ self._dct_basis  # (B, C, 4)

            psd_v = psd[:, m_valid, :]
            tot_v = np.sum(psd_v, axis=1, keepdims=True) + 1e-12
            p_norm = psd_v / tot_v
            mnf = np.sum(p_norm * valid_freqs[None, :, None], axis=1) / 500.0
            spec_ent = -np.sum(p_norm * np.log(p_norm + 1e-12), axis=1) / np.log(float(len(valid_freqs)))

            # 5. 4-Level Haar Wavelet Log-Energies (5)
            cur = w_centered
            haar_list = []
            for _ in range(4):
                even_len = (cur.shape[1] // 2) * 2
                even = cur[:, 0:even_len:2, :]
                odd = cur[:, 1:even_len:2, :]
                detail = (even - odd) * 0.7071067811865475
                approx = (even + odd) * 0.7071067811865475
                haar_list.append(np.log(np.mean(detail * detail, axis=1) + 1e-8))
                cur = approx
            haar_list.append(np.log(np.mean(cur * cur, axis=1) + 1e-8))

            # 6. Sub-window Temporal Envelope: 4x Log-MAV + 4x Log-WL (8)
            q_len = t_len // 4
            sub_mav = np.empty((n_b, c, 4), dtype=np.float64)
            sub_wl = np.empty((n_b, c, 4), dtype=np.float64)
            for q in range(4):
                seg = w[:, q * q_len : (q + 1) * q_len, :]
                sub_mav[:, :, q] = np.log(np.mean(np.abs(seg), axis=1) + 1e-6)
                sub_wl[:, :, q] = np.log(np.sum(np.abs(seg[:, 1:, :] - seg[:, :-1, :]), axis=1) + 1e-6)

            per_ch = np.stack(
                [
                    mav,
                    log_rms,
                    wl,
                    mfl,
                    dasdv,
                    wamp,
                    form_factor,
                    crest_factor,
                    zc,
                    ssc,
                    mob,
                    comp,
                    skew,
                    kurt_log,
                    ar4[:, :, 0],
                    ar4[:, :, 1],
                    ar4[:, :, 2],
                    ar4[:, :, 3],
                    bp1,
                    bp2,
                    bp3,
                    bp4,
                    bp5,
                    mnf,
                    spec_ent,
                    emgcc[:, :, 0],
                    emgcc[:, :, 1],
                    emgcc[:, :, 2],
                    emgcc[:, :, 3],
                    haar_list[0],
                    haar_list[1],
                    haar_list[2],
                    haar_list[3],
                    haar_list[4],
                    sub_mav[:, :, 0],
                    sub_mav[:, :, 1],
                    sub_mav[:, :, 2],
                    sub_mav[:, :, 3],
                    sub_wl[:, :, 0],
                    sub_wl[:, :, 1],
                    sub_wl[:, :, 2],
                    sub_wl[:, :, 3],
                ],
                axis=2,
            ).reshape(n_b, self.PER_CHANNEL_DIM * c)

            # 7. Cross-Channel Spatial Synergy Matrix (Pairwise Correlations + Pairwise Log-RMS Ratios)
            if self.pair_dim > 0:
                corr_mat = np.einsum("ntc,ntd->ncd", z_w, z_w) / float(t_len)
                corr_mat = np.clip(corr_mat, -1.0, 1.0)
                corr_flat = corr_mat[:, self._triu_r, self._triu_c]

                # Full pairwise log-RMS differences: log(RMS_i) - log(RMS_j)
                ratio_flat = log_rms[:, self._triu_r] - log_rms[:, self._triu_c]
                combined = np.concatenate([per_ch, corr_flat, ratio_flat], axis=1)
            else:
                combined = per_ch

            out[start:end] = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        return out
