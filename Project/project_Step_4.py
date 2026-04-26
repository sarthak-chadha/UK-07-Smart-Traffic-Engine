import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.loader import DataLoader
import numpy as np
import os
import glob
import re
import copy
import logging
import pickle
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, "Models")
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {DEVICE}")

# ── Hyper-parameters (all in one place) ───────────────────────────────────────
HIDDEN_DIM    = 128          # doubled: 102K-edge graph needs more capacity
HEADS_LAYER1  = 4
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
MAX_EPOCHS    = 200          # more room before early-stop kicks in
PATIENCE      = 25           # was 15 — too aggressive; val was still descending
GRAD_CLIP     = 1.0
TRAIN_FRAC    = 0.80
BATCH_SIZE    = 1
WARMUP_EPOCHS = 10           # linear LR warmup — prevents epoch-1 R²=-33 explosion

SMALL_DATASET_THRESHOLD = 40
NOISE_STD               = 0.01

# Tier 1: anything meaningfully below free-flow mean (0.9928)
SLIGHT_THRESHOLD  = 0.97    # 0.85 ≤ factor < 0.97  → slight slowdown
SLIGHT_UPWEIGHT   = 30.0

# Tier 2: genuinely jammed roads (< 0.7 confirmed by test.py)
HEAVY_THRESHOLD   = 0.85    # factor < 0.85  → heavy congestion
HEAVY_UPWEIGHT    = 120.0   # ~1/0.0089 ≈ 112 needed just to equalise

# Keep for backward-compat with metric logging (meaningful boundary)
CONGESTION_THRESHOLD = HEAVY_THRESHOLD

HIGHWAY_CATEGORIES = [
    "motorway", "trunk", "primary", "secondary",
    "tertiary", "residential", "service", "track"
]
# Importance weights: major arteries matter more, small roads matter less
HIGHWAY_LOSS_WEIGHTS = {
    "motorway"    : 5.0,
    "trunk"       : 4.0,
    "primary"     : 3.0,
    "secondary"   : 2.0,
    "tertiary"    : 1.5,
    "residential" : 1.0,
    "service"     : 0.6,
    "track"       : 0.4,
}
# One-hot slice starts at index 7 in edge_attr (after 7 continuous features)
ONEHOT_START_IDX = 7

def get_temporal_features(timestamp_str: str) -> torch.Tensor:
    """
    '2026-04-13_18-45'  →  5-dim cyclical feature vector.
      hour_sin/cos  →  time-of-day periodicity
      day_sin/cos   →  day-of-week periodicity
      is_weekend    →  binary flag
    """
    dt       = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M")
    hour     = dt.hour + dt.minute / 60.0
    day      = dt.weekday()
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin  = np.sin(2 * np.pi * day  / 7)
    day_cos  = np.cos(2 * np.pi * day  / 7)
    is_wknd  = 1.0 if day >= 5 else 0.0
    return torch.tensor(
        [hour_sin, hour_cos, day_sin, day_cos, is_wknd], dtype=torch.float
    )

N_TEMPORAL = 5

def load_all_snapshots() -> list:
    """
    Loads every dm_graph_live_*.pt snapshot in chronological order.

    For each snapshot Sₜ:
      • Appends 5 temporal features to every node (time-of-day, day-of-week).
      • Appends the traffic factor from Sₜ₋₁ as an extra edge feature ("lag").
        → For the first snapshot (no history), lag = 1.0 (free-flow assumed).
      • Normalises confidence and y to 1-D tensors.

    The lag feature lets the GNN see the traffic *trend* (clearing or worsening),
    not just the instantaneous value.
    """
    snapshot_files = sorted(
        glob.glob(os.path.join(DATASET_DIR, "dm_graph_live_*.pt"))
    )
    if not snapshot_files:
        raise FileNotFoundError(
            "No traffic snapshots found in Dataset/. Run Step 3 first."
        )

    dataset       = []
    prev_y        = None   # traffic factor from the previous snapshot [E]

    for fpath in snapshot_files:
        data = torch.load(fpath, weights_only=False)

        # ── Shape guards ──────────────────────────────────────────────────
        if data.confidence.dim() == 2:
            data.confidence = data.confidence.squeeze(-1)
        if data.y.dim() == 2:
            data.y = data.y.squeeze(-1)

        # ── Temporal node features ────────────────────────────────────────
        ts_match = re.search(r"live_(.+?)\.pt$", fpath)
        if ts_match:
            t_feats    = get_temporal_features(ts_match.group(1))         # [5]
            t_expanded = t_feats.unsqueeze(0).expand(data.num_nodes, -1) # [N,5]
            data.x     = torch.cat([data.x, t_expanded], dim=-1)

        # ── Spatio-Temporal Lag feature  ──────────────────
        # Use previous snapshot's y; if none (first snapshot), assume free-flow = 1.0
        if prev_y is not None and prev_y.shape[0] == data.edge_attr.shape[0]:
            lag = prev_y.unsqueeze(-1)       # [E, 1]
        else:
            lag = torch.ones(data.edge_attr.shape[0], 1)  # [E, 1]  free-flow

        data.edge_attr = torch.cat([data.edge_attr, lag], dim=-1)  # [E, F+1]
        prev_y         = data.y.clone()

        dataset.append(data)

    log.info(f"Loaded {len(dataset)} snapshot(s). "
             f"Edge features now: {dataset[0].num_edge_features} "
             f"(+1 lag column).")
    return dataset

def build_highway_weight_vector(edge_attr: torch.Tensor) -> torch.Tensor:
    """
    Reads the 8 one-hot highway columns from edge_attr and returns a
    per-edge importance weight tensor for use in the loss function.

    Edge layout (Step 2):  [length, speed, travel_time, oneway, lanes,
                             grade, grade_abs,  OH₀…OH₇]
    ONEHOT_START_IDX = 7
    """
    weight_values = torch.tensor(
        list(HIGHWAY_LOSS_WEIGHTS.values()), dtype=torch.float,
        device=edge_attr.device
    )                                                      # [8]
    one_hot = edge_attr[:, ONEHOT_START_IDX : ONEHOT_START_IDX + 8]  # [E, 8]
    # Each edge has exactly one '1' in the one-hot block → dot with weight_values
    hw_weight = (one_hot * weight_values).sum(dim=-1)      # [E]
    # Guard: if edge fell into the 'other' bucket, weight_values[-1] covers it
    return hw_weight.clamp(min=0.4)

class ElevationAwareGAT(torch.nn.Module):
    """
    Two-layer GATv2 encoder with residual (skip) connections + edge MLP head.

    Residual design :
      Layer 1 output is added back to a linearly projected copy of the raw
      input → prevents over-smoothing, preserves topographical features
      (elevation, grade) across layers.

    Architecture:
      proj  : Linear(node_in → hidden*heads)       [residual projection]
      conv1 : GATv2Conv  →  [N, hidden*heads]
      conv2 : GATv2Conv  →  [N, hidden]
      head  : MLP(node_u ‖ node_v ‖ edge_attr)  →  scalar ∈ [0,1]
    """

    def __init__(self, node_in: int, edge_in: int, hidden_dim: int = 64,
                 heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dropout    = dropout
        self.heads      = heads
        self.hidden_dim = hidden_dim

        # ── Encoder ───────────────────────────────────────────────────────
        self.conv1      = GATv2Conv(
            node_in, hidden_dim,
            edge_dim=edge_in, heads=heads, concat=True, dropout=dropout
        )
        self.conv2      = GATv2Conv(
            hidden_dim * heads, hidden_dim,
            edge_dim=edge_in, heads=1, concat=False, dropout=dropout
        )
        self.bn1        = torch.nn.BatchNorm1d(hidden_dim * heads)
        self.bn2        = torch.nn.BatchNorm1d(hidden_dim)

        # Residual projection: map raw input into the same dim as conv1 output
        self.res_proj   = torch.nn.Linear(node_in, hidden_dim * heads, bias=False)

        # ── Edge regression head ──────────────────────────────────────────
        self.regressor  = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2 + edge_in, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim // 2, 1),
            torch.nn.Sigmoid(),   # output ∈ [0,1] → matches TomTom factor
        )

    def forward(self, x, edge_index, edge_attr):
        # ── Layer 1 + Residual ────────────────────────────────────────────
        h   = self.conv1(x, edge_index, edge_attr)          # [N, hidden*heads]
        h   = self.bn1(h)
        h   = F.elu(h + self.res_proj(x))                   # ← residual add
        h   = F.dropout(h, p=self.dropout, training=self.training)

        # ── Layer 2 ────────────────────────────────────────────────────────
        h   = self.conv2(h, edge_index, edge_attr)           # [N, hidden]
        h   = self.bn2(h)
        h   = F.elu(h)

        # ── Edge prediction ───────────────────────────────────────────────
        row, col   = edge_index
        edge_repr  = torch.cat([h[row], h[col], edge_attr], dim=-1)
        return self.regressor(edge_repr).squeeze(-1)          # [E]

    def forward_with_attention(self, x, edge_index, edge_attr):
        """
        Identical to forward() but also returns the attention coefficients
        from both GATv2 layers for XAI analysis.

        Returns:
            pred        : [E]  — edge traffic factor predictions
            alpha1      : [E_att, heads]  — layer-1 attention weights
            alpha2      : [E_att, 1]      — layer-2 attention weights

        Note: E_att may differ from E when the graph has self-loops added
        internally by GATv2Conv. Use edge_index for alignment.
        """
        # Layer 1 — with attention
        h, (att_edge_idx1, alpha1) = self.conv1(
            x, edge_index, edge_attr,
            return_attention_weights=True
        )                                                   # alpha1: [E_att, heads]
        h = self.bn1(h)
        h = F.elu(h + self.res_proj(x))

        # Layer 2 — with attention
        h, (att_edge_idx2, alpha2) = self.conv2(
            h, edge_index, edge_attr,
            return_attention_weights=True
        )                                                   # alpha2: [E_att, 1]
        h = self.bn2(h)
        h = F.elu(h)

        row, col   = edge_index
        edge_repr  = torch.cat([h[row], h[col], edge_attr], dim=-1)
        pred       = self.regressor(edge_repr).squeeze(-1)
        return pred, att_edge_idx1, alpha1, att_edge_idx2, alpha2

def weighted_mse(pred: torch.Tensor, target: torch.Tensor,
                 confidence: torch.Tensor,
                 highway_weight: torch.Tensor) -> torch.Tensor:
    """
    Four-factor weighted MSE calibrated to the dataset distribution.

    Factor 1 — confidence        : TomTom reliability [0, 1]
    Factor 2 — highway_weight    : road importance (trunk=4× … track=0.4×)
    Factor 3 — slight_fac (30×) : slight slowdown (0.85 ≤ factor < 0.97)
    Factor 4 — heavy_fac (120×) : heavy jam (factor < 0.85)

    Gradient budget (from test.py data):
      Free-flow  99.11% ×  1.0 = 99.1 parts
      Slight     ~2.00% × 30.0 = 60.0 parts
      Heavy jam   0.89% ×120.0 = 106.8 parts  ← now competes on equal footing
    """
    combined   = confidence * highway_weight
    is_slight  = ((target < SLIGHT_THRESHOLD) & (target >= HEAVY_THRESHOLD)).float()
    is_heavy   = (target < HEAVY_THRESHOLD).float()
    cong_fac   = (1.0
                  + (SLIGHT_UPWEIGHT - 1.0) * is_slight
                  + (HEAVY_UPWEIGHT  - 1.0) * is_heavy)
    return (combined * cong_fac * (pred - target) ** 2).mean()

def run_epoch(model, loader, optimizer=None, grad_clip: float = 0.0,
              use_noise: bool = False):
    """
    Single pass over the loader. Returns (mean_loss, preds_np, targets_np).

    use_noise : if True, adds Gaussian noise to node features during the
                forward pass (training only) to prevent coordinate memorisation.
                Noise is NOT applied during validation.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss              = 0.0
    all_preds, all_targets  = [], []

    with torch.set_grad_enabled(is_train):
        for data in loader:
            data = data.to(DEVICE)

            # ── Gaussian Noise Augmentation ───────────────────────────────
            # Applied only during training and only when dataset is small.
            # Perturbs node features (coords, elevation) so the model learns
            # terrain topology, not exact OSM numbers.
            x_in = data.x
            if is_train and use_noise:
                noise = torch.randn_like(x_in) * NOISE_STD
                x_in  = x_in + noise

            pred   = model(x_in, data.edge_index, data.edge_attr)     # [E]
            target = data.y.to(DEVICE)                                 # [E]
            conf   = data.confidence.to(DEVICE)                        # [E]
            hw_w   = build_highway_weight_vector(data.edge_attr)       # [E]

            loss = weighted_mse(pred, target, conf, hw_w)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss   += loss.item()
            all_preds    .append(pred.detach().cpu().numpy())
            all_targets  .append(target.detach().cpu().numpy())

    mean_loss   = total_loss / max(len(loader), 1)
    concat_pred = np.concatenate(all_preds)
    concat_tgt  = np.concatenate(all_targets)
    return mean_loss, concat_pred, concat_tgt

def analyze_attention(model: torch.nn.Module, val_loader, save_dir: str) -> None:
    """
    Extracts GATv2 attention weights on the validation set and produces:
      • Top-10 most-attended edges (by mean layer-1 attention)
      • A bar chart of attention distribution over highway types
      • A heatmap showing how each attention head focuses differently

    This is Explainable AI (XAI): it answers questions like
    'Does high elevation cause the model to attend more to uphill roads?'
    Saved as attention_analysis.png in Models/.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        model.eval()
        all_alpha1   = []   # layer-1 attention per edge, per head
        all_edge_hw  = []   # highway type label for each edge (for grouping)

        with torch.no_grad():
            for data in val_loader:
                data = data.to(DEVICE)
                _, att_eidx1, alpha1, _, _ = model.forward_with_attention(
                    data.x, data.edge_index, data.edge_attr
                )
                # alpha1: [E_att, heads] — mean over heads → [E_att]
                mean_attn = alpha1.mean(dim=-1).cpu().numpy()      # [E_att]

                # Map attention edge index back to highway type
                # att_eidx1 has shape [2, E_att]; col = destination node
                # We match to original edge_attr by destination column index
                # (approximate — GATv2 may add self-loops)
                n_edges   = min(len(mean_attn), data.edge_attr.shape[0])
                one_hot   = data.edge_attr[:n_edges,
                            ONEHOT_START_IDX : ONEHOT_START_IDX + 8].cpu().numpy()
                hw_labels = [HIGHWAY_CATEGORIES[row.argmax()] for row in one_hot]

                all_alpha1.extend(mean_attn[:n_edges].tolist())
                all_edge_hw.extend(hw_labels)

        all_alpha1  = np.array(all_alpha1)
        all_edge_hw = np.array(all_edge_hw)

        # ── Build per-highway mean attention (skip absent categories) ──────────
        hw_mean_attn = {}           # cat → float (only present categories)
        hw_absent    = []           # categories not in this dataset
        for cat in HIGHWAY_CATEGORIES:
            mask = all_edge_hw == cat
            if mask.any():
                hw_mean_attn[cat] = float(all_alpha1[mask].mean())
            else:
                hw_absent.append(cat)

        # ── Plot ──────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#0f1117")
        fig.suptitle("GATv2 Attention Weight Analysis (XAI)",
                     color="white", fontsize=14, fontweight="bold", y=1.01)

        # Left: Mean attention by highway type (present only)
        ax = axes[0]
        ax.set_facecolor("#1a1d27")
        color_map = {
            "motorway": "#ff4d6d", "trunk": "#ff7c4d", "primary": "#ffd166",
            "secondary": "#06d6a0", "tertiary": "#118ab2",
            "residential": "#7c5cfc", "service": "#aaaaaa", "track": "#666666",
        }
        # Sort by attention descending for clarity
        sorted_cats = sorted(hw_mean_attn, key=lambda c: -hw_mean_attn[c])
        present_vals   = [hw_mean_attn[c] for c in sorted_cats]
        present_colors = [color_map[c] for c in sorted_cats]
        bars = ax.barh(sorted_cats, present_vals,
                       color=present_colors, edgecolor="#333", height=0.6)
        ax.set_xlabel("Mean Attention Weight (Layer 1)", color="white", fontsize=10)
        ax.set_title("Attention by Road Type",
                     color="white", fontsize=12, fontweight="bold")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        for bar, val in zip(bars, present_vals):
            ax.text(val + 0.0005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", ha="left",
                    color="white", fontsize=8)
        if hw_absent:
            ax.set_title(
                f"Attention by Road Type\n"
                f"(absent in dataset: {', '.join(hw_absent)})",
                color="white", fontsize=11, fontweight="bold"
            )

        # Right: Attention distribution histogram
        ax2 = axes[1]
        ax2.set_facecolor("#1a1d27")
        ax2.hist(all_alpha1, bins=80, color="#00d4ff", edgecolor="#333", alpha=0.85)
        ax2.axvline(all_alpha1.mean(), color="#ffd166", linestyle="--",
                    linewidth=1.5, label=f"Mean = {all_alpha1.mean():.4f}")
        ax2.set_xlabel("Attention Weight",   color="white", fontsize=10)
        ax2.set_ylabel("Edge Count",         color="white", fontsize=10)
        ax2.set_title("Overall Attention Distribution",
                      color="white", fontsize=12, fontweight="bold")
        ax2.tick_params(colors="white")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#333")
        ax2.legend(fontsize=9, facecolor="#0f1117", labelcolor="white",
                   edgecolor="#444")

        plt.tight_layout(pad=2.0)
        out_path = os.path.join(save_dir, "attention_analysis.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        log.info(f"Attention analysis saved → {out_path}")

        # ── Console XAI summary ───────────────────────────────────────────
        log.info("─" * 54)
        log.info("  GATv2 Attention by Road Type (Layer 1 mean, sorted)")
        max_val = max(present_vals) if present_vals else 1.0
        for cat in sorted_cats:
            val = hw_mean_attn[cat]
            bar = "█" * int(val / max_val * 20)
            log.info(f"  {cat:<14} {val:.5f}  {bar}")
        for cat in hw_absent:
            log.info(f"  {cat:<14}  —  (absent in Dehradun–Mussoorie dataset)")
        log.info("─" * 54)

    except ImportError:
        log.warning("matplotlib not installed — skipping attention analysis.")
    except Exception as exc:
        log.warning(f"Attention analysis failed ({exc}) — skipping.")

def plot_validation(preds: np.ndarray, targets: np.ndarray,
                    save_dir: str, epoch: int) -> None:
    """
    Scatter plot: Predicted vs Actual traffic factor.
    A perfect model → all points on the diagonal.
    Saved as a PNG so it can be inspected without a display.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")      # non-interactive backend — safe for servers
        import matplotlib.pyplot as plt

        r2  = r2_score(targets, preds) if len(preds) > 1 else float("nan")
        mae = mean_absolute_error(targets, preds)
        mse = mean_squared_error(targets, preds)
        rmse = np.sqrt(mse)

        # ── Accuracy band metric ──────────────────────────────────────────
        # "Within 10%" accuracy: fraction of predictions within ±0.10 of target
        within_10pct = np.mean(np.abs(preds - targets) <= 0.10) * 100
        within_5pct  = np.mean(np.abs(preds - targets) <= 0.05) * 100

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor("#0f1117")

        # ── Left: Scatter ─────────────────────────────────────────────────
        ax = axes[0]
        ax.set_facecolor("#1a1d27")
        ax.scatter(targets, preds, alpha=0.3, s=4, color="#00d4ff",
                   label="Edge predictions")
        lims = [0.0, 1.05]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Actual Traffic Factor",  color="white", fontsize=11)
        ax.set_ylabel("Predicted Traffic Factor", color="white", fontsize=11)
        ax.set_title("Predicted vs Actual (Validation Set)",
                     color="white", fontsize=13, fontweight="bold")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        # ── Congestion-zone metrics ───────────────────────────────────────
        cong_mask = targets < CONGESTION_THRESHOLD
        r2_cong   = (
            r2_score(targets[cong_mask], preds[cong_mask])
            if cong_mask.sum() > 1 else float("nan")
        )
        cong_pct  = cong_mask.mean() * 100

        # Metric box
        metrics_text = (
            f"R²  (all)    = {r2:+.4f}\n"
            f"R²  (cong.)  = {r2_cong:+.4f}  ← key metric\n"
            f"Cong. edges  = {cong_pct:.1f}%\n"
            f"MAE          = {mae:.4f}\n"
            f"RMSE         = {rmse:.4f}\n"
            f"±10% acc     = {within_10pct:.1f}%\n"
            f"±5%  acc     = {within_5pct:.1f}%"
        )
        ax.text(0.04, 0.96, metrics_text, transform=ax.transAxes,
                fontsize=8.5, va="top", ha="left", color="#00ff88",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1d27",
                          edgecolor="#00ff88", alpha=0.85),
                fontfamily="monospace")

        # Highlight congested points in red on the scatter
        if cong_mask.any():
            ax.scatter(targets[cong_mask], preds[cong_mask],
                       alpha=0.6, s=8, color="#ff4d6d", zorder=5,
                       label=f"Congested ({cong_pct:.0f}%)")
        ax.legend(fontsize=9, facecolor="#0f1117", labelcolor="white",
                  edgecolor="#444")

        # ── Right: Residual histogram ─────────────────────────────────────
        ax2 = axes[1]
        ax2.set_facecolor("#1a1d27")
        residuals = preds - targets
        ax2.hist(residuals[~cong_mask], bins=60, color="#7c5cfc",
                 edgecolor="#333", alpha=0.75, label="Free-flow edges")
        if cong_mask.any():
            ax2.hist(residuals[cong_mask], bins=30, color="#ff4d6d",
                     edgecolor="#333", alpha=0.85, label="Congested edges")
        ax2.axvline(0, color="red", linestyle="--", linewidth=1.5,
                    label="Zero error")
        ax2.axvline(residuals.mean(), color="#00ff88", linestyle="-",
                    linewidth=1.5, label=f"Mean = {residuals.mean():.4f}")
        ax2.set_xlabel("Residual (Pred − Actual)", color="white", fontsize=11)
        ax2.set_ylabel("Count",                    color="white", fontsize=11)
        ax2.set_title("Residual Distribution (free-flow vs congested)",
                      color="white", fontsize=12, fontweight="bold")
        ax2.tick_params(colors="white")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#444")
        ax2.legend(fontsize=9, facecolor="#0f1117", labelcolor="white",
                   edgecolor="#444")

        plt.tight_layout(pad=2.0)
        out_path = os.path.join(save_dir, f"validation_plot_ep{epoch}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        log.info(f"Validation plot saved → {out_path}")

        # ── Console summary ───────────────────────────────────────────────
        log.info("─" * 50)
        log.info(f"  R² (all edges)   : {r2:+.4f}   ← misleading when 95% = free-flow")
        log.info(f"  R² (congested)   : {r2_cong:+.4f}   ← meaningful metric")
        log.info(f"  Congested edges  : {cong_pct:.1f}% of val set")
        log.info(f"  MAE              : {mae:.4f}")
        log.info(f"  RMSE             : {rmse:.4f}")
        log.info(f"  Accuracy ±10%%   : {within_10pct:.1f}%")
        log.info(f"  Accuracy ±5%%    :  {within_5pct:.1f}%")
        log.info("─" * 50)

    except ImportError:
        log.warning("matplotlib not installed — skipping validation plot.")

def train():
    # ── Load data ─────────────────────────────────────────────────────────
    all_data  = load_all_snapshots()
    n_train   = max(1, int(TRAIN_FRAC * len(all_data)))
    train_set = all_data[:n_train]
    val_set   = all_data[n_train:] if len(all_data) > 1 else all_data

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)

    log.info(f"Train snapshots  : {len(train_set)}")
    log.info(f"Val   snapshots  : {len(val_set)}")

    # ── Auto-detect feature dimensions ────────────────────────────────────
    sample      = all_data[0]
    node_in_dim = sample.num_node_features   # base node feats + N_TEMPORAL
    edge_in_dim = sample.num_edge_features   # Step-2 feats + 1 lag column
    log.info(f"Node feature dim : {node_in_dim}  (base + {N_TEMPORAL} temporal)")
    log.info(f"Edge feature dim : {edge_in_dim}  (Step-2 + 1 lag)")

    # ── Gaussian Noise decision ───────────────────────────────────────────
    # Only enable noise augmentation when we have few training snapshots.
    # With many snapshots the model sees enough natural variety on its own.
    use_noise = len(train_set) < SMALL_DATASET_THRESHOLD
    if use_noise:
        log.info(
            f"Gaussian noise augmentation ENABLED  "
            f"(train_set={len(train_set)} < threshold={SMALL_DATASET_THRESHOLD}, "
            f"std={NOISE_STD})"
        )
    else:
        log.info(
            f"Gaussian noise augmentation DISABLED  "
            f"(train_set={len(train_set)} >= threshold={SMALL_DATASET_THRESHOLD})"
        )

    # ── Build model ───────────────────────────────────────────────────────
    model = ElevationAwareGAT(
        node_in    = node_in_dim,
        edge_in    = edge_in_dim,
        hidden_dim = HIDDEN_DIM,
        heads      = HEADS_LAYER1,
        dropout    = 0.2,    # 0.1→0.2 — better generalisation on imbalanced classes
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters : {n_params:,}")

    # ── Optimiser + scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    warmup    = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.05, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    cosine    = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS - WARMUP_EPOCHS, eta_min=1e-5
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS]
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    best_weights   = None
    patience_count = 0
    final_val_pred = None
    final_val_tgt  = None

    log.info("=" * 60)
    log.info("Starting training …")
    log.info("=" * 60)

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, _, _             = run_epoch(model, train_loader, optimizer, GRAD_CLIP,
                                                  use_noise=use_noise)
        val_loss,  val_pred, val_tgt = run_epoch(model, val_loader)   # no noise on val

        scheduler.step()

        val_r2  = r2_score(val_tgt, val_pred) if len(val_pred) > 1 else float("nan")
        val_mae = mean_absolute_error(val_tgt, val_pred)

        # ── Three-tier congestion metrics ──────────────────────────────────
        slight_mask  = (val_tgt < SLIGHT_THRESHOLD) & (val_tgt >= HEAVY_THRESHOLD)
        heavy_mask   = val_tgt < HEAVY_THRESHOLD
        cong_mask    = val_tgt < SLIGHT_THRESHOLD   # any slowdown

        r2_slight = (
            r2_score(val_tgt[slight_mask], val_pred[slight_mask])
            if slight_mask.sum() > 1 else float("nan")
        )
        r2_heavy = (
            r2_score(val_tgt[heavy_mask], val_pred[heavy_mask])
            if heavy_mask.sum() > 1 else float("nan")
        )
        val_r2_cong = r2_heavy   # expose as 'cong' for checkpoint
        cong_pct    = cong_mask.mean() * 100

        if epoch % 10 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            log.info(
                f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"R²: {val_r2:+.3f} | "
                f"R²(slight {slight_mask.mean()*100:.1f}%): {r2_slight:+.3f} | "
                f"R²(heavy {heavy_mask.mean()*100:.2f}%): {r2_heavy:+.3f} | "
                f"MAE: {val_mae:.4f} | LR: {lr_now:.2e}"
            )

        # ── Checkpoint best model ──────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_weights   = copy.deepcopy(model.state_dict())
            final_val_pred = val_pred.copy()
            final_val_tgt  = val_tgt.copy()
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            log.info(
                f"Early stopping at epoch {epoch} "
                f"(no improvement for {PATIENCE} epochs)."
            )
            break

    # ── Restore best ──────────────────────────────────────────────────────
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # ── Metadata Persistence  ─────────────────────────────
    # Load Step 2's scaler objects so Step 5 (inference) can normalise live data
    # exactly the same way training data was normalised.
    node_scaler, edge_scaler = None, None
    scaler_path = os.path.join(MODELS_DIR, "scalers.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scalers_dict = pickle.load(f)
        node_scaler = scalers_dict.get("node_scaler")
        edge_scaler = scalers_dict.get("edge_scaler")
        log.info("Scalers loaded from Models/scalers.pkl")
    else:
        log.warning(
            "scalers.pkl not found — checkpoint will not include scalers. "
            "Run Step 2 with scaler saving enabled for full inference support."
        )

    # ── Save checkpoint ───────────────────────────────────────────────────
    save_path = os.path.join(MODELS_DIR, "traffic_model_v1.pth")
    torch.save(
        {
            # ── Architecture config (needed to rebuild model in Step 5) ──
            "model_state_dict"  : model.state_dict(),
            "node_in_dim"       : node_in_dim,
            "edge_in_dim"       : edge_in_dim,
            "hidden_dim"        : HIDDEN_DIM,
            "heads"             : HEADS_LAYER1,
            # ── Training provenance ───────────────────────────────────────
            "best_val_loss"     : best_val_loss,
            "epochs_trained"    : epoch,
            # ── Feature metadata (Step 5 uses these to reconstruct inputs) ─
            "node_feature_names": ["lat", "lon", "degree", "elevation",
                                   "hour_sin", "hour_cos",
                                   "day_sin",  "day_cos", "is_weekend"],
            "edge_feature_names": ["length", "speed", "travel_time",
                                   "oneway", "lanes", "grade", "grade_abs",
                                   *[f"hw_{c}" for c in HIGHWAY_CATEGORIES],
                                   "lag_traffic_factor"],
            "highway_categories": HIGHWAY_CATEGORIES,
            "n_temporal_feats"  : N_TEMPORAL,
            # ── Scalers (crucial for live-data normalisation in Step 5) ──
            "node_scaler"       : node_scaler,
            "edge_scaler"       : edge_scaler,
        },
        save_path,
    )

    log.info("=" * 60)
    log.info(f"Training complete.")
    log.info(f"  Best val loss   : {best_val_loss:.6f}")
    log.info(f"  Checkpoint      : {save_path}")
    log.info("=" * 60)

    # ── Visual Validation ─────────────────────────────────────────────────
    if final_val_pred is not None and final_val_tgt is not None:
        log.info("Generating validation plots …")
        plot_validation(final_val_pred, final_val_tgt, MODELS_DIR, epoch)
    else:
        log.warning("No validation predictions to plot.")

    # ── Attention Analysis (XAI) ──────────────────────────────────────────
    log.info("Generating attention weight analysis …")
    analyze_attention(model, val_loader, MODELS_DIR)

if __name__ == "__main__":
    train()