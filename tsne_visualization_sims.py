"""
t-SNE Feature Distribution Visualization for CH-SIMS dataset

Subplot descriptions:
  - Left: Raw features (RoBERTa + Hubert extracted, before CME interaction)
  - Middle: Features after Concatenation processing (direct concatenation, no cross-modal interaction)
  - Right: Features after CME processing (processed through cross-modal encoder CME layers)
"""

import argparse
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from utils.ch_model import rob_hub_cc, rob_hub_cme
from utils.ch_train import ChConfig
from utils.data_loader import data_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Subclasses with intermediate feature output
# ─────────────────────────────────────────────────────────────────────────────

class CME_with_features(rob_hub_cme):
    """Inherits from rob_hub_cme, overrides forward to return:
    - orig_features:  (B, 2048) Raw features = concat(T_pooler:1024, A_mean:1024), before CME
    - cme_features:   (B, 512)  After CME processing = penultimate layer of fused_output_layers
    """

    def forward_with_features(self, text_inputs, text_mask, audio_inputs, audio_mask):
        # ── Text feature extraction ──────────────────────────────────────────
        raw_output = self.roberta_model(text_inputs, text_mask, return_dict=True)
        T_hidden_states = raw_output.last_hidden_state
        T_features = raw_output["pooler_output"]          # (B, 1024)

        # ── Audio feature extraction ──────────────────────────────────────────
        audio_out = self.hubert_model(audio_inputs, audio_mask)
        A_hidden_states = audio_out.last_hidden_state
        audio_lengths = audio_mask.sum(dim=1).long()
        feat_lengths = self.hubert_model._get_feat_extract_output_lengths(audio_lengths)

        A_features = []
        audio_mask_idx_new = []
        for b in range(A_hidden_states.shape[0]):
            feat_len = max(int(feat_lengths[b].item()), 1)
            audio_mask_idx_new.append(feat_len)
            A_features.append(torch.mean(A_hidden_states[b][:feat_len], 0))  # (1024,)
        A_features = torch.stack(A_features, 0).to(audio_inputs.device)      # (B, 1024)

        # New audio mask (valid frames after downsampling)
        audio_mask_new = torch.zeros(
            A_hidden_states.shape[0], A_hidden_states.shape[1]
        ).to(audio_inputs.device)
        for b in range(len(audio_mask_idx_new)):
            audio_mask_new[b][:audio_mask_idx_new[b]] = 1

        # ── Raw features = backbone concatenation ─────────────────────────────
        orig_features = torch.cat([T_features, A_features], dim=-1)  # (B, 2048)

        # ── CME cross-modal interaction ─────────────────────────────────────
        # detach, consistent with training
        text_hidden_for_cme = T_hidden_states.detach()
        audio_hidden_for_cme = A_hidden_states.detach()

        t_in, t_attn = self.prepend_cls(text_hidden_for_cme, text_mask, 'text')
        a_in, a_attn = self.prepend_cls(audio_hidden_for_cme, audio_mask_new, 'audio')

        for layer in self.CME_layers:
            t_in, a_in = layer(t_in, t_attn, a_in, a_attn)

        # v1: fused = concat(text_cls, audio_cls) = (B, 2048)
        fused_hidden = torch.cat([t_in[:, 0, :], a_in[:, 0, :]], dim=-1)  # (B, 2048)

        # ── Extract penultimate layer representation from fused_output_layers ──
        # fused_output_layers: Dropout, Linear(2048→1024), ReLU, Linear(1024→512), ReLU, Linear(512→1)
        # penultimate = output before last Linear (B, 512)
        penultimate_layers = self.fused_output_layers[:-1]  # Exclude last Linear layer
        cme_features = penultimate_layers(fused_hidden)       # (B, 512)

        # ── Single-modality outputs (for complete forward, maintain interface) ──
        T_output = self.T_output_layers(T_features)
        A_output = self.A_output_layers(A_features)
        fused_correction = self.fused_output_layers(fused_hidden)
        fused_output = (T_output + A_output) / 2 + fused_correction

        outputs = {'T': T_output, 'A': A_output, 'M': fused_output}
        return outputs, orig_features, cme_features


class CC_with_features(rob_hub_cc):
    """Inherits from rob_hub_cc, extracts penultimate layer representation from fused_output_layers:
    cc_features = fused_output_layers[:-1](concat(T:1024, A:1024)) = (B, 512)
    """

    def forward_with_features(self, text_inputs, text_mask, audio_inputs, audio_mask):
        raw_output = self.roberta_model(text_inputs, text_mask, return_dict=True)
        T_features = raw_output["pooler_output"]          # (B, 1024)

        audio_out = self.hubert_model(audio_inputs, audio_mask)
        A_hidden_states = audio_out.last_hidden_state
        audio_lengths = audio_mask.sum(dim=1).long()
        feat_lengths = self.hubert_model._get_feat_extract_output_lengths(audio_lengths)

        A_features = []
        for b in range(A_hidden_states.shape[0]):
            feat_len = max(int(feat_lengths[b].item()), 1)
            A_features.append(torch.mean(A_hidden_states[b][:feat_len], 0))
        A_features = torch.stack(A_features, 0).to(audio_inputs.device)      # (B, 1024)

        fused_input = torch.cat([T_features, A_features], dim=-1)             # (B, 2048)

        # Extract penultimate layer representation from fused_output_layers, align with CME
        # fused_output_layers: Dropout, Linear(2048→1024), ReLU, Linear(1024→1024), ReLU,
        #                       Linear(1024→512), ReLU, Linear(512→1)
        # penultimate = output before last Linear (B, 512)
        penultimate_layers = self.fused_output_layers[:-1]
        cc_features = penultimate_layers(fused_input)                          # (B, 512)
        return cc_features


# ─────────────────────────────────────────────────────────────────────────────
# 2. Main feature extraction loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_cme_features(model, loader):
    """Extract orig / cme features from CME_with_features model."""
    model.eval()
    orig_list, cme_list, label_list = [], [], []

    for batch in tqdm(loader, desc="Extracting CME features"):
        text_inputs  = batch["text_tokens"].to(device)
        text_mask    = batch["text_masks"].to(device)
        audio_inputs = batch["audio_inputs"].to(device)
        audio_mask   = batch["audio_masks"].to(device)
        targets      = batch["targets"]["M"].cpu().numpy().flatten()

        _, orig_f, cme_f = model.forward_with_features(
            text_inputs, text_mask, audio_inputs, audio_mask
        )
        orig_list.append(orig_f.cpu().float().numpy())
        cme_list.append(cme_f.cpu().float().numpy())
        label_list.append(targets)

    orig_arr  = np.concatenate(orig_list, axis=0)   # (N, 2048)
    cme_arr   = np.concatenate(cme_list,  axis=0)   # (N, 2048)
    labels    = np.concatenate(label_list, axis=0)   # (N,)
    return orig_arr, cme_arr, labels


@torch.no_grad()
def extract_cc_features(model, loader):
    """Extract concatenation features from CC_with_features model."""
    model.eval()
    cc_list = []

    for batch in tqdm(loader, desc="Extracting CC features"):
        text_inputs  = batch["text_tokens"].to(device)
        text_mask    = batch["text_masks"].to(device)
        audio_inputs = batch["audio_inputs"].to(device)
        audio_mask   = batch["audio_masks"].to(device)

        cc_f = model.forward_with_features(
            text_inputs, text_mask, audio_inputs, audio_mask
        )
        cc_list.append(cc_f.cpu().float().numpy())

    return np.concatenate(cc_list, axis=0)   # (N, 2048)


# ─────────────────────────────────────────────────────────────────────────────
# 3. t-SNE dimensionality reduction + visualization
# ─────────────────────────────────────────────────────────────────────────────

def run_tsne(features, n_components=2, perplexity=30, random_state=42):
    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                random_state=random_state, max_iter=1000, verbose=1)
    return tsne.fit_transform(features)


def plot_tsne(orig_emb, cc_emb, cme_emb, labels, output_path, dataset='SIMS'):
    cmap = 'viridis'
    vmin, vmax = labels.min(), labels.max()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5),
                             gridspec_kw={'wspace': 0.12})

    titles = [
        'Raw Features',
        'Processed by Concatenation',
        'Processed by CAGF'
    ]
    embs = [orig_emb, cc_emb, cme_emb]

    sc = None
    for ax, emb, title in zip(axes, embs, titles):
        sc = ax.scatter(emb[:, 0], emb[:, 1],
                        c=labels, cmap=cmap,
                        vmin=vmin, vmax=vmax,
                        alpha=0.75, s=14, linewidths=0)
        ax.set_title(title, fontsize=12, pad=8)
        ax.tick_params(labelsize=8, length=3)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color('#888888')
        ax.set_xticks([])
        ax.set_yticks([])

    # Shared colorbar on the right
    cbar = fig.colorbar(sc, ax=axes, orientation='vertical',
                        fraction=0.018, pad=0.02, shrink=0.85)
    cbar.set_label('Sentiment Score', fontsize=10, labelpad=8,
                   rotation=270, va='bottom')
    cbar.ax.tick_params(labelsize=9)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Image saved to: {output_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main program
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Build ChConfig (consistent with training)
    # Note: SIMS does not have use_gated_fusion / text_context_len / audio_context_len
    config = ChConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        dataset_name='sims',
        num_hidden_layers=args.num_hidden_layers,
        model='cme',
    )

    # Load data (test set)
    print("Loading CH-SIMS dataset...")
    _, test_loader, _ = data_loader(
        config.batch_size,
        'sims',
    )

    def load_model(model_cls, ckpt_path):
        m = model_cls(config).to(device)
        sd = torch.load(ckpt_path, map_location=device)
        new_sd = {k.replace('module.', ''): v for k, v in sd.items()}
        m.load_state_dict(new_sd, strict=True)
        print(f"✅ Checkpoint loaded successfully: {ckpt_path}")
        return m

    # ── CME model ────────────────────────────────────────────────
    print("\nInitializing CME model (Cross-Modal Encoder)...")
    model_cme = load_model(CME_with_features, args.checkpoint)

    # ── CC model ─────────────────────────────────────────────────
    print("\nInitializing Concatenation model...")
    model_cc = load_model(CC_with_features, args.checkpoint_cc)

    # ── Extract features ────────────────────────────────────────────────
    print("\nExtracting CME model features...")
    orig_feat, cme_feat, labels = extract_cme_features(model_cme, test_loader)
    print(f"  Raw features {orig_feat.shape}, CME features {cme_feat.shape}, labels {labels.shape}")

    print("\nExtracting CC features...")
    cc_feat = extract_cc_features(model_cc, test_loader)
    print(f"  Concatenation features {cc_feat.shape}")

    # ── t-SNE dimensionality reduction ──────────────────────────────────────────────
    print("\nRunning t-SNE (raw features)...")
    orig_emb = run_tsne(orig_feat, perplexity=args.perplexity)

    print("\nRunning t-SNE (Concatenation features)...")
    cc_emb = run_tsne(cc_feat, perplexity=args.perplexity)

    print("\nRunning t-SNE (CME features)...")
    cme_emb = run_tsne(cme_feat, perplexity=args.perplexity)

    # ── Visualization ──────────────────────────────────────────────────
    plot_tsne(orig_emb, cc_emb, cme_emb, labels, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="t-SNE visualization for CH-SIMS")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to CME (Cross-Modal Encoder) checkpoint')
    parser.add_argument('--checkpoint_cc', type=str, required=True,
                        help='Path to Concatenation (concatenation fusion) checkpoint'))
    parser.add_argument('--seed',             type=int, default=1)
    parser.add_argument('--batch_size',       type=int, default=8)
    parser.add_argument('--num_hidden_layers',type=int, default=5,
                        help='Number of CME layers (default 5, consistent with run.py --num_hidden_layers default)')
    parser.add_argument('--perplexity',       type=int, default=30,
                        help='t-SNE perplexity parameter (recommended 20-50)')
    parser.add_argument('--output', type=str, default='tsne_sims.png',
                        help='Output image path')
    args = parser.parse_args()
    main(args)
