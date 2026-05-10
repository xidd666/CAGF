"""
t-SNE Feature Distribution Visualization for CAGF Model

Subplot descriptions:
  - Left: Raw features (RoBERTa + WavLM extracted, before CME/Gate fusion)
  - Middle: Features after Concatenation processing (direct concatenation, no cross-modal interaction)
  - Right: Features after CAGF processing (processed through CME + Gated Fusion)
"""

import argparse
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.manifold import TSNE
from tqdm import tqdm

from utils.context_model import rob_wavlm_cme_context, rob_wavlm_cc_context
from utils.data_loader import data_loader
from utils.en_train import EnConfig

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Subclasses with intermediate feature output (Hook forward)
# ─────────────────────────────────────────────────────────────────────────────

class CAGF_with_features(rob_wavlm_cme_context):
    """Inherits from original model, overrides forward to return intermediate features for visualization."""

    def forward_with_features(self,
                               text_inputs, text_mask,
                               text_context_inputs, text_context_mask,
                               audio_inputs, audio_mask,
                               audio_context_inputs, audio_context_mask):
        """
        Returns:
            outputs: dict {'T', 'A', 'M'}  (raw logit outputs)
            orig_features:  (B, 2048)  Raw features = concat(input_pooler, A_features_mean)
            cagf_features:  (B, 1024)  Fused features after CAGF processing
        """
        # ── Text feature extraction (original + context) ──────────────────────────
        raw_output = self.roberta_model(text_inputs, text_mask, return_dict=True)
        T_hidden_states = raw_output.last_hidden_state
        input_pooler = raw_output["pooler_output"]           # (B, 1024)

        raw_output_context = self.roberta_model(text_context_inputs, text_context_mask, return_dict=True)
        T_context_hidden_states = raw_output_context.last_hidden_state
        context_pooler = raw_output_context["pooler_output"] # (B, 1024)

        # ── Audio feature extraction (original + context) ──────────────────────────
        def extract_audio_feat(a_inputs, a_mask):
            out = self.wavlm_model(a_inputs, attention_mask=a_mask,
                                   output_hidden_states=True, return_dict=True)
            all_hs = torch.stack(out.hidden_states[1:], dim=0)  # (24, B, T, 1024)
            weights = torch.softmax(self.layer_weights, dim=0)
            A_hs = torch.sum(weights.view(-1, 1, 1, 1) * all_hs, dim=0)  # (B, T, 1024)
            A_hs = self.audio_adapter(A_hs)
            feats, mask_new_list = [], []
            for b in range(A_hs.shape[0]):
                pad = int(a_mask[b].sum().item())
                if pad == 0:
                    pad = A_hs.shape[1]
                mask_new_list.append(pad)
                feats.append(torch.mean(A_hs[b][:pad], 0))
            feats = torch.stack(feats, 0).to(a_inputs.device)   # (B, 1024)
            mask_new = torch.zeros(A_hs.shape[0], A_hs.shape[1]).to(a_inputs.device)
            for b in range(len(mask_new_list)):
                mask_new[b][:mask_new_list[b]] = 1
            return feats, A_hs, mask_new

        A_features, A_hidden_states, audio_mask_new = extract_audio_feat(audio_inputs, audio_mask)
        A_context_features, A_context_hidden_states, audio_context_mask_new = extract_audio_feat(audio_context_inputs, audio_context_mask)

        T_features = torch.cat((input_pooler, context_pooler), dim=1)          # (B, 2048)
        A_feats_cat = torch.cat((A_features, A_context_features), dim=1)       # (B, 2048)
        T_output = self.T_output_layers(T_features)
        A_output = self.A_output_layers(A_feats_cat)

        # ── CME cross-modal interaction ─────────────────────────────────────────
        t_in, t_attn = self.prepend_cls(T_hidden_states, text_mask, 'text')
        a_in, a_attn = self.prepend_cls(A_hidden_states, audio_mask_new, 'audio')
        tc_in, tc_attn = self.prepend_cls(T_context_hidden_states, text_context_mask, 'text')
        ac_in, ac_attn = self.prepend_cls(A_context_hidden_states, audio_context_mask_new, 'audio')

        for layer in self.CME_layers:
            t_in, a_in = layer(t_in, t_attn, a_in, a_attn)
        for layer in self.CME_layers:
            tc_in, ac_in = layer(tc_in, tc_attn, ac_in, ac_attn)

        # ── Gated fusion ───────────────────────────────────────────────
        text_cls  = t_in[:, 0, :]    # (B, 1024)
        audio_cls = a_in[:, 0, :]    # (B, 1024)
        text_ctx_cls  = tc_in[:, 0, :]  # (B, 1024)
        audio_ctx_cls = ac_in[:, 0, :]  # (B, 1024)

        fused_hidden_states = self.gated_fusion(
            text_cls, audio_cls, text_ctx_cls, audio_ctx_cls
        )  # (B, 1024)

        fused_output = self.fused_output_layers(fused_hidden_states)

        # ── Raw feature definition ───────────────────────────────────────────
        # Take text pooler + audio mean pooled average, concatenate to 1024+1024=2048 dims
        orig_features = torch.cat([input_pooler, A_features], dim=-1)  # (B, 2048)

        outputs = {'T': T_output, 'A': A_output, 'M': fused_output}
        return outputs, orig_features, fused_hidden_states


class CC_with_features(rob_wavlm_cc_context):
    """rob_wavlm_cc_context (WavLM-base + direct concatenation, no CME) feature extraction.
    Corresponds to training command: --model cc
    Extract representation through fused_output_layers[:-1] (penultimate):
      [0]Dropout→[1]Linear(3584→2048)→[2]ReLU→[3]Linear(2048→1024)→[4]ReLU → [5]Linear(1024→1)
    [:-1] output (B, 1024), consistent with CAGF gated_fusion output dimension
    """

    def forward_with_features(self,
                               text_inputs, text_mask,
                               text_context_inputs, text_context_mask,
                               audio_inputs, audio_mask,
                               audio_context_inputs, audio_context_mask):
        raw_output = self.roberta_model(text_inputs, text_mask, return_dict=True)
        input_pooler = raw_output["pooler_output"]              # (B, 1024)

        raw_output_context = self.roberta_model(text_context_inputs, text_context_mask, return_dict=True)
        context_pooler = raw_output_context["pooler_output"]    # (B, 1024)

        def avg_pool(a_inputs, a_mask):
            out = self.wavlm_model(a_inputs, attention_mask=a_mask)
            hs = out.last_hidden_state  # (B, T, 768)
            feats = []
            for b in range(hs.shape[0]):
                pad = int(a_mask[b].sum().item())
                if pad == 0:
                    pad = hs.shape[1]
                feats.append(torch.mean(hs[b][:pad], 0))  # (768,)
            return torch.stack(feats, 0).to(a_inputs.device)   # (B, 768)

        A_features = avg_pool(audio_inputs, audio_mask)                      # (B, 768)
        A_context_features = avg_pool(audio_context_inputs, audio_context_mask)  # (B, 768)

        T_features = torch.cat((input_pooler, context_pooler), dim=1)        # (B, 2048)
        A_cat = torch.cat((A_features, A_context_features), dim=1)           # (B, 1536)
        fused_input = torch.cat((T_features, A_cat), dim=1)                  # (B, 3584)

        # Extract CC model learned representation through fused_output_layers[:-1] (remove prediction head)
        cc_features = self.fused_output_layers[:-1](fused_input)             # (B, 1024)
        return cc_features


# ─────────────────────────────────────────────────────────────────────────────
# 2. Main feature extraction loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(model, loader):
    model.eval()
    orig_list, cagf_list, label_list = [], [], []

    for batch in tqdm(loader, desc="Extracting features"):
        text_inputs          = batch["text_tokens"].to(device)
        text_mask            = batch["text_masks"].to(device)
        text_context_inputs  = batch["text_context_tokens"].to(device)
        text_context_mask    = batch["text_context_masks"].to(device)
        audio_inputs         = batch["audio_inputs"].to(device)
        audio_mask           = batch["audio_masks"].to(device)
        audio_context_inputs = batch["audio_context_inputs"].to(device)
        audio_context_mask   = batch["audio_context_masks"].to(device)
        targets              = batch["targets"].cpu().numpy().flatten()

        _, orig_feat, cagf_feat = model.forward_with_features(
            text_inputs, text_mask,
            text_context_inputs, text_context_mask,
            audio_inputs, audio_mask,
            audio_context_inputs, audio_context_mask
        )

        orig_list.append(orig_feat.cpu().float().numpy())
        cagf_list.append(cagf_feat.cpu().float().numpy())
        label_list.append(targets)

    orig_arr  = np.concatenate(orig_list, axis=0)   # (N, 2048)
    cagf_arr  = np.concatenate(cagf_list, axis=0)   # (N, 1024)
    labels    = np.concatenate(label_list, axis=0)   # (N,)
    return orig_arr, cagf_arr, labels


# ─────────────────────────────────────────────────────────────────────────────
# 3. t-SNE dimensionality reduction + visualization
# ─────────────────────────────────────────────────────────────────────────────

def run_tsne(features, n_components=2, perplexity=30, random_state=42):
    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                random_state=random_state, max_iter=1000, verbose=1)
    return tsne.fit_transform(features)


def plot_tsne(orig_emb, cc_emb, cagf_emb, labels, output_path, dataset='MOSI'):
    cmap = 'viridis'
    vmin, vmax = labels.min(), labels.max()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5),
                              gridspec_kw={'wspace': 0.12})

    titles = [
        'Raw Features',
        'Processed by Concatenation',
        'Processed by CAGF'
    ]
    embs = [orig_emb, cc_emb, cagf_emb]

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

    # Right side shared colorbar
    cbar = fig.colorbar(sc, ax=axes, orientation='vertical',
                        fraction=0.018, pad=0.02, shrink=0.85)
    cbar.set_label('Sentiment Score', fontsize=10, labelpad=8, rotation=270, va='bottom')
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

    # Build config (consistent with training)
    config = EnConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        dataset_name=args.dataset,
        text_context_len=args.text_context_len,
        audio_context_len=args.audio_context_len,
        use_gated_fusion=True,
        num_hidden_layers=5,
        gpu_ids='0',
    )

    # Load data (test set)
    print("Loading dataset...")
    _, test_loader, _ = data_loader(
        config.batch_size,
        config.dataset_name,
        text_context_length=config.text_context_len,
        audio_context_length=config.audio_context_len
    )

    def load_model(model_cls, use_gated, ckpt_path):
        # rob_wavlm_cc_context does not accept use_gated_fusion parameter
        if model_cls is CC_with_features:
            m = model_cls(config).to(device)
        else:
            m = model_cls(config, use_gated_fusion=use_gated).to(device)
        for param in m.wavlm_model.feature_extractor.parameters():
            param.requires_grad = False
        sd = torch.load(ckpt_path, map_location=device)
        new_sd = {k.replace('module.', ''): v for k, v in sd.items()}
        m.load_state_dict(new_sd, strict=True)
        print(f"✅ Checkpoint loaded successfully: {ckpt_path}")
        return m

    # ── CAGF model (gated fusion) ────────────────────────────────────────
    print("\nInitializing CAGF model (Gated Fusion)...")
    model_cagf = load_model(CAGF_with_features, True, args.checkpoint)

    # ── CC model (concatenation fusion) ──────────────────────────────────────────
    print("\nInitializing Concatenation model...")
    model_cc = load_model(CC_with_features, False, args.checkpoint_cc)

    # ── Extract features ──────────────────────────────────────────────
    print("\nExtracting CAGF features...")
    orig_feat, cagf_feat, labels = extract_features(model_cagf, test_loader)
    print(f"  Raw features {orig_feat.shape}, CAGF features {cagf_feat.shape}, labels {labels.shape}")

    print("\nExtracting Concatenation features...")
    model_cc.eval()
    cc_list = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting CC features"):
            cc_f = model_cc.forward_with_features(
                batch["text_tokens"].to(device), batch["text_masks"].to(device),
                batch["text_context_tokens"].to(device), batch["text_context_masks"].to(device),
                batch["audio_inputs"].to(device), batch["audio_masks"].to(device),
                batch["audio_context_inputs"].to(device), batch["audio_context_masks"].to(device)
            )
            cc_list.append(cc_f.cpu().float().numpy())
    cc_feat = np.concatenate(cc_list, axis=0)
    print(f"  Concatenation features {cc_feat.shape}")

    # ── t-SNE dimensionality reduction ──────────────────────────────────────────────
    print("\nRunning t-SNE (raw features)...")
    orig_emb = run_tsne(orig_feat, perplexity=args.perplexity)

    print("\nRunning t-SNE (Concatenation features)...")
    cc_emb = run_tsne(cc_feat, perplexity=args.perplexity)

    print("\nRunning t-SNE (CAGF features)...")
    cagf_emb = run_tsne(cagf_feat, perplexity=args.perplexity)

    # ── Visualization ──────────────────────────────────────────────────────
    plot_tsne(orig_emb, cc_emb, cagf_emb, labels, args.output, dataset=args.dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="t-SNE visualization for CAGF model")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to CAGF (Gated Fusion) checkpoint')
    parser.add_argument('--checkpoint_cc', type=str, required=True,
                        help='Path to Concatenation (concatenation fusion) checkpoint')
    parser.add_argument('--dataset',    type=str, default='mosi',
                        help='Dataset name: mosi or mosei')
    parser.add_argument('--seed',       type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--text_context_len',  type=int, default=5)
    parser.add_argument('--audio_context_len', type=int, default=5)
    parser.add_argument('--perplexity', type=int, default=30,
                        help='t-SNE perplexity parameter (recommended 20-50)')
    parser.add_argument('--output', type=str, default='tsne_mosi.png',
                        help='Output image path')
    args = parser.parse_args()
    main(args)
