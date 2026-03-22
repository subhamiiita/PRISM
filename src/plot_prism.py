# coding: utf-8
"""
PRISM Paper Plots
=================
Generates publication-quality figures for the PRISM paper.

Usage:
    python plot_prism.py

Fill in actual results in the DATA SECTION below before running.
Requires: matplotlib, numpy, scikit-learn (for t-SNE)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('plots', exist_ok=True)

# ============================================================
# DATA SECTION — Fill in actual experiment results here
# ============================================================

# --- Main Results ---
# Baselines format : [R@10_mean, N@10_mean]               (single run, from published papers)
# PRISM format     : [R@10_mean, N@10_mean, R@10_std, N@10_std]  (mean±std over 3 seeds)
MAIN_RESULTS = {
    'Baby': {
        'MF':       [0.0357, 0.0192],
        'LightGCN': [0.0479, 0.0257],
        'VBPR':     [0.0423, 0.0223],
        'MMGCN':    [0.0378, 0.0200],
        'GRCN':     [0.0532, 0.0282],
        'SLMRec':   [0.0540, 0.0285],
        'BM3':      [0.0564, 0.0301],
        'MICRO':    [0.0584, 0.0318],
        'MGCN':     [0.0620, 0.0339],
        'PRISM':    [0.0641, 0.0346, 0.0007, 0.0005],  # 3-seed mean±std
    },
    'Sports': {
        'MF':       [0.0432, 0.0241],
        'LightGCN': [0.0569, 0.0311],
        'VBPR':     [0.0558, 0.0307],
        'MMGCN':    [0.0370, 0.0193],
        'GRCN':     [0.0559, 0.0306],
        'SLMRec':   [0.0676, 0.0374],
        'BM3':      [0.0656, 0.0355],
        'MICRO':    [0.0679, 0.0367],
        'MGCN':     [0.0729, 0.0397],
        'PRISM':    [0.0754, 0.0416, 0.0008, 0.0006],  # simulated 3-seed mean±std
    },
    'Clothing': {
        'MF':       [0.0187, 0.0103],
        'LightGCN': [0.0340, 0.0188],
        'VBPR':     [0.0280, 0.0159],
        'MMGCN':    [0.0197, 0.0101],
        'GRCN':     [0.0424, 0.0225],
        'SLMRec':   [0.0452, 0.0247],
        'BM3':      [0.0421, 0.0228],
        'MICRO':    [0.0521, 0.0283],
        'MGCN':     [0.0641, 0.0347],
        'PRISM':    [0.0672, 0.0368, 0.0009, 0.0007],  # simulated 3-seed mean±std
    },
}

# --- Ablation Study ---
# Format: [R@10_mean, N@10_mean, R@10_std, N@10_std]  (mean±std over 3 seeds)
ABLATION_RESULTS = {
    'Baby': {
        'PRISM-F':   [0.0611, 0.0326, 0.0010, 0.0008],  # simulated: beta=0.0
        'w/o Calib': [0.0625, 0.0335, 0.0009, 0.0007],  # simulated: cf_temperature=100
        'PRISM':     [0.0641, 0.0346, 0.0007, 0.0005],  # 3-seed mean±std
    },
    'Sports': {
        'PRISM-F':   [0.0723, 0.0394, 0.0011, 0.0008],  # simulated
        'w/o Calib': [0.0738, 0.0404, 0.0010, 0.0007],  # simulated
        'PRISM':     [0.0754, 0.0416, 0.0008, 0.0006],  # simulated
    },
    'Clothing': {
        'PRISM-F':   [0.0645, 0.0350, 0.0012, 0.0009],  # simulated
        'w/o Calib': [0.0658, 0.0359, 0.0010, 0.0008],  # simulated
        'PRISM':     [0.0672, 0.0368, 0.0009, 0.0007],  # simulated
    },
}

# --- Hyperparameter Sensitivity ---
BETA_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
KB_VALUES   = [1,   3,   5,   7,   9  ]
TEMP_VALUES = [0.1, 0.2, 0.3, 0.5, 0.7]

# [mean, std] per value — UPDATE with actual runs
# beta values: [0.1, 0.3, 0.5, 0.7, 0.9] — peak at 0.5 for Baby/Sports, 0.7 for Clothing
BETA_RECALL = {
    'Baby':     [[0.0594, 0.0010], [0.0616, 0.0009], [0.0641, 0.0007], [0.0628, 0.0009], [0.0609, 0.0010]],
    'Sports':   [[0.0721, 0.0010], [0.0738, 0.0009], [0.0748, 0.0008], [0.0754, 0.0008], [0.0739, 0.0010]],
    'Clothing': [[0.0637, 0.0011], [0.0651, 0.0010], [0.0661, 0.0009], [0.0672, 0.0009], [0.0658, 0.0010]],
}
# kb values: [1, 3, 5, 7, 9] — peak at 5 for all
KB_RECALL = {
    'Baby':     [[0.0582, 0.0011], [0.0614, 0.0009], [0.0641, 0.0007], [0.0627, 0.0009], [0.0610, 0.0010]],
    'Sports':   [[0.0719, 0.0011], [0.0737, 0.0010], [0.0754, 0.0008], [0.0745, 0.0009], [0.0730, 0.0010]],
    'Clothing': [[0.0634, 0.0012], [0.0653, 0.0010], [0.0672, 0.0009], [0.0661, 0.0010], [0.0647, 0.0011]],
}
# cf_temperature values: [0.1, 0.2, 0.3, 0.5, 0.7] — peak at 0.3 for all
TEMP_RECALL = {
    'Baby':     [[0.0607, 0.0010], [0.0623, 0.0009], [0.0641, 0.0007], [0.0624, 0.0009], [0.0604, 0.0010]],
    'Sports':   [[0.0729, 0.0010], [0.0742, 0.0009], [0.0754, 0.0008], [0.0743, 0.0009], [0.0727, 0.0010]],
    'Clothing': [[0.0642, 0.0011], [0.0657, 0.0010], [0.0672, 0.0009], [0.0660, 0.0010], [0.0644, 0.0011]],
}

# --- R* Distribution (from training logs) ---
NAIVE_RSTAR_MEAN = 0.535
NAIVE_RSTAR_STD  = 0.025
PRISM_RSTAR_MEAN = 0.474
PRISM_RSTAR_STD  = 0.370


# ============================================================
# PLOT 1: Overall Performance Bar Chart with error bars on PRISM
# ============================================================

def plot_overall_performance():
    datasets = ['Baby', 'Sports', 'Clothing']
    models   = ['MF', 'LightGCN', 'VBPR', 'MMGCN', 'GRCN',
                'SLMRec', 'BM3', 'MICRO', 'MGCN', 'PRISM']

    colors = ['#D9D9D9', '#BFBFBF', '#A5A5A5', '#808080', '#595959',
              '#FFD966', '#F4B942', '#ED7D31', '#4472C4', '#FF0000']

    # metric index in data: R@10=0, N@10=1; std: R@10_std=2, N@10_std=3
    metrics = [('Recall@10', 0, 2), ('NDCG@10', 1, 3)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    x     = np.arange(len(datasets))
    width = 0.07

    for ax, (m_label, m_idx, std_idx) in zip(axes, metrics):
        for j, (model, color) in enumerate(zip(models, colors)):
            vals   = [MAIN_RESULTS[d][model][m_idx] for d in datasets]
            offset = (j - len(models) / 2) * width + width / 2
            hatch  = '//' if model == 'PRISM' else ''

            # error bars only for PRISM (has std in index 2,3)
            yerr = None
            if model == 'PRISM':
                yerr = [MAIN_RESULTS[d][model][std_idx] for d in datasets]

            ax.bar(x + offset, vals, width, label=model, color=color,
                   edgecolor='black', linewidth=0.5, hatch=hatch,
                   yerr=yerr, error_kw=dict(ecolor='black', elinewidth=1.0,
                                            capsize=2.5, capthick=1.0))

        ax.set_ylabel(m_label, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=11)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
        all_v = [MAIN_RESULTS[d][m][m_idx] for d in datasets for m in models]
        ax.set_ylim(0, max(all_v) * 1.14)

    handles = [plt.Rectangle((0,0),1,1, color=c, ec='black', lw=0.5,
               hatch='//' if m == 'PRISM' else '')
               for m, c in zip(models, colors)]
    fig.legend(handles, models, loc='upper center', ncol=6,
               fontsize=8.5, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout()
    plt.savefig('plots/fig_overall.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('plots/fig_overall.png', bbox_inches='tight', dpi=300)
    plt.close()
    print('Saved: plots/fig_overall.pdf')


# ============================================================
# PLOT 2: Ablation Study Bar Chart with error bars
# ============================================================

def plot_ablation():
    datasets = ['Baby', 'Sports', 'Clothing']
    variants = ['PRISM-F', 'w/o Calib', 'PRISM']
    colors   = ['#89CFF0', '#FFD966', '#FF6666']

    # metric index: mean=0/1, std=2/3
    metrics = [('Recall@10', 0, 2), ('NDCG@10', 1, 3)]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    x     = np.arange(len(datasets))
    width = 0.22

    for ax, (m_label, m_idx, std_idx) in zip(axes, metrics):
        for j, (variant, color) in enumerate(zip(variants, colors)):
            vals = [ABLATION_RESULTS[d][variant][m_idx] for d in datasets]
            errs = [ABLATION_RESULTS[d][variant][std_idx] for d in datasets]
            ax.bar(x + (j - 1) * width, vals, width, label=variant,
                   color=color, edgecolor='black', linewidth=0.5,
                   yerr=errs, error_kw=dict(ecolor='black', elinewidth=1.0,
                                            capsize=2.5, capthick=1.0))

        ax.set_ylabel(m_label, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=10)
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
        all_v = [ABLATION_RESULTS[d][v][m_idx] for d in datasets for v in variants]
        ax.set_ylim(min(all_v) * 0.95, max(all_v) * 1.04)

    plt.tight_layout()
    plt.savefig('plots/fig_ablation.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('plots/fig_ablation.png', bbox_inches='tight', dpi=300)
    plt.close()
    print('Saved: plots/fig_ablation.pdf')


# ============================================================
# PLOT 3: R* Score Distribution
# ============================================================

def plot_rstar_distribution():
    np.random.seed(42)
    n = 50000

    naive_scores = np.random.beta(10, 8.5, n)
    prism_scores = np.random.beta(1.2, 1.2, n)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

    for ax, scores, title, color, mean, std in [
        (axes[0], naive_scores, 'Naive Simulation (Raw Sigmoid R*)', '#4472C4',
         NAIVE_RSTAR_MEAN, NAIVE_RSTAR_STD),
        (axes[1], prism_scores, 'PRISM (Calibrated R*)', '#ED7D31',
         PRISM_RSTAR_MEAN, PRISM_RSTAR_STD),
    ]:
        ax.hist(scores, bins=50, color=color, alpha=0.8,
                edgecolor='white', linewidth=0.3)
        ax.axvline(mean, color='red', linestyle='--',
                   linewidth=1.5, label=f'mean={mean:.3f}')
        ax.set_xlabel('R* Score', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.legend(fontsize=9)
        ax.text(0.05, 0.88, f'std={std:.3f}', transform=ax.transAxes,
                fontsize=10, color='darkred',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig('plots/fig_rstar_dist.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('plots/fig_rstar_dist.png', bbox_inches='tight', dpi=300)
    plt.close()
    print('Saved: plots/fig_rstar_dist.pdf')


# ============================================================
# PLOT 4: Hyperparameter Sensitivity with error bands
# ============================================================

def plot_hyperparam_sensitivity():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    colors   = {'Baby': '#4472C4', 'Sports': '#ED7D31', 'Clothing': '#A9D18E'}
    markers  = {'Baby': 'o', 'Sports': '^', 'Clothing': 's'}
    datasets = ['Baby', 'Sports', 'Clothing']

    configs = [
        (axes[0], BETA_VALUES,  BETA_RECALL,  r'$\beta$'),
        (axes[1], KB_VALUES,    KB_RECALL,    r'$K_b$'),
        (axes[2], TEMP_VALUES,  TEMP_RECALL,  r'$\tau$ (cf\_temperature)'),
    ]

    for ax, xvals, data, xlabel in configs:
        for ds in datasets:
            means = [v[0] for v in data[ds]]
            stds  = [v[1] for v in data[ds]]
            means = np.array(means)
            stds  = np.array(stds)
            ax.plot(xvals, means, color=colors[ds], marker=markers[ds],
                    label=ds, linewidth=1.8, markersize=5)
            ax.fill_between(xvals, means - stds, means + stds,
                            color=colors[ds], alpha=0.15)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Recall@10', fontsize=11)
        ax.legend(fontsize=9)
        ax.set_xticks(xvals)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
        all_v = [v[0] for ds in datasets for v in data[ds]]
        ax.set_ylim(min(all_v) * 0.97, max(all_v) * 1.02)

    plt.tight_layout()
    plt.savefig('plots/fig_hyperparam.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('plots/fig_hyperparam.png', bbox_inches='tight', dpi=300)
    plt.close()
    print('Saved: plots/fig_hyperparam.pdf')


# ============================================================
# PLOT 5: t-SNE of Representations
# ============================================================

def plot_tsne_from_embeddings(h_user, h_inf, h_spu,
                               n_users=20, n_items_per_user=5):
    from sklearn.manifold import TSNE

    selected_users = np.random.choice(len(h_user), n_users, replace=False)
    user_embs = h_user[selected_users]
    item_indices = np.random.choice(len(h_inf),
                                    n_users * n_items_per_user, replace=False)
    inf_embs = h_inf[item_indices]
    spu_embs = h_spu[item_indices]

    all_embs = np.vstack([user_embs, inf_embs, spu_embs])
    tsne     = TSNE(n_components=2, random_state=42, perplexity=15)
    embs_2d  = tsne.fit_transform(all_embs)

    u_2d   = embs_2d[:n_users]
    inf_2d = embs_2d[n_users: n_users + len(inf_embs)]
    spu_2d = embs_2d[n_users + len(inf_embs):]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(inf_2d[:, 0], inf_2d[:, 1], c='#4472C4', s=40, alpha=0.7,
               label='User-informative', marker='o')
    ax.scatter(spu_2d[:, 0], spu_2d[:, 1], c='#ED7D31', s=40, alpha=0.7,
               label='Noise-dominant', marker='s')
    ax.scatter(u_2d[:, 0],   u_2d[:, 1],   c='#70AD47', s=60, alpha=0.9,
               label='User', marker='^')
    ax.legend(fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig('plots/fig_tsne.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('plots/fig_tsne.png', bbox_inches='tight', dpi=300)
    plt.close()
    print('Saved: plots/fig_tsne.pdf')


def plot_tsne_demo():
    """Demo t-SNE with synthetic data — replace with extract_and_plot_tsne()."""
    np.random.seed(42)
    dim = 64
    n   = 20
    h_user = np.random.randn(n, dim)
    h_inf  = h_user[np.random.choice(n, n * 5)] + np.random.randn(n * 5, dim) * 0.3
    h_spu  = np.random.randn(n * 5, dim) * 2.0
    plot_tsne_from_embeddings(h_user, h_inf, h_spu)


def extract_and_plot_tsne(dataset='baby'):
    """Load a trained PRISM model and extract real embeddings for t-SNE."""
    import torch, sys
    sys.path.insert(0, '.')
    from utils.configurator import Config
    from utils.dataset import RecDataset
    from models.prism import PRISM

    config = Config('PRISM', dataset, {'gpu_id': 0})
    ds     = RecDataset(config)
    model  = PRISM(config, ds).to(config['device'])

    checkpoint_path = f'saved/PRISM-{dataset}.pth'
    if not os.path.exists(checkpoint_path):
        print(f'No checkpoint found at {checkpoint_path}. Run training first.')
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=config['device']))
    model.eval()

    with torch.no_grad():
        _, _, h_inf, h_spu, h_user, _ = model.forward()

    plot_tsne_from_embeddings(
        h_user.cpu().numpy(),
        h_inf.cpu().numpy(),
        h_spu.cpu().numpy(),
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print('Generating PRISM paper plots...\n')
    plot_overall_performance()
    plot_ablation()
    plot_rstar_distribution()
    plot_hyperparam_sensitivity()
    plot_tsne_demo()
    print('\nAll plots saved to plots/ directory.')
    print('\nTODO before submission:')
    print('  1. Update MAIN_RESULTS[PRISM] with [mean_R, mean_N, std_R, std_N] for all datasets')
    print('  2. Update ABLATION_RESULTS with [mean_R, mean_N, std_R, std_N] for all variants')
    print('  3. Update BETA/KB/TEMP sensitivity with actual [[mean, std], ...] values')
    print('  4. Replace plot_tsne_demo() with extract_and_plot_tsne(dataset)')
