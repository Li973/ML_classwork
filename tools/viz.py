import matplotlib.pyplot as plt
import pandas as pd
#支持可视化

def plot_causal_chain(findings, src, tgt, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    chain = findings[(findings['src'] == src) & (findings['tgt'] == tgt)]
    if chain.empty:
        ax.text(0.5, 0.5, f"无 {src}→{tgt} 因果证据", ha='center', va='center')
        ax.axis('off')
        return ax

    ax.scatter(chain['window_center'], chain['coef'],
               c=chain['pcorr'], cmap='RdYlGn', s=60, vmin=-0.6, vmax=0.6)
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_title(f"{src} → {tgt} 因果强度演化")
    ax.set_ylabel('回归系数 β')
    plt.colorbar(ax.collections[0], ax=ax, label='偏相关 ρ')
    return ax