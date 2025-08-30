# plot_heatmap.py
# Render token×layer divergence heatmap
import numpy as np
import matplotlib.pyplot as plt

def plot_heatmap(np_path, out_png, window_tokens=40):
    H = np.load(np_path)  # [layers, tokens]
    plt.figure(figsize=(8, 4.5))
    im = plt.imshow(H, aspect='auto', origin='lower')
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Cosine distance (1 - cos)")
    xticks = [0, window_tokens, 2*window_tokens]
    xtlabels = [f"-{window_tokens}", "0 (insertion)", f"+{window_tokens}"]
    plt.xticks(xticks, xtlabels)
    plt.yticks(range(H.shape[0]), [f"L{l}" for l in range(1, H.shape[0]+1)])
    plt.title("Token×Layer Divergence Around Cue Insertion (Monitored–Hidden)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
