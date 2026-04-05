import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset_loader import load_dataset_fast

SPLIT_PATH = "abc_filtered_final.json"
SAVE_DIR = "distribution_plots2"


def get_counts(args):
    path, split = args
    try:
        with np.load(path) as f:
            return {
                "Face Count": len(f["face_points"]),
                "Edge Count": len(f["edge_adjacency"]),
                "Split": split,
            }
    except:
        return None


if __name__ == "__main__":
    dataset = load_dataset_fast(
        SPLIT_PATH,
        root_dir="/cache/yanko/dataset/abc_preprocessed/organized_by_face_count/",
    )

    tasks = []
    for split in ["train", "val", "test"]:
        label = split.capitalize()
        tasks.extend([(p, label) for p in dataset[split]])

    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(get_counts, tasks), total=len(tasks)))

    df = pd.DataFrame([r for r in results if r])
    Path(SAVE_DIR).mkdir(exist_ok=True)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    colors = sns.color_palette("deep", 3)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for i, col in enumerate(["Face Count", "Edge Count"]):
        sns.histplot(
            data=df,
            x=col,
            hue="Split",
            element="step",
            stat="density",
            common_norm=False,
            kde=True,
            ax=axes[i],
            palette=colors,
            alpha=0.3,
        )
        axes[i].set_title(f"Distribution of {col}s")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/distribution_hist_kde.png", dpi=300)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for i, col in enumerate(["Face Count", "Edge Count"]):
        sns.boxplot(
            data=df,
            x="Split",
            y=col,
            hue="Split",
            ax=axes[i],
            palette=colors,
            legend=False,
        )
        axes[i].set_title(f"Boxplot of {col}s")
        axes[i].set_yscale("log")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/distribution_boxplot.png", dpi=300)
    plt.close()

    plot_df = df.sample(n=min(len(df), 10000), random_state=42)
    g = sns.jointplot(
        data=plot_df,
        x="Face Count",
        y="Edge Count",
        hue="Split",
        kind="scatter",
        palette=colors,
        height=8,
        alpha=0.5,
        s=15,
    )
    g.fig.suptitle("Face vs Edge Count Relationship (Sampled)", y=1.02)
    plt.savefig(f"{SAVE_DIR}/distribution_joint.png", dpi=300)
    plt.close()
