#!/usr/bin/env python3

from io import BytesIO
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
from chemplot import Plotter
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Circle
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.cluster import KMeans


def make_distinct_colors(n: int):
    """Good distinct colors; fall back to HSV if needed."""
    bases = []
    for name in ("tab20", "tab20b", "tab20c"):
        cmap = plt.cm.get_cmap(name, 20)
        bases.extend([cmap(i) for i in range(cmap.N)])
    if n <= len(bases):
        return bases[:n]
    hues = np.linspace(0, 1, n, endpoint=False)
    return [mpl.colors.hsv_to_rgb((h, 0.65, 0.95)) for h in hues]


def pick_embedding_columns(df: pd.DataFrame):
    """Pick (x, y) columns or the first two numeric ones."""
    preferred = [
        ("x", "y"),
        ("X", "Y"),
        ("tsne-1", "tsne-2"),
        ("TSNE-1", "TSNE-2"),
        ("umap-1", "umap-2"),
        ("UMAP-1", "UMAP-2"),
        ("pc1", "pc2"),
        ("PC1", "PC2"),
        ("dim1", "dim2"),
        ("Dim1", "Dim2"),
        ("component_1", "component_2"),
    ]
    for a, b in preferred:
        if a in df.columns and b in df.columns:
            return a, b
    num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    var_cols = [c for c in num_cols if np.nan_to_num(df[c].var()) > 0]
    if len(var_cols) < 2:
        raise RuntimeError("Could not find 2 numeric embedding columns")
    return var_cols[0], var_cols[1]


def mol_to_transparent_image(mol, size=(70, 70)):
    """RDKit Cairo drawer with transparent background; keeps alpha when grayscaled."""
    if mol is None:
        return None
    drawer = Draw.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    if hasattr(opts, "clearBackground"):
        opts.clearBackground = False
    if hasattr(opts, "bondLineWidth"):
        opts.bondLineWidth = 4
    if hasattr(opts, "minFontSize"):
        opts.minFontSize = 45
    if hasattr(opts, "maxFontSize"):
        opts.maxFontSize = 45

    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    png_data = drawer.GetDrawingText()
    pil_img = Image.open(BytesIO(png_data)).convert("RGBA")

    # gray_rgb = pil_img.convert("L").convert("RGBA")
    # gray_rgb.putalpha(pil_img.getchannel("A"))
    return pil_img


def main():
    # Inputs
    data_path = Path(__file__).parent.parent.parent / "data" / "functional_groups.csv"
    smiles = list(pd.read_csv(data_path)["SMILES"])
    n_samples = len(smiles)

    sim_type = "structural"  # "structural" or "tailored"
    method = "tsne"  # "tsne", "umap", or "pca"
    n_clusters = 25
    random_state = 0

    plot_filename = "tsne_clusters.pdf"

    # Embedding
    cp = Plotter.from_smiles(smiles, sim_type=sim_type)
    if method == "tsne":
        perplexity = max(5, min(50, max(2, (n_samples - 1) // 3)))
        cp.tsne(perplexity=perplexity, random_state=random_state)
    elif method == "umap":
        cp.umap(
            n_neighbors=min(15, max(2, n_samples - 1)),
            min_dist=0.1,
            random_state=random_state,
        )
    elif method == "pca":
        cp.pca()
    else:
        raise ValueError(f"Unknown method: {method}")

    # Clusters (use Chemplot if available, else KMeans)
    try:
        cp.cluster(n_clusters=min(n_clusters, n_samples), random_state=random_state)
        labels = np.asarray(cp.df_plot_xy.get("cluster", cp.df_plot_xy.get("Cluster")))
    except Exception:
        labels = None

    ax = cp.visualize_plot(size=12, clusters=True)
    ax.set_title("")

    # Styling
    fig = ax.get_figure()
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_xlabel(ax.get_xlabel(), fontsize=10, labelpad=6)
    ax.set_ylabel(ax.get_ylabel(), fontsize=10, labelpad=6)

    df = cp.df_plot_xy.copy()
    xcol, ycol = pick_embedding_columns(df)
    coords = df[[xcol, ycol]].to_numpy()

    if labels is None:
        if hasattr(cp, "cluster_labels_"):
            labels = np.asarray(cp.cluster_labels_)
        elif hasattr(cp, "labels_"):
            labels = np.asarray(cp.labels_)
        else:
            kmeans = KMeans(
                n_clusters=min(n_clusters, n_samples),
                n_init=10,
                random_state=random_state,
            )
            labels = kmeans.fit_predict(coords)
            df["cluster"] = labels

    # Choose one representative per cluster (closest to centroid)
    rep_indices = []
    uniq = sorted(np.unique(labels))
    for k in uniq:
        mask = labels == k
        cl = coords[mask]
        centroid = cl.mean(axis=0)
        j = np.argmin(((cl - centroid) ** 2).sum(axis=1))
        rep_idx = df.index[mask][j]
        rep_indices.append(int(rep_idx))

    # Overlay molecule thumbnails at representatives
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    THUMB_PX = (512, 512)
    THUMB_ZOOM = 0.2

    for idx in rep_indices:
        mol = mols[idx]
        if mol is None:
            continue
        pil_img = mol_to_transparent_image(mol, size=THUMB_PX)
        if pil_img is None:
            continue
        im = OffsetImage(np.asarray(pil_img), zoom=THUMB_ZOOM)
        ab = AnnotationBbox(
            im, (coords[idx, 0], coords[idx, 1]), frameon=False, zorder=10
        )
        ax.add_artist(ab)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Recolor points by cluster
    for coll in list(ax.collections):
        coll.remove()
    palette = make_distinct_colors(len(uniq))
    for i, k in enumerate(uniq):
        pts = coords[labels == k]
        ax.scatter(
            pts[:, 0], pts[:, 1], s=16, color=palette[i], edgecolors="none", alpha=0.9
        )

    # Circles around representative thumbnails
    r = 0.0034 * max(xlim[1] - xlim[0], ylim[1] - ylim[0])  # Smaller multiplier
    for idx in rep_indices:
        k = labels[idx]
        i = np.where(np.array(uniq) == k)[0][0]
        c = palette[i]
        ax.add_patch(
            Circle(
                (coords[idx, 0], coords[idx, 1]),
                radius=r,
                fill=False,
                linewidth=0.5,
                edgecolor="black",
                zorder=12,
            )
        )
    ax.set_aspect("equal")

    fig.savefig(plot_filename, dpi=600, bbox_inches="tight", facecolor="white")
    print(f"Cluster plot saved to {plot_filename} (centroids shown as molecule images)")


if __name__ == "__main__":
    main()
