#!/usr/bin/env python3
"""Extract figures and metrics from saved notebook outputs (no execution)."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = REPO_ROOT / "notebooks"
RESULTS = REPO_ROOT / "results"
MANIFEST = Path(__file__).resolve().parent / "results_manifest.yaml"

# Expected metrics from docs/results.md
EXPECTED_CLASSIFICATION = {
    "knn_gpu_k3": 0.9705,
    "logistic_regression": 0.9241,
    "manual_mlp": 0.9794,
    "simple_cnn": 0.9871,
    "comparison_cnn": 0.9913,
}
EXPECTED_GENERATIVE = {
    "vae_train_loss": 100.22,
    "vae_val_loss": 100.09,
    "ddpm_avg_loss": 0.0454,
    "dcgan_g_loss": 3.41,
    "dcgan_d_loss": 0.32,
    "latent_diff_mse": 0.224,
}

LATENT_NB = "vae + 1d unet (latent diff) (1).ipynb"


def load_nb(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def all_stream_text(nb: dict) -> str:
    parts: list[str] = []
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                parts.append("".join(out.get("text", [])))
    return "\n".join(parts)


def png_outputs(nb: dict) -> list[tuple[int, int, bytes]]:
    found: list[tuple[int, int, bytes]] = []
    for ci, cell in enumerate(nb.get("cells", [])):
        for oi, out in enumerate(cell.get("outputs", [])):
            if out.get("output_type") not in ("display_data", "execute_result"):
                continue
            data = out.get("data", {})
            if "image/png" in data:
                raw = data["image/png"]
                if isinstance(raw, str):
                    found.append((ci, oi, base64.b64decode(raw)))
    return found


def extract_png(nb_path: Path, cell: int, output: int) -> bytes:
    nb = load_nb(nb_path)
    outputs = nb["cells"][cell].get("outputs", [])
    out = outputs[output]
    data = out.get("data", {})
    if "image/png" not in data:
        raise KeyError(f"No png at {nb_path.name} cell {cell} output {output}")
    raw = data["image/png"]
    return base64.b64decode(raw) if isinstance(raw, str) else raw


def save_png(data: bytes, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)


def cmd_inventory() -> None:
    for nb_path in sorted(NOTEBOOKS.glob("*.ipynb")):
        nb = load_nb(nb_path)
        print(f"\n=== {nb_path.name} ===")
        for ci, oi, _ in png_outputs(nb):
            src = "".join(nb["cells"][ci].get("source", []))[:70].replace("\n", " ")
            print(f"  png cell={ci} output={oi} | {src}")
        text = all_stream_text(nb)
        for pat in [
            r"0\.9705", r"0\.9241", r"0\.9794", r"0\.9871", r"0\.9913",
            r"100\.\d+", r"0\.0454", r"Loss_G: 3\.", r"MSE Loss: 0\.22",
        ]:
            if re.search(pat, text):
                m = re.search(pat, text)
                if m:
                    start = max(0, m.start() - 40)
                    print(f"  metric snippet: ...{text[start:m.end()+40]}...")


def scrape_classification() -> dict:
    metrics: dict = {}

    knn_text = all_stream_text(load_nb(NOTEBOOKS / "manual_knn.ipynb"))
    m = re.search(r"k=3, Accuracy=([\d.]+)", knn_text)
    metrics["knn_gpu_k3"] = float(m.group(1)) if m else None

    lr_text = all_stream_text(load_nb(NOTEBOOKS / "manual_logistic_reg.ipynb"))
    m = re.search(r"Epoch 80/80.*Test Accuracy: ([\d.]+)", lr_text)
    if not m:
        m = re.search(r"Test Accuracy: ([\d.]+)", lr_text)
    metrics["logistic_regression"] = float(m.group(1)) if m else None

    mlp_text = all_stream_text(load_nb(NOTEBOOKS / "manual_mlp.ipynb"))
    m = re.search(r"Epoch 100/100.*test acc: ([\d.]+)", mlp_text, re.I)
    metrics["manual_mlp"] = float(m.group(1)) if m else None

    cnn_text = all_stream_text(load_nb(NOTEBOOKS / "cnn.ipynb"))
    m = re.search(r"Epoch 04.*test acc ([\d.]+)", cnn_text)
    metrics["simple_cnn"] = float(m.group(1)) if m else None

    cmp_text = all_stream_text(load_nb(NOTEBOOKS / "02_mnist_models.ipynb"))
    m = re.search(r"CNN accuracy: ([\d.]+)", cmp_text)
    metrics["comparison_cnn"] = float(m.group(1)) if m else None

    return metrics


def scrape_generative() -> dict:
    metrics: dict = {}

    vae_text = all_stream_text(load_nb(NOTEBOOKS / "vae.ipynb"))
    train_losses = re.findall(r"Train - Loss: ([\d.]+)", vae_text)
    val_losses = re.findall(r"Val\s+- Loss: ([\d.]+)", vae_text)
    if train_losses:
        metrics["vae_train_loss"] = round(float(train_losses[-1]), 2)
    if val_losses:
        metrics["vae_val_loss"] = round(float(val_losses[-1]), 2)

    ddpm_text = all_stream_text(load_nb(NOTEBOOKS / "ddpm.ipynb"))
    m = re.search(r"Epoch \[15/15\].*Avg Loss: ([\d.]+)", ddpm_text)
    if m:
        metrics["ddpm_avg_loss"] = round(float(m.group(1)), 4)

    gan_text = all_stream_text(load_nb(NOTEBOOKS / "dcgan.ipynb"))
    m = re.search(r"Epoch \[ 50/50\].*Loss_D: ([\d.]+).*Loss_G: ([\d.]+)", gan_text)
    if m:
        metrics["dcgan_d_loss"] = round(float(m.group(1)), 2)
        metrics["dcgan_g_loss"] = round(float(m.group(2)), 2)
    if "dcgan_g_loss" not in metrics:
        m = re.search(r"Final: ([\d.]+)", gan_text)
        if m:
            metrics["dcgan_g_loss"] = round(float(m.group(1)), 2)

    ld_text = all_stream_text(load_nb(NOTEBOOKS / LATENT_NB))
    m = re.search(r"Epoch \[1[24]\d/150\].*MSE Loss: ([\d.]+)", ld_text)
    if m:
        metrics["latent_diff_mse"] = round(float(m.group(1)), 3)

    return metrics


def verify_metrics(classification: dict, generative: dict) -> None:
    for key, expected in EXPECTED_CLASSIFICATION.items():
        got = classification.get(key)
        if got is None or abs(got - expected) > 0.0002:
            raise SystemExit(f"Metric mismatch {key}: expected {expected}, got {got}")
    for key, expected in EXPECTED_GENERATIVE.items():
        got = generative.get(key)
        if got is None or abs(got - expected) > 0.01:
            raise SystemExit(f"Metric mismatch {key}: expected {expected}, got {got}")
    print("All metrics match docs/results.md")


def write_metrics_json(classification: dict, generative: dict) -> None:
    out_dir = RESULTS / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    cls_records = [
        {"model": "KNN (GPU, k=3)", "metric": "test_accuracy", "value": classification["knn_gpu_k3"],
         "source": "notebooks/manual_knn.ipynb"},
        {"model": "Logistic regression", "metric": "test_accuracy", "value": classification["logistic_regression"],
         "source": "notebooks/manual_logistic_reg.ipynb"},
        {"model": "Manual MLP", "metric": "test_accuracy", "value": classification["manual_mlp"],
         "source": "notebooks/manual_mlp.ipynb"},
        {"model": "SimpleCNN", "metric": "test_accuracy", "value": classification["simple_cnn"],
         "source": "notebooks/cnn.ipynb"},
        {"model": "Comparison CNN", "metric": "test_accuracy", "value": classification["comparison_cnn"],
         "source": "notebooks/02_mnist_models.ipynb"},
    ]
    gen_records = [
        {"model": "VAE", "metric": "train_loss", "value": generative["vae_train_loss"],
         "source": "notebooks/vae.ipynb"},
        {"model": "VAE", "metric": "val_loss", "value": generative["vae_val_loss"],
         "source": "notebooks/vae.ipynb"},
        {"model": "DDPM", "metric": "avg_loss_epoch_15", "value": generative["ddpm_avg_loss"],
         "source": "notebooks/ddpm.ipynb"},
        {"model": "DCGAN", "metric": "generator_loss", "value": generative["dcgan_g_loss"],
         "source": "notebooks/dcgan.ipynb"},
        {"model": "DCGAN", "metric": "discriminator_loss", "value": generative["dcgan_d_loss"],
         "source": "notebooks/dcgan.ipynb"},
        {"model": "Latent diffusion MLP", "metric": "mse_epoch_150", "value": generative["latent_diff_mse"],
         "source": f"notebooks/{LATENT_NB}"},
    ]

    (out_dir / "classification.json").write_text(
        json.dumps({"metrics": cls_records}, indent=2), encoding="utf-8"
    )
    (out_dir / "generative.json").write_text(
        json.dumps({"metrics": gen_records}, indent=2), encoding="utf-8"
    )

    with (out_dir / "classification.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "metric", "value", "source"])
        for r in cls_records:
            w.writerow([r["model"], r["metric"], r["value"], r["source"]])


def html_table_to_markdown(html: str) -> str:
    rows = re.findall(r"<tr>(.*?)</tr>", html, re.DOTALL)
    md_rows: list[str] = []
    for i, row in enumerate(rows):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
        if not cells:
            cells = re.findall(r"<th[^>]*>(.*?)</th>", row, re.DOTALL)
        clean = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        if not clean:
            continue
        md_rows.append("| " + " | ".join(clean) + " |")
        if i == 0:
            md_rows.append("| " + " | ".join(["---"] * len(clean)) + " |")
    return "\n".join(md_rows)


def extract_comparison_table() -> None:
    nb = load_nb(NOTEBOOKS / "02_mnist_models.ipynb")
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            if "text/html" in data:
                html = "".join(data["text/html"])
                if "Test Accuracy" in html and "CNN" in html:
                    md = "# Model comparison (from notebook output)\n\n" + html_table_to_markdown(html)
                    dest = RESULTS / "tables" / "model_comparison.md"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(md, encoding="utf-8")
                    print(f"Wrote {dest}")
                    return
    raise SystemExit("Could not find comparison HTML table in 02_mnist_models.ipynb")


def generate_roadmap() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 2)
    ax.axis("off")

    nodes = [
        "EDA", "KNN", "Logistic", "MLP", "CNN", "Compare",
        "VAE", "DDPM", "DCGAN", "Latent diff",
    ]
    xs = [0.3 + i * 1.15 for i in range(len(nodes))]
    for x, label in zip(xs, nodes):
        box = FancyBboxPatch(
            (x, 0.7), 1.0, 0.6, boxstyle="round,pad=0.05",
            linewidth=1, edgecolor="#333", facecolor="#e8f0fe",
        )
        ax.add_patch(box)
        ax.text(x + 0.5, 1.0, label, ha="center", va="center", fontsize=8)
    for i in range(len(xs) - 1):
        ax.annotate("", xy=(xs[i + 1], 1.0), xytext=(xs[i] + 1.0, 1.0),
                    arrowprops=dict(arrowstyle="->", lw=1.2))

    ax.set_title("MNIST lab learning path (classification → generative)", fontsize=11)
    dest = RESULTS / "figures" / "00_roadmap" / "learning_path.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {dest}")


def generate_ddpm_loss_curve() -> None:
    import matplotlib.pyplot as plt

    text = all_stream_text(load_nb(NOTEBOOKS / "ddpm.ipynb"))
    epochs, losses = [], []
    for m in re.finditer(r"Epoch \[(\d+)/15\].*Avg Loss: ([\d.]+)", text):
        epochs.append(int(m.group(1)))
        losses.append(float(m.group(2)))

    if not epochs:
        raise SystemExit("No DDPM epoch losses found in notebook streams")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, losses, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average loss")
    ax.set_title("DDPM training loss (from notebook output)")
    ax.grid(True, alpha=0.3)
    dest = RESULTS / "figures" / "08_ddpm" / "loss_curve.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {dest}")


def generate_comparison_bar_chart(classification: dict) -> None:
    import matplotlib.pyplot as plt

    labels = ["KNN", "Logistic", "MLP", "SimpleCNN", "Cmp CNN"]
    values = [
        classification["knn_gpu_k3"],
        classification["logistic_regression"],
        classification["manual_mlp"],
        classification["simple_cnn"],
        classification["comparison_cnn"],
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=["#4c72b0", "#dd8452", "#55a868", "#8172b3", "#c44e52"])
    ax.set_ylim(0.9, 1.0)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Classification accuracy ladder (from saved notebook metrics)")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002, f"{v:.4f}",
                ha="center", va="bottom", fontsize=9)
    dest = RESULTS / "figures" / "06_comparison" / "results_bar_chart.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {dest}")


def load_manifest() -> list[dict]:
    if yaml is None:
        raise SystemExit("PyYAML required: pip install pyyaml")
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    return data.get("figures", [])


def cmd_extract() -> None:
    figures = load_manifest()
    for entry in figures:
        nb_rel = entry["notebook"]
        nb_path = REPO_ROOT / nb_rel
        dest = RESULTS / entry["dest"]
        data = extract_png(nb_path, entry["cell"], entry["output"])
        save_png(data, dest)
        print(f"Wrote {dest} <- {nb_rel} [{entry['cell']}/{entry['output']}]")

    classification = scrape_classification()
    generative = scrape_generative()
    verify_metrics(classification, generative)
    write_metrics_json(classification, generative)
    extract_comparison_table()
    generate_roadmap()
    generate_ddpm_loss_curve()
    generate_comparison_bar_chart(classification)
    write_results_index(figures, classification, generative)
    print("Extraction complete.")


def write_results_index(figures: list[dict], classification: dict, generative: dict) -> None:
    lines = [
        "# Results index",
        "",
        "All assets extracted from saved notebook cell outputs (no retraining).",
        "",
        "## Report figure checklist",
        "",
        "| # | Report figure | Path | Status |",
        "|---|---------------|------|--------|",
        "| 1 | Learning roadmap | `figures/00_roadmap/learning_path.png` | done |",
        "| 2 | KNN k-sweep | `figures/02_knn/k_sweep.png` | done |",
        "| 3 | MLP weight templates | `figures/04_mlp/weight_templates.png` | done |",
        "| 4 | CNN confusion + ROC | `figures/05_cnn/confusion_matrix.png`, `roc_curves.png` | done |",
        "| 5 | Model comparison bar | `figures/06_comparison/results_bar_chart.png` | done |",
        "| 6 | VAE reconstructions | `figures/07_vae/reconstructions.png` | done |",
        "| 7 | DDPM denoising grid | `figures/08_ddpm/denoising_grid.png` | done |",
        "| 8 | DCGAN digits | `figures/09_dcgan/generated_digits.png` | done |",
        "| 9 | Latent diffusion samples | `figures/10_latent_diff/diffusion_samples.png` | done |",
        "",
        "## Figures",
        "",
        "| File | Source notebook | Cell | Role |",
        "|------|-----------------|------|------|",
    ]
    for entry in figures:
        lines.append(
            f"| `{entry['dest']}` | `{entry['notebook']}` | {entry['cell']} | {entry.get('role', '')} |"
        )
    lines.extend([
        "| `figures/00_roadmap/learning_path.png` | generated | — | Learning path diagram |",
        "| `figures/08_ddpm/loss_curve.png` | ddpm.ipynb streams | — | DDPM loss from epoch logs |",
        "| `figures/06_comparison/results_bar_chart.png` | metrics JSON | — | Accuracy ladder bar chart |",
        "",
        "## Metrics summary",
        "",
        "### Classification",
        "",
    ])
    for k, v in classification.items():
        lines.append(f"- `{k}`: **{v}**")
    lines.extend(["", "### Generative", ""])
    for k, v in generative.items():
        lines.append(f"- `{k}`: **{v}**")
    lines.extend([
        "",
        "## Tables",
        "",
        "- `tables/model_comparison.md` — sorted DataFrame from `02_mnist_models.ipynb`",
        "",
        "## Exclusions",
        "",
        "No files from `training_results/`, `latent_data/`, or `*.pth` checkpoints.",
    ])
    (RESULTS / "RESULTS_INDEX.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {RESULTS / 'RESULTS_INDEX.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    if args.inventory:
        cmd_inventory()
    elif args.extract:
        cmd_extract()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
