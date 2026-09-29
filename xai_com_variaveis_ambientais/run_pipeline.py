"""Executa, de ponta a ponta, o exemplo sintético da X-TCN para E-nose."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from xtcn.constants import CLASS_NAMES, FEATURE_NAMES, SENSOR_NAMES
from xtcn.dataset import FingerprintDataset
from xtcn.features import extract_features
from xtcn.fingerprint import FeatureStandardizer, build_fingerprint_map
from xtcn.gradcam import GradCAM
from xtcn.model import XTCN
from xtcn.preprocessing import relative_conductance
from xtcn.synthetic import generate_synthetic_enose, stratified_split
from xtcn.training import evaluate_model, train_model
from xtcn.visualization import save_confusion_matrix, save_gradcam_explanation, save_training_history


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--samples-per-class", type=int, default=24)
    parser.add_argument("--time-steps", type=int, default=256)
    parser.add_argument("--sampling-rate", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "results")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def select_device(option: str) -> torch.device:
    if option == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA foi solicitada, mas nao esta disponivel.")
    if option == "auto":
        option = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(option)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dispositivo: {device}")
    conductance, labels = generate_synthetic_enose(
        samples_per_class=args.samples_per_class,
        time_steps=args.time_steps,
        sampling_rate=args.sampling_rate,
        seed=args.seed,
    )
    relative, _g0 = relative_conductance(conductance, baseline_samples=max(5, args.time_steps // 16))
    features = extract_features(relative, sampling_rate=args.sampling_rate)
    train_idx, val_idx, test_idx = stratified_split(labels, seed=args.seed)

    standardizer = FeatureStandardizer().fit(features[train_idx])
    standardized = standardizer.transform(features)
    fingerprints = build_fingerprint_map(standardized, size=224)

    datasets = {
        "train": FingerprintDataset(fingerprints[train_idx], labels[train_idx]),
        "val": FingerprintDataset(fingerprints[val_idx], labels[val_idx]),
        "test": FingerprintDataset(fingerprints[test_idx], labels[test_idx]),
    }
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, generator=generator),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False),
    }

    model = XTCN(num_classes=len(CLASS_NAMES))
    history = train_model(
        model,
        loaders["train"],
        loaders["val"],
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
    )
    test_accuracy, confusion = evaluate_model(model, loaders["test"], device)
    print(f"Acuracia no teste sintetico: {test_accuracy:.3f}")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        # Tensores mantêm o checkpoint compatível com torch.load(weights_only=True).
        "feature_mean": torch.from_numpy(standardizer.mean_.copy()),
        "feature_scale": torch.from_numpy(standardizer.scale_.copy()),
        "sensor_names": SENSOR_NAMES,
        "feature_names": FEATURE_NAMES,
        "class_names": CLASS_NAMES,
        "sampling_rate": args.sampling_rate,
    }
    torch.save(checkpoint, args.output_dir / "xtcn_checkpoint.pt")
    save_training_history(history, args.output_dir / "historico_treino.png")
    save_confusion_matrix(confusion.numpy(), args.output_dir / "matriz_confusao.png")

    sample, true_label = datasets["test"][0]
    sample_batch = sample.unsqueeze(0).to(device)
    with GradCAM(model, model.gradcam_target_layer) as gradcam:
        heatmap, logits = gradcam(sample_batch)
    predicted = int(logits.argmax(dim=1).item())
    importance = save_gradcam_explanation(
        sample.numpy(),
        heatmap[0, 0].cpu().numpy(),
        int(true_label.item()),
        predicted,
        args.output_dir / "gradcam_explicacao.png",
    )
    with (args.output_dir / "relevancia_sensor_caracteristica.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sensor", *FEATURE_NAMES])
        for sensor, row in zip(SENSOR_NAMES, importance):
            writer.writerow([sensor, *[f"{value:.7f}" for value in row]])
    metrics = {
        "test_accuracy": test_accuracy,
        "train_samples": len(train_idx),
        "validation_samples": len(val_idx),
        "test_samples": len(test_idx),
        "true_class": int(true_label.item()),
        "predicted_class": predicted,
        "device": str(device),
    }
    (args.output_dir / "metricas.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Artefactos guardados em: {args.output_dir}")


if __name__ == "__main__":
    main()

