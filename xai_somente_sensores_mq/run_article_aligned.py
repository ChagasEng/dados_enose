"""Compara a X-TCN alinhada ao artigo em dados reais do E-nose.

Executa (1) avaliação principal com coletas disjuntas e (2) diagnóstico com
split aleatório 80/20 de fingerprints, como descrito no texto do artigo.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from run_real_data import (
    CLASS_NAMES,
    DEFAULT_DATASET,
    SENSOR_LABELS,
    SENSORS,
    load_real_data,
    make_windows,
    metric_bundle,
    seed_everything,
    select_device,
    split_collection_groups,
)
from xtcn.article_features import (
    ArticleMinMaxScaler,
    build_article_fingerprints,
    extract_article_features,
    heatmap_to_cells,
)
from xtcn.article_model import ArticleAlignedXTCN
from xtcn.gradcam import GradCAM


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DEFAULT = PROJECT_DIR / "resultados_artigo_alinhado"
FEATURE_NAMES = ("AUC", "ES", "PSD", "PW", "Var", "PP", "DFC", "FTF")
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)


class ArticleImageDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index].float().div_(255.0)
        return (image - IMAGENET_MEAN) / IMAGENET_STD, self.labels[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--random-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--baseline-samples", type=int, default=20)
    parser.add_argument("--sampling-rate", type=float, default=4.0)
    parser.add_argument("--dominant-count", type=int, default=3)
    parser.add_argument("--fourier-bins", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def to_uint8_maps(features: np.ndarray, chunk_size: int = 64) -> np.ndarray:
    output = np.empty((len(features), 3, 224, 224), dtype=np.uint8)
    for start in range(0, len(features), chunk_size):
        maps = build_article_fingerprints(features[start : start + chunk_size])
        output[start : start + len(maps)] = np.round(maps * 255).astype(np.uint8)
    return output


def make_loader(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        ArticleImageDataset(images, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * len(targets)
            correct += int((logits.argmax(1) == targets).sum().item())
            count += len(targets)
    return total_loss / count, correct / count


def fit_with_validation(
    model: ArticleAlignedXTCN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> tuple[pd.DataFrame, int]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    rows = []
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device, None)
        rows.append(
            {
                "epoca": epoch,
                "loss_treino": train_loss,
                "acuracia_treino": train_accuracy,
                "loss_validacao": val_loss,
                "acuracia_validacao": val_accuracy,
            }
        )
        if val_accuracy > best_accuracy or (
            np.isclose(val_accuracy, best_accuracy) and val_loss < best_loss
        ):
            best_accuracy = val_accuracy
            best_loss = val_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        print(
            f"[agrupado] {epoch:02d}/{epochs} "
            f"train={train_accuracy:.3f} val={val_accuracy:.3f} "
            f"loss_val={val_loss:.4f}"
        )
    if best_state is None:
        raise RuntimeError("Nenhum estado de treino foi produzido.")
    model.load_state_dict(best_state)
    return pd.DataFrame(rows), best_epoch


def fit_fixed(
    model: ArticleAlignedXTCN,
    loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    label: str,
) -> None:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        loss, accuracy = run_epoch(model, loader, criterion, device, optimizer)
        print(f"[{label}] {epoch:02d}/{epochs} accuracy={accuracy:.3f} loss={loss:.4f}")


def predict(
    model: ArticleAlignedXTCN,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = []
    labels = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            probabilities.append(torch.softmax(model(inputs.to(device)), 1).cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def save_history(history: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.epoca, history.loss_treino, label="Treino")
    axes[0].plot(history.epoca, history.loss_validacao, label="Validação")
    axes[0].set(xlabel="Época", ylabel="Loss", title="Cross-entropy")
    axes[1].plot(history.epoca, history.acuracia_treino, label="Treino")
    axes[1].plot(history.epoca, history.acuracia_validacao, label="Validação")
    axes[1].set(xlabel="Época", ylabel="Acurácia", title="Seleção agrupada")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_confusion(matrix: np.ndarray, path: Path, title: str) -> None:
    fig, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(matrix, cmap="Blues")
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(column, row, str(value), ha="center", va="center", fontsize=13)
    axis.set_xticks([0, 1], CLASS_NAMES, rotation=15, ha="right")
    axis.set_yticks([0, 1], CLASS_NAMES)
    axis.set(xlabel="Previsto", ylabel="Real", title=title)
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def grouped_predictions(
    probabilities: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "coleta": groups,
            "classe_real": labels,
            "probabilidade_com_nematoide": probabilities[:, 0],
            "probabilidade_saudavel": probabilities[:, 1],
        }
    )
    result = frame.groupby("coleta", as_index=False).agg(
        classe_real=("classe_real", "first"),
        probabilidade_com_nematoide=("probabilidade_com_nematoide", "mean"),
        probabilidade_saudavel=("probabilidade_saudavel", "mean"),
        numero_janelas=("classe_real", "size"),
    )
    result["classe_prevista"] = result[
        ["probabilidade_com_nematoide", "probabilidade_saudavel"]
    ].to_numpy().argmax(axis=1)
    return result


def gradcam_summary(
    model: ArticleAlignedXTCN,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = []
    with GradCAM(model, model.gradcam_target_layer) as gradcam:
        for inputs, _targets in loader:
            heatmaps, _ = gradcam(inputs.to(device))
            cells.extend(heatmap_to_cells(item) for item in heatmaps[:, 0].cpu().numpy())
    mean_cells = np.mean(cells, axis=0)
    percentages = mean_cells / max(float(mean_cells.sum()), 1e-12) * 100.0
    table = pd.DataFrame(percentages, index=FEATURE_NAMES, columns=SENSOR_LABELS)
    table.index.name = "caracteristica"
    table.to_csv(output_dir / "gradcam_sensor_caracteristica_percentual.csv")
    sensor_table = pd.DataFrame(
        {
            "sensor": SENSOR_LABELS,
            "importancia_percentual": percentages.sum(axis=0),
        }
    ).sort_values("importancia_percentual", ascending=False)
    sensor_table.to_csv(output_dir / "importancia_sensores_gradcam.csv", index=False)
    feature_table = pd.DataFrame(
        {
            "caracteristica": FEATURE_NAMES,
            "importancia_percentual": percentages.sum(axis=1),
        }
    ).sort_values("importancia_percentual", ascending=False)
    feature_table.to_csv(
        output_dir / "importancia_caracteristicas_gradcam.csv", index=False
    )
    fig, axis = plt.subplots(figsize=(9, 6))
    image = axis.imshow(percentages, cmap="magma")
    axis.set_xticks(range(6), SENSOR_LABELS, rotation=25, ha="right")
    axis.set_yticks(range(8), FEATURE_NAMES)
    axis.set_title("Grad-CAM médio — X-TCN alinhada ao artigo (%)")
    for (row, column), value in np.ndenumerate(percentages):
        axis.text(column, row, f"{value:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis, label="Importância percentual")
    fig.tight_layout()
    fig.savefig(output_dir / "gradcam_medio.png", dpi=180)
    plt.close(fig)
    return sensor_table, feature_table


def metrics_lines(metrics: dict[str, float]) -> list[str]:
    return [f"  {key}: {value:.6f} ({value * 100:.2f}%)" for key, value in metrics.items()]


def write_report(
    path: Path,
    args: argparse.Namespace,
    model_parameters: int,
    best_epoch: int,
    split: pd.DataFrame,
    counts: dict[str, int],
    grouped_window_metrics: dict[str, float],
    collection_metrics: dict[str, float],
    grouped_matrix: np.ndarray,
    class_report: dict,
    random_metrics: dict[str, float],
    random_matrix: np.ndarray,
    shared_collections: int,
    sensor_table: pd.DataFrame,
    feature_table: pd.DataFrame,
) -> None:
    lines = [
        "TESTE REAL — X-TCN ALINHADA AO ARTIGO",
        "=" * 68,
        f"Gerado em: {datetime.now().isoformat(timespec='seconds')}",
        f"Dataset: {args.dataset.resolve()}",
        f"Parametros treinaveis: {model_parameters}",
        "Sensores: " + ", ".join(SENSOR_LABELS),
        "Classes: 0=com nematoide; 1=saudavel/sem nematoide.",
        "",
        "ALINHAMENTO IMPLEMENTADO",
        "-" * 68,
        "Backbone de blocos invertidos, SE interno, Hardswish, saída 576 canais,",
        "CBAM com reducao 16, classificador 576->1024->2 e dropout 0.2.",
        "Fingerprint 224x224 RGB crest, descritores nas linhas e sensores nas colunas.",
        "Min-Max por sensor/caracteristica ajustado somente no treino.",
        "Normalizacao ImageNet, Adam, CrossEntropyLoss e Grad-CAM na convolucao final.",
        f"Janela={args.window}; stride={args.stride}; G0={args.baseline_samples} amostras; taxa={args.sampling_rate} Hz.",
        f"DFC: {args.dominant_count} componentes; FTF: {args.fourier_bins} bins.",
        "",
        "LIMITACAO DE REPRODUCAO",
        "-" * 68,
        "O suplemento define ES, PSD, DFC e FTF como vetores, mas o fingerprint reserva",
        "uma celula por descritor e o repositorio nao publica o codigo que os reduz a",
        "escalares. Foram usadas reducoes explicitamente documentadas no codigo.",
        "",
        "1) RESULTADO PRINCIPAL — SPLIT POR COLETA",
        "-" * 68,
        "Nenhuma coleta aparece em mais de uma particao.",
        f"Melhor epoca escolhida na validacao: {best_epoch}/{args.epochs}.",
    ]
    for partition in ("treino", "validacao", "teste"):
        subset = split[split.particao == partition]
        lines.append(
            f"{partition.capitalize()}: {len(subset)} coletas, {counts[partition]} janelas."
        )
    lines.extend(["", "Metricas por janela:", *metrics_lines(grouped_window_metrics)])
    lines.extend(["", "Metricas por coleta:", *metrics_lines(collection_metrics)])
    lines.extend(
        [
            "",
            "Matriz por coleta [linhas=reais, colunas=previstas]:",
            f"  [[{grouped_matrix[0,0]}, {grouped_matrix[0,1]}],",
            f"   [{grouped_matrix[1,0]}, {grouped_matrix[1,1]}]]",
            "",
            "Por classe:",
        ]
    )
    for class_id, name in enumerate(CLASS_NAMES):
        item = class_report[str(class_id)]
        lines.append(
            f"  {class_id} {name}: precision={item['precision']:.4f}, "
            f"recall={item['recall']:.4f}, f1={item['f1-score']:.4f}, "
            f"suporte={int(item['support'])}"
        )
    lines.extend(
        [
            "",
            "2) DIAGNOSTICO — SPLIT ALEATORIO 80/20 DE JANELAS",
            "-" * 68,
            f"Epocas: {args.random_epochs}; coletas presentes simultaneamente nos dois lados: {shared_collections}.",
            "Este resultado se aproxima da divisao textual do artigo, mas NAO e uma",
            "estimativa valida de generalizacao no nosso conjunto: janelas sobrepostas da",
            "mesma coleta podem ficar no treino e no teste.",
            *metrics_lines(random_metrics),
            "Matriz por janela:",
            f"  [[{random_matrix[0,0]}, {random_matrix[0,1]}],",
            f"   [{random_matrix[1,0]}, {random_matrix[1,1]}]]",
            "",
            "IMPORTANCIA GRAD-CAM — TESTE AGRUPADO",
            "-" * 68,
            "Sensores:",
        ]
    )
    lines.extend(
        f"  {row.sensor}: {row.importancia_percentual:.3f}%"
        for row in sensor_table.itertuples()
    )
    lines.append("Caracteristicas:")
    lines.extend(
        f"  {row.caracteristica}: {row.importancia_percentual:.3f}%"
        for row in feature_table.itertuples()
    )
    compact_path = PROJECT_DIR / "resultados_reais" / "metricas_reais.json"
    temporal_path = PROJECT_DIR.parent / "x_tcn" / "resultados" / "resumo_avaliacao_x_tcn.json"
    compact_accuracy = None
    temporal_accuracy = None
    if compact_path.exists():
        compact_accuracy = json.loads(compact_path.read_text(encoding="utf-8"))[
            "metricas_por_coleta"
        ]["accuracy"]
    if temporal_path.exists():
        temporal_accuracy = json.loads(temporal_path.read_text(encoding="utf-8"))[
            "metricas_por_coleta"
        ]["accuracy"]
    lines.extend(["", "COMPARACAO DE ACURACIA", "-" * 68])
    if temporal_accuracy is not None:
        lines.append(f"X-TCN temporal 1D anterior, por coleta: {temporal_accuracy * 100:.2f}%")
    if compact_accuracy is not None:
        lines.append(f"X-TCN fingerprint compacta anterior, por coleta: {compact_accuracy * 100:.2f}%")
    lines.extend(
        [
            f"X-TCN alinhada, split por coleta: {collection_metrics['accuracy'] * 100:.2f}%",
            f"X-TCN alinhada, split aleatorio de janelas: {random_metrics['accuracy'] * 100:.2f}%",
            "X-TCN do artigo, amostras controladas de insetos: 99.28%",
            "Os 99.28% nao sao diretamente comparaveis aos nossos dados de solo-soja.",
        ]
    )
    lines.extend(
        [
            "",
            "INTERPRETACAO",
            "-" * 68,
            "A metrica cientifica principal e a acuracia por coleta do protocolo 1.",
            "O protocolo 2 mede o quanto o modelo reconhece janelas parecidas das mesmas",
            "coletas e tende a ser otimista. A arquitetura alinhada nao elimina a diferenca",
            "entre 36 coletas reais e as 700 amostras controladas do artigo.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_real_data(args.dataset)
    train_df, val_df, test_df, split = split_collection_groups(data, args.seed)
    split.to_csv(args.output_dir / "split_coletas_agrupado.csv", index=False)

    frames = {"treino": train_df, "validacao": val_df, "teste": test_df}
    windows = {
        name: make_windows(frame, args.window, args.stride, args.baseline_samples)
        for name, frame in frames.items()
    }
    features = {
        name: extract_article_features(
            item[0], args.sampling_rate, args.dominant_count, args.fourier_bins
        )
        for name, item in windows.items()
    }
    print(
        "Particoes agrupadas: "
        + ", ".join(
            f"{name}={frames[name].Coleta.nunique()} coletas/{len(windows[name][1])} janelas"
            for name in frames
        )
    )
    grouped_scaler = ArticleMinMaxScaler().fit(features["treino"])
    grouped_maps = {
        name: to_uint8_maps(grouped_scaler.transform(item))
        for name, item in features.items()
    }
    grouped_loaders = {
        name: make_loader(
            grouped_maps[name],
            windows[name][1],
            args.batch_size,
            name == "treino",
            args.seed,
        )
        for name in frames
    }
    grouped_model = ArticleAlignedXTCN(num_classes=2)
    parameters = sum(parameter.numel() for parameter in grouped_model.parameters())
    print(f"Modelo alinhado: {parameters} parametros; dispositivo={device}")
    history, best_epoch = fit_with_validation(
        grouped_model,
        grouped_loaders["treino"],
        grouped_loaders["validacao"],
        args.epochs,
        args.learning_rate,
        device,
    )
    history.to_csv(args.output_dir / "historico_treino_agrupado.csv", index=False)
    save_history(history, args.output_dir / "historico_treino_agrupado.png")
    grouped_probabilities, grouped_labels = predict(
        grouped_model, grouped_loaders["teste"], device
    )
    grouped_window_predictions = grouped_probabilities.argmax(1)
    grouped_window_metrics = metric_bundle(
        grouped_labels, grouped_window_predictions
    )
    collection_predictions = grouped_predictions(
        grouped_probabilities, grouped_labels, windows["teste"][2]
    )
    collection_predictions.to_csv(
        args.output_dir / "predicoes_por_coleta_agrupado.csv", index=False
    )
    collection_true = collection_predictions.classe_real.to_numpy()
    collection_predicted = collection_predictions.classe_prevista.to_numpy()
    collection_metrics = metric_bundle(collection_true, collection_predicted)
    grouped_matrix = confusion_matrix(
        collection_true, collection_predicted, labels=[0, 1]
    )
    pd.DataFrame(
        grouped_matrix,
        index=["real_0_com_nematoide", "real_1_saudavel"],
        columns=["previsto_0_com_nematoide", "previsto_1_saudavel"],
    ).to_csv(args.output_dir / "matriz_confusao_agrupado.csv")
    save_confusion(
        grouped_matrix,
        args.output_dir / "matriz_confusao_agrupado.png",
        "X-TCN alinhada — teste por coleta",
    )
    class_report = classification_report(
        collection_true,
        collection_predicted,
        labels=[0, 1],
        output_dict=True,
        zero_division=0,
    )
    sensor_table, feature_table = gradcam_summary(
        grouped_model, grouped_loaders["teste"], device, args.output_dir
    )

    # Diagnóstico do protocolo textual do artigo: split aleatório dos fingerprints.
    all_signals, all_labels, all_groups = make_windows(
        data, args.window, args.stride, args.baseline_samples
    )
    all_features = extract_article_features(
        all_signals, args.sampling_rate, args.dominant_count, args.fourier_bins
    )
    all_indices = np.arange(len(all_labels))
    random_train_idx, random_test_idx = train_test_split(
        all_indices,
        test_size=0.20,
        random_state=args.seed,
        stratify=all_labels,
    )
    random_scaler = ArticleMinMaxScaler().fit(all_features[random_train_idx])
    random_train_maps = to_uint8_maps(
        random_scaler.transform(all_features[random_train_idx])
    )
    random_test_maps = to_uint8_maps(
        random_scaler.transform(all_features[random_test_idx])
    )
    random_train_loader = make_loader(
        random_train_maps,
        all_labels[random_train_idx],
        args.batch_size,
        True,
        args.seed,
    )
    random_test_loader = make_loader(
        random_test_maps,
        all_labels[random_test_idx],
        args.batch_size,
        False,
        args.seed,
    )
    seed_everything(args.seed)
    random_model = ArticleAlignedXTCN(num_classes=2)
    fit_fixed(
        random_model,
        random_train_loader,
        args.random_epochs,
        args.learning_rate,
        device,
        "aleatorio80_20",
    )
    random_probabilities, random_labels = predict(
        random_model, random_test_loader, device
    )
    random_predictions = random_probabilities.argmax(1)
    random_metrics = metric_bundle(random_labels, random_predictions)
    random_matrix = confusion_matrix(random_labels, random_predictions, labels=[0, 1])
    save_confusion(
        random_matrix,
        args.output_dir / "matriz_confusao_aleatorio_80_20.png",
        "X-TCN alinhada — split aleatório 80/20",
    )
    shared_collections = len(
        set(all_groups[random_train_idx]) & set(all_groups[random_test_idx])
    )

    summary = {
        "modelo": "X-TCN alinhada ao artigo Chen et al. (2026)",
        "doi": "10.1016/j.snb.2026.140724",
        "parametros": parameters,
        "classes_reais": {"0": CLASS_NAMES[0], "1": CLASS_NAMES[1]},
        "configuracao": {
            "epochs_grouped": args.epochs,
            "best_epoch_grouped": best_epoch,
            "epochs_random_80_20": args.random_epochs,
            "batch_size": args.batch_size,
            "window": args.window,
            "stride": args.stride,
            "baseline_samples": args.baseline_samples,
            "sampling_rate": args.sampling_rate,
            "dominant_count": args.dominant_count,
            "fourier_bins": args.fourier_bins,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "device": str(device),
        },
        "principal_split_por_coleta": {
            "metricas_por_janela": grouped_window_metrics,
            "metricas_por_coleta": collection_metrics,
            "matriz_confusao_por_coleta": grouped_matrix.tolist(),
            "coletas_teste": int(test_df.Coleta.nunique()),
        },
        "diagnostico_split_aleatorio_80_20": {
            "metricas_por_janela": random_metrics,
            "matriz_confusao": random_matrix.tolist(),
            "coletas_compartilhadas": shared_collections,
            "aviso": "Nao representa generalizacao: ha janelas da mesma coleta nos dois lados.",
        },
        "limitacao": (
            "O repositorio oficial nao publica a reducao vetorial para escalar de "
            "ES, PSD, DFC e FTF; as reducoes usadas estao documentadas no codigo."
        ),
    }
    (args.output_dir / "metricas_comparativas.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.save(
        {
            "model_state_dict": grouped_model.state_dict(),
            "minima": torch.from_numpy(grouped_scaler.minimum_.copy()),
            "spans": torch.from_numpy(grouped_scaler.span_.copy()),
            "sensors": SENSORS,
            "features": FEATURE_NAMES,
            "best_epoch": best_epoch,
            "parameters": parameters,
        },
        args.output_dir / "xtcn_artigo_alinhado_checkpoint.pt",
    )
    write_report(
        args.output_dir / "RELATORIO_TESTE_ARTIGO_ALINHADO.txt",
        args,
        parameters,
        best_epoch,
        split,
        {name: len(windows[name][1]) for name in frames},
        grouped_window_metrics,
        collection_metrics,
        grouped_matrix,
        class_report,
        random_metrics,
        random_matrix,
        shared_collections,
        sensor_table,
        feature_table,
    )
    print(
        json.dumps(
            {
                "principal_por_coleta": collection_metrics,
                "principal_por_janela": grouped_window_metrics,
                "diagnostico_aleatorio_80_20": random_metrics,
                "coletas_compartilhadas_no_diagnostico": shared_collections,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Relatorio: {args.output_dir / 'RELATORIO_TESTE_ARTIGO_ALINHADO.txt'}")


if __name__ == "__main__":
    main()

