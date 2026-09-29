"""Treina e explica a X-TCN 2D nos dados reais, usando somente sensores MQ.

O conjunto de teste é separado por coleta antes de criar janelas. Assim,
janelas da mesma unidade experimental nunca aparecem em treino e teste.
"""

from __future__ import annotations

import argparse
import csv
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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from torch import nn

from xtcn.features import extract_features
from xtcn.fingerprint import FeatureStandardizer, aggregate_heatmap_by_cell, build_fingerprint_map
from xtcn.gradcam import GradCAM
from xtcn.model import XTCN
from xtcn.preprocessing import relative_conductance
from xtcn.training import History, train_model
from xtcn.visualization import save_training_history


PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent
DEFAULT_DATASET = (
    ROOT_DIR
    / "06_07"
    / "4_polimento_inicial_modelagem"
    / "datasets_limpos"
    / "antes_dia_20_pressao_filtrada_estrito.csv"
)
SENSORS = ("MQ2", "MQ3", "MQ7", "MQ8", "MQ135", "MQ138")
SENSOR_LABELS = ("MQ-2", "MQ-3", "MQ-7", "MQ-8", "MQ-135", "MQ-138")
FEATURES = ("AUC", "ES", "PSD", "PW", "Var", "PP", "DFC", "FTF")
CLASS_NAMES = ("Com nematoide", "Saudavel/sem nematoide")


class Uint8FingerprintDataset(Dataset):
    """Armazena fingerprints em uint8 e converte para float sob demanda."""

    def __init__(self, maps: np.ndarray, labels: np.ndarray) -> None:
        self.maps = torch.from_numpy(maps)
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.maps[index].float().div_(255.0), self.labels[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "resultados_reais")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--baseline-samples", type=int, default=20)
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=4.0,
        help="Taxa em Hz; Tempo avanca aproximadamente 0,25 s na base real.",
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(option: str) -> torch.device:
    if option == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA solicitada, mas indisponivel.")
    return torch.device("cuda" if option == "auto" and torch.cuda.is_available() else ("cpu" if option == "auto" else option))


def load_real_data(path: Path) -> pd.DataFrame:
    required = ["Coleta", "Tempo", "Classe", *SENSORS]
    data = pd.read_csv(path)
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")
    data = data[required].copy()
    for column in ["Tempo", "Classe", *SENSORS]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=required).copy()
    data["Classe"] = data["Classe"].astype(int)
    observed = sorted(data["Classe"].unique().tolist())
    if observed != [0, 1]:
        raise ValueError(f"A base real deveria conter classes [0, 1], mas contem {observed}.")
    return data


def split_collection_groups(data: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Separa 70% das coletas para treino/validação e 30% para teste.

    Dentro dos 70%, aproximadamente 20% das coletas formam a validação. A
    amostragem reproduz o protocolo histórico do projeto para o teste.
    """
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    records: list[dict[str, object]] = []
    for class_id, block in data.groupby("Classe", sort=True):
        groups = pd.Series(block["Coleta"].unique()).sample(frac=1, random_state=seed).tolist()
        n_trainval = int(len(groups) * 0.70)
        trainval_groups = groups[:n_trainval]
        test_groups = groups[n_trainval:]
        shuffled_trainval = (
            pd.Series(trainval_groups).sample(frac=1, random_state=seed + 101).tolist()
        )
        n_val = max(1, int(round(len(shuffled_trainval) * 0.20)))
        val_groups = set(shuffled_trainval[:n_val])
        train_groups = set(shuffled_trainval[n_val:])
        test_groups_set = set(test_groups)
        train_parts.append(block[block["Coleta"].isin(train_groups)])
        val_parts.append(block[block["Coleta"].isin(val_groups)])
        test_parts.append(block[block["Coleta"].isin(test_groups_set)])
        for group in groups:
            partition = "treino" if group in train_groups else ("validacao" if group in val_groups else "teste")
            records.append({"coleta": group, "classe": int(class_id), "particao": partition})
    split = pd.DataFrame(records).sort_values(["particao", "classe", "coleta"])
    return pd.concat(train_parts), pd.concat(val_parts), pd.concat(test_parts), split


def make_windows(
    data: pd.DataFrame,
    window: int,
    stride: int,
    baseline_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signals: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    for group, block in data.sort_values(["Coleta", "Tempo"]).groupby("Coleta", sort=False):
        raw_values = block[list(SENSORS)].to_numpy(dtype=np.float32)
        # G0 é único por coleta, não por janela sobreposta.
        relative_values, _ = relative_conductance(
            raw_values.T, baseline_samples=baseline_samples
        )
        values = relative_values.T
        label_values = block["Classe"].unique()
        if len(label_values) != 1:
            raise ValueError(f"A coleta {group!r} possui mais de uma classe.")
        if len(values) < window:
            starts = [0]
        else:
            starts = list(range(0, len(values) - window + 1, stride))
            if starts[-1] != len(values) - window:
                starts.append(len(values) - window)
        for start in starts:
            chunk = values[start : start + window]
            if len(chunk) < window:
                chunk = np.pad(chunk, ((0, window - len(chunk)), (0, 0)), mode="edge")
            signals.append(chunk.T)
            labels.append(int(label_values[0]))
            groups.append(str(group))
    return np.stack(signals), np.asarray(labels, dtype=np.int64), np.asarray(groups)


def signals_to_features(
    signals: np.ndarray,
    sampling_rate: float,
) -> np.ndarray:
    return extract_features(signals, sampling_rate=sampling_rate)


def to_uint8_maps(standardized: np.ndarray, chunk_size: int = 64) -> np.ndarray:
    maps = np.empty((len(standardized), 3, 224, 224), dtype=np.uint8)
    for start in range(0, len(standardized), chunk_size):
        batch = build_fingerprint_map(standardized[start : start + chunk_size], size=224)
        maps[start : start + len(batch)] = np.round(batch * 255.0).astype(np.uint8)
    return maps


def predict(
    model: XTCN,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            probabilities.append(torch.softmax(model(inputs.to(device)), dim=1).cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def fit_fixed_epochs(
    model: XTCN,
    loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> None:
    """Retreina o modelo final no desenvolvimento completo pelo prazo escolhido."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    for _ in range(epochs):
        model.train()
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def save_confusion(confusion: np.ndarray, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(confusion, cmap="Blues")
    for (row, col), value in np.ndenumerate(confusion):
        axis.text(col, row, str(value), ha="center", va="center", fontsize=13)
    axis.set_xticks([0, 1], CLASS_NAMES, rotation=15, ha="right")
    axis.set_yticks([0, 1], CLASS_NAMES)
    axis.set(xlabel="Classe prevista", ylabel="Classe real", title="X-TCN real — matriz por coleta")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def gradcam_importance(
    model: XTCN,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    all_cells: list[np.ndarray] = []
    by_class: dict[int, list[np.ndarray]] = {0: [], 1: []}
    with GradCAM(model, model.gradcam_target_layer) as gradcam:
        for inputs, targets in loader:
            heatmaps, _ = gradcam(inputs.to(device))
            for heatmap, target in zip(heatmaps[:, 0].cpu().numpy(), targets.numpy()):
                cells = aggregate_heatmap_by_cell(heatmap)
                all_cells.append(cells)
                by_class[int(target)].append(cells)
    global_mean = np.mean(all_cells, axis=0)
    class_means = {key: np.mean(value, axis=0) for key, value in by_class.items()}
    return global_mean, class_means


def normalize_importance(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    return values / total * 100.0 if total > 0 else np.zeros_like(values)


def save_importance_outputs(
    global_cells: np.ndarray,
    class_cells: dict[int, np.ndarray],
    output_dir: Path,
) -> tuple[np.ndarray, list[tuple[str, str, float]]]:
    global_percent = normalize_importance(global_cells)
    table = pd.DataFrame(global_percent, index=SENSOR_LABELS, columns=FEATURES)
    table.index.name = "sensor"
    table.to_csv(output_dir / "gradcam_sensor_caracteristica_percentual.csv")

    class_rows = []
    for class_id, cells in class_cells.items():
        percentages = normalize_importance(cells)
        for sensor_idx, sensor in enumerate(SENSOR_LABELS):
            row = {"classe": class_id, "nome_classe": CLASS_NAMES[class_id], "sensor": sensor}
            row.update({feature: float(percentages[sensor_idx, feature_idx]) for feature_idx, feature in enumerate(FEATURES)})
            class_rows.append(row)
    pd.DataFrame(class_rows).to_csv(output_dir / "gradcam_por_classe_percentual.csv", index=False)

    sensor_importance = global_percent.sum(axis=1)
    pd.DataFrame({"sensor": SENSOR_LABELS, "importancia_percentual": sensor_importance}).sort_values(
        "importancia_percentual", ascending=False
    ).to_csv(output_dir / "importancia_sensores_gradcam.csv", index=False)
    feature_importance = global_percent.sum(axis=0)
    pd.DataFrame({"caracteristica": FEATURES, "importancia_percentual": feature_importance}).sort_values(
        "importancia_percentual", ascending=False
    ).to_csv(output_dir / "importancia_caracteristicas_gradcam.csv", index=False)

    ranking = [
        (SENSOR_LABELS[row], FEATURES[col], float(global_percent[row, col]))
        for row in range(6)
        for col in range(8)
    ]
    ranking.sort(key=lambda item: item[2], reverse=True)

    fig, axis = plt.subplots(figsize=(10, 6))
    image = axis.imshow(global_percent, cmap="magma")
    axis.set_xticks(range(8), FEATURES, rotation=45, ha="right")
    axis.set_yticks(range(6), SENSOR_LABELS)
    axis.set_title("Grad-CAM médio no teste real (%)")
    for (row, col), value in np.ndenumerate(global_percent):
        axis.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=8, color="white" if value < global_percent.max() * 0.65 else "black")
    fig.colorbar(image, ax=axis, label="Importância percentual")
    fig.tight_layout()
    fig.savefig(output_dir / "gradcam_medio_teste_real.png", dpi=180)
    plt.close(fig)
    return global_percent, ranking


def format_metrics(title: str, metrics: dict[str, float]) -> list[str]:
    return [title, *[f"  {name}: {value:.6f} ({value * 100:.2f}%)" for name, value in metrics.items()]]


def write_report(
    path: Path,
    args: argparse.Namespace,
    data: pd.DataFrame,
    split: pd.DataFrame,
    window_counts: dict[str, int],
    history: History,
    window_metrics: dict[str, float],
    collection_metrics: dict[str, float],
    confusion: np.ndarray,
    report: dict,
    ranking: list[tuple[str, str, float]],
    sensor_percent: np.ndarray,
) -> None:
    selected_epoch = int(np.argmax(history.val_accuracy)) + 1
    lines = [
        "RELATORIO COMPLETO — X-TCN/XAI COM DADOS REAIS",
        "=" * 62,
        f"Gerado em: {datetime.now().isoformat(timespec='seconds')}",
        f"Dataset: {args.dataset.resolve()}",
        "Escopo: somente os seis sensores MQ; sem variaveis ambientais como entrada.",
        f"Sensores: {', '.join(SENSOR_LABELS)}",
        "Classes reais disponiveis: 0 = com nematoide; 1 = saudavel/sem nematoide.",
        "IMPORTANTE: a base nao possui rotulos dos seis tratamentos solicitados; por isso,",
        "esta avaliacao real e binaria. Nenhum rotulo foi inferido ou fabricado.",
        "",
        "PROTOCOLO",
        "-" * 62,
        "Split estratificado por coleta: aproximadamente 70% desenvolvimento e 30% teste.",
        "A validacao foi retirada apenas do desenvolvimento. Nao ha coleta compartilhada",
        "entre treino, validacao e teste.",
        f"Janela: {args.window}; stride: {args.stride}; baseline G0: {args.baseline_samples} amostras.",
        f"Epocas solicitadas para selecao: {args.epochs}; epoca escolhida pela maior acuracia de validacao: {selected_epoch}.",
        "Apos a escolha, o modelo final foi reinicializado e retreinado nas 24 coletas",
        "de desenvolvimento (treino + validacao) pelo numero de epocas escolhido.",
        f"Seed: {args.seed}; dispositivo: {args.device}.",
        "",
        "DADOS E PARTICOES",
        "-" * 62,
        f"Linhas validas totais: {len(data)}",
    ]
    for partition in ("treino", "validacao", "teste"):
        groups = split.loc[split["particao"] == partition]
        class_counts = groups.groupby("classe").size().to_dict()
        lines.append(
            f"{partition.capitalize()}: {len(groups)} coletas "
            f"(classe 0={class_counts.get(0, 0)}, classe 1={class_counts.get(1, 0)}), "
            f"{window_counts[partition]} janelas"
        )
    lines.extend(["", *format_metrics("METRICAS POR JANELA", window_metrics)])
    lines.extend(["", *format_metrics("METRICAS PRINCIPAIS POR COLETA", collection_metrics)])
    lines.extend(
        [
            "",
            "MATRIZ DE CONFUSAO POR COLETA",
            "-" * 62,
            "Linhas = classe real; colunas = classe prevista",
            "                         Prev. com nematoide  Prev. saudavel",
            f"Real com nematoide              {confusion[0, 0]:4d}              {confusion[0, 1]:4d}",
            f"Real saudavel                   {confusion[1, 0]:4d}              {confusion[1, 1]:4d}",
            "",
            "METRICAS POR CLASSE (POR COLETA)",
            "-" * 62,
        ]
    )
    for class_id, class_name in enumerate(CLASS_NAMES):
        values = report[str(class_id)]
        lines.append(
            f"{class_id} — {class_name}: precision={values['precision']:.6f}, "
            f"recall={values['recall']:.6f}, f1={values['f1-score']:.6f}, "
            f"suporte={int(values['support'])}"
        )
    lines.extend(["", "IMPORTANCIA GLOBAL GRAD-CAM POR SENSOR", "-" * 62])
    for index in np.argsort(sensor_percent)[::-1]:
        lines.append(f"{SENSOR_LABELS[index]}: {sensor_percent[index]:.3f}%")
    feature_percent = {feature: 0.0 for feature in FEATURES}
    for _sensor, feature, percentage in ranking:
        feature_percent[feature] += percentage
    lines.extend(["", "IMPORTANCIA GLOBAL GRAD-CAM POR CARACTERISTICA", "-" * 62])
    for feature, percentage in sorted(
        feature_percent.items(), key=lambda item: item[1], reverse=True
    ):
        lines.append(f"{feature}: {percentage:.3f}%")
    lines.extend(["", "TOP 15 COMBINACOES SENSOR x CARACTERISTICA", "-" * 62])
    for position, (sensor, feature, percentage) in enumerate(ranking[:15], start=1):
        lines.append(f"{position:02d}. {sensor} / {feature}: {percentage:.3f}%")
    historical_path = ROOT_DIR / "x_tcn" / "resultados" / "resumo_avaliacao_x_tcn.json"
    if historical_path.exists():
        historical = json.loads(historical_path.read_text(encoding="utf-8"))
        historical_window = historical["metricas_por_janela"]
        historical_collection = historical["metricas_por_coleta"]
        lines.extend(
            [
                "",
                "COMPARACAO COM A X-TCN TEMPORAL ANTERIOR",
                "-" * 62,
                f"X-TCN 2D fingerprint/XAI atual — acuracia por coleta: {collection_metrics['accuracy'] * 100:.2f}%",
                f"X-TCN 2D fingerprint/XAI atual — acuracia por janela: {window_metrics['accuracy'] * 100:.2f}%",
                f"X-TCN temporal 1D anterior — acuracia por coleta: {historical_collection['accuracy'] * 100:.2f}%",
                f"X-TCN temporal 1D anterior — acuracia por janela: {historical_window['accuracy'] * 100:.2f}%",
                "A X-TCN anterior recebe as series temporais padronizadas diretamente e usa",
                "Integrated Gradients. A atual aplica G/G0, resume cada janela em 48 descritores,",
                "gera fingerprint 2D, usa atencao dupla e Grad-CAM. Logo, os numeros medem",
                "pipelines diferentes e nao devem ser apresentados como repeticoes do mesmo modelo.",
            ]
        )
    lines.extend(
        [
            "",
            "INTERPRETACAO E LIMITACOES",
            "-" * 62,
            "A acuracia por coleta e a metrica principal porque as janelas da mesma coleta",
            "sao correlacionadas. O Grad-CAM indica regioes que influenciaram as previsoes,",
            "mas nao demonstra causalidade biologica. O teste possui poucas coletas e pertence",
            "ao mesmo conjunto experimental; e necessaria validacao externa em novas campanhas.",
            "Os percentuais Grad-CAM foram normalizados para somar 100% no conjunto de teste.",
            "",
            "ARQUIVOS GERADOS",
            "-" * 62,
            "metricas_reais.json; predicoes_por_janela.csv; predicoes_por_coleta.csv;",
            "matriz_confusao_por_coleta.csv/png; historico_treino.csv/png; split_coletas.csv;",
            "gradcam_sensor_caracteristica_percentual.csv; gradcam_por_classe_percentual.csv;",
            "importancia_sensores_gradcam.csv; importancia_caracteristicas_gradcam.csv;",
            "gradcam_medio_teste_real.png; xtcn_real_checkpoint.pt.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.window < 8 or args.stride < 1:
        raise ValueError("Parametros de treino invalidos.")
    seed_everything(args.seed)
    device = select_device(args.device)
    args.device = str(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Carregando dados reais: {args.dataset}")
    data = load_real_data(args.dataset)
    train_df, val_df, test_df, split = split_collection_groups(data, args.seed)
    split.to_csv(args.output_dir / "split_coletas.csv", index=False)

    partitions = {"treino": train_df, "validacao": val_df, "teste": test_df}
    windows: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    features: dict[str, np.ndarray] = {}
    for name, frame in partitions.items():
        windows[name] = make_windows(
            frame, args.window, args.stride, args.baseline_samples
        )
        features[name] = signals_to_features(windows[name][0], args.sampling_rate)
        print(f"{name.capitalize()}: {frame['Coleta'].nunique()} coletas, {len(windows[name][1])} janelas")

    standardizer = FeatureStandardizer().fit(features["treino"])
    maps = {
        name: to_uint8_maps(standardizer.transform(partition_features))
        for name, partition_features in features.items()
    }
    datasets = {
        name: Uint8FingerprintDataset(maps[name], windows[name][1]) for name in partitions
    }
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "treino": DataLoader(datasets["treino"], batch_size=args.batch_size, shuffle=True, generator=generator),
        "validacao": DataLoader(datasets["validacao"], batch_size=args.batch_size, shuffle=False),
        "teste": DataLoader(datasets["teste"], batch_size=args.batch_size, shuffle=False),
    }

    model = XTCN(num_classes=2)
    history = train_model(
        model,
        loaders["treino"],
        loaders["validacao"],
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
    )
    save_training_history(history, args.output_dir / "historico_treino.png")
    pd.DataFrame(
        {
            "epoca": np.arange(1, len(history.train_loss) + 1),
            "loss_treino": history.train_loss,
            "acuracia_treino": history.train_accuracy,
            "loss_validacao": history.val_loss,
            "acuracia_validacao": history.val_accuracy,
        }
    ).to_csv(args.output_dir / "historico_treino.csv", index=False)

    # Seleciona a duração sem olhar o teste e retreina do zero em todo o
    # desenvolvimento, procedimento mais adequado para um conjunto pequeno.
    selected_epoch = int(np.argmax(history.val_accuracy)) + 1
    development_features = np.concatenate((features["treino"], features["validacao"]))
    development_labels = np.concatenate((windows["treino"][1], windows["validacao"][1]))
    final_standardizer = FeatureStandardizer().fit(development_features)
    development_maps = to_uint8_maps(final_standardizer.transform(development_features))
    test_maps = to_uint8_maps(final_standardizer.transform(features["teste"]))
    final_train_dataset = Uint8FingerprintDataset(development_maps, development_labels)
    final_test_dataset = Uint8FingerprintDataset(test_maps, windows["teste"][1])
    final_generator = torch.Generator().manual_seed(args.seed)
    final_train_loader = DataLoader(
        final_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=final_generator,
    )
    final_test_loader = DataLoader(
        final_test_dataset, batch_size=args.batch_size, shuffle=False
    )
    seed_everything(args.seed)
    model = XTCN(num_classes=2)
    print(
        f"Retreino final: {len(development_labels)} janelas de 24 coletas, "
        f"{selected_epoch} epocas selecionadas sem consultar o teste."
    )
    fit_fixed_epochs(
        model,
        final_train_loader,
        selected_epoch,
        args.learning_rate,
        device,
    )
    standardizer = final_standardizer
    loaders["teste"] = final_test_loader

    test_probabilities, test_labels = predict(model, loaders["teste"], device)
    test_predictions = test_probabilities.argmax(axis=1)
    window_metrics = metric_bundle(test_labels, test_predictions)
    window_predictions = pd.DataFrame(
        {
            "coleta": windows["teste"][2],
            "classe_real": test_labels,
            "classe_prevista": test_predictions,
            "probabilidade_com_nematoide": test_probabilities[:, 0],
            "probabilidade_saudavel": test_probabilities[:, 1],
        }
    )
    window_predictions.to_csv(args.output_dir / "predicoes_por_janela.csv", index=False)
    collection_predictions = (
        window_predictions.groupby("coleta", as_index=False)
        .agg(
            classe_real=("classe_real", "first"),
            probabilidade_com_nematoide=("probabilidade_com_nematoide", "mean"),
            probabilidade_saudavel=("probabilidade_saudavel", "mean"),
            numero_janelas=("classe_real", "size"),
        )
    )
    collection_predictions["classe_prevista"] = np.where(
        collection_predictions["probabilidade_com_nematoide"]
        >= collection_predictions["probabilidade_saudavel"],
        0,
        1,
    )
    collection_predictions.to_csv(args.output_dir / "predicoes_por_coleta.csv", index=False)
    y_collection = collection_predictions["classe_real"].to_numpy()
    p_collection = collection_predictions["classe_prevista"].to_numpy()
    collection_metrics = metric_bundle(y_collection, p_collection)
    confusion = confusion_matrix(y_collection, p_collection, labels=[0, 1])
    pd.DataFrame(
        confusion,
        index=["real_0_com_nematoide", "real_1_saudavel"],
        columns=["previsto_0_com_nematoide", "previsto_1_saudavel"],
    ).to_csv(args.output_dir / "matriz_confusao_por_coleta.csv")
    save_confusion(confusion, args.output_dir / "matriz_confusao_por_coleta.png")
    report = classification_report(
        y_collection,
        p_collection,
        labels=[0, 1],
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )
    # Inclui chaves numéricas para facilitar a composição do TXT.
    numeric_report = classification_report(
        y_collection, p_collection, labels=[0, 1], output_dict=True, zero_division=0
    )

    global_cells, class_cells = gradcam_importance(model, loaders["teste"], device)
    global_percent, ranking = save_importance_outputs(global_cells, class_cells, args.output_dir)
    sensor_percent = global_percent.sum(axis=1)

    summary = {
        "modelo": "X-TCN 2D com atencao dupla e Grad-CAM",
        "dados": "reais",
        "dataset": str(args.dataset.resolve()),
        "sensores": list(SENSORS),
        "classes": {"0": CLASS_NAMES[0], "1": CLASS_NAMES[1]},
        "protocolo_split": "por coleta; teste 30%; validacao retirada do desenvolvimento",
        "parametros": {
            "seed": args.seed,
            "window": args.window,
            "stride": args.stride,
            "baseline_samples": args.baseline_samples,
            "sampling_rate": args.sampling_rate,
            "epochs": args.epochs,
            "selected_epoch_by_validation_accuracy": selected_epoch,
            "final_training_collections": int(
                pd.concat((train_df, val_df))["Coleta"].nunique()
            ),
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "device": str(device),
        },
        "quantidades": {
            "linhas": len(data),
            "coletas_treino": int(train_df["Coleta"].nunique()),
            "coletas_validacao": int(val_df["Coleta"].nunique()),
            "coletas_teste": int(test_df["Coleta"].nunique()),
            "janelas_treino": len(windows["treino"][1]),
            "janelas_validacao": len(windows["validacao"][1]),
            "janelas_teste": len(windows["teste"][1]),
        },
        "metricas_por_janela": window_metrics,
        "metricas_por_coleta": collection_metrics,
        "matriz_confusao_por_coleta": confusion.tolist(),
        "relatorio_por_classe": report,
        "nota": "A base real possui duas classes, nao seis tratamentos rotulados.",
    }
    (args.output_dir / "metricas_reais.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean": torch.from_numpy(standardizer.mean_.copy()),
            "feature_scale": torch.from_numpy(standardizer.scale_.copy()),
            "sensors": SENSORS,
            "features": FEATURES,
            "class_names": CLASS_NAMES,
            "window": args.window,
            "stride": args.stride,
            "baseline_samples": args.baseline_samples,
        },
        args.output_dir / "xtcn_real_checkpoint.pt",
    )
    write_report(
        args.output_dir / "RELATORIO_ACURACIAS_E_RESULTADOS.txt",
        args,
        data,
        split,
        {name: len(windows[name][1]) for name in partitions},
        history,
        window_metrics,
        collection_metrics,
        confusion,
        numeric_report,
        ranking,
        sensor_percent,
    )
    print(json.dumps({"metricas_por_janela": window_metrics, "metricas_por_coleta": collection_metrics}, indent=2))
    print(f"Relatorio: {args.output_dir / 'RELATORIO_ACURACIAS_E_RESULTADOS.txt'}")


if __name__ == "__main__":
    main()

