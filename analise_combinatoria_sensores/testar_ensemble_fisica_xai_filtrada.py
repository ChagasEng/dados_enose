"""Exploratory, nested LOSO test of a six-MQ temporal/tabular stack.

The retained data already underwent pressure-based cropping. Physical R0 is
unavailable, so the per-collection median of the first 20 retained signal
readings is used only as a relative signal baseline. Grad-CAM selects a
temporal region using the training vessels of each fit, never test labels.
"""

from __future__ import annotations

import json
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch import nn

from rodar_combinacoes import DATASET
from validacao_por_vaso import vessel_id


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ensemble_fisica_xai_filtrada"
EXCLUDED = ROOT / "validacao_sem_cinco_coletas" / "coletas_excluidas.csv"
REFERENCE = ROOT / "ensemble_cinco_modelos_filtrada" / "predicoes_mq_ambiente.csv"
MQ = ["MQ2", "MQ3", "MQ7", "MQ8", "MQ135", "MQ138"]
ENV = ["Temp_C", "Soil_indice_0_1"]
SEED = 42
N_POINTS = 256
WINDOW = 128
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


class Block(nn.Module):
    def __init__(self, n_in: int, n_out: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_in, n_out, 3, padding=2 * dilation, dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(n_out, n_out, 3, padding=2 * dilation, dilation=dilation),
            nn.ReLU(),
        )
        self.skip = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.net(x)[..., : x.shape[-1]] + self.skip(x))


class SmallTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(Block(6, 16, 1), Block(16, 24, 2))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Dropout(0.40), nn.Linear(24, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(x))


def load_data() -> tuple[pd.DataFrame, list[pd.DataFrame], np.ndarray, np.ndarray]:
    excluded = set(pd.read_csv(EXCLUDED).Coleta)
    frame = pd.read_csv(DATASET)
    frame = frame.loc[~frame.Coleta.isin(excluded)].copy()
    frame["Vaso_id"] = vessel_id(frame)
    assert (len(frame), frame.Coleta.nunique(), frame.Vaso_id.nunique()) == (55946, 31, 10)
    blocks = [b.sort_values("Tempo").copy() for _, b in frame.groupby("Coleta", sort=True)]
    labels = np.array([int(b.Classe.iloc[0]) for b in blocks], dtype=np.int64)
    groups = np.array([b.Vaso_id.iloc[0] for b in blocks])
    assert all(b.Classe.nunique() == b.Vaso_id.nunique() == 1 for b in blocks)
    return frame, blocks, labels, groups


def corrected_sequences(blocks: list[pd.DataFrame], train_idx: np.ndarray) -> np.ndarray:
    # Each regression is fitted only on the training vessels of this fit.
    env_train = np.concatenate([blocks[i][ENV].to_numpy(np.float64) for i in train_idx])
    mq_train = np.concatenate([blocks[i][MQ].to_numpy(np.float64) for i in train_idx])
    design = np.column_stack([np.ones(len(env_train)), env_train])
    coefficients = np.linalg.lstsq(design, mq_train, rcond=None)[0]
    prediction_mean = (design @ coefficients).mean(axis=0)
    series = []
    for block in blocks:
        env = block[ENV].to_numpy(np.float64)
        raw = block[MQ].to_numpy(np.float64)
        predicted = np.column_stack([np.ones(len(env)), env]) @ coefficients
        adjusted = raw - predicted + prediction_mean
        baseline = np.median(adjusted[:20], axis=0)
        baseline = np.where(np.abs(baseline) < 1e-6, 1.0, baseline)
        relative = adjusted / baseline - 1.0
        time_axis = block.Tempo.to_numpy(np.float64)
        time_axis = (time_axis - time_axis[0]) / (time_axis[-1] - time_axis[0])
        target_axis = np.linspace(0, 1, N_POINTS)
        sampled = np.stack([np.interp(target_axis, time_axis, relative[:, j])
                            for j in range(len(MQ))])
        series.append(np.clip(sampled, -5, 5))
    return np.asarray(series, dtype=np.float32)


def train_tcn(x: np.ndarray, y: np.ndarray, epochs: int) -> SmallTCN:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    network = SmallTCN()
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3, weight_decay=1e-4)
    count = np.bincount(y, minlength=2)
    assert np.all(count > 0)
    weights = torch.tensor(len(y) / (2 * count), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    for _ in range(epochs):
        network.train()
        optimizer.zero_grad()
        criterion(network(xt), yt).backward()
        optimizer.step()
    network.eval()
    return network


def gradcam_region(model: SmallTCN, x: np.ndarray, y: np.ndarray) -> int:
    captured: dict[str, torch.Tensor] = {}
    layer = model.blocks[-1].net[2]
    forward = layer.register_forward_hook(lambda _, __, output: captured.update(activation=output))
    backward = layer.register_full_backward_hook(
        lambda _, __, grad_output: captured.update(gradient=grad_output[0]))
    try:
        model.zero_grad()
        score = model(torch.from_numpy(x)).gather(
            1, torch.from_numpy(y).reshape(-1, 1)).sum()
        score.backward()
        activation = captured["activation"]
        gradient = captured["gradient"]
        importance = gradient.mean(dim=2, keepdim=True)
        saliency = torch.relu((importance * activation).sum(dim=1)).detach().numpy()
        saliency = saliency[:, :N_POINTS]
        saliency /= np.maximum(saliency.sum(axis=1, keepdims=True), 1e-8)
        average = saliency.mean(axis=0)
        if not np.any(average > 0):
            return (N_POINTS - WINDOW) // 2
        sums = np.convolve(average, np.ones(WINDOW), mode="valid")
        return int(np.argmax(sums))
    finally:
        forward.remove()
        backward.remove()


def handcrafted_features(sequences: np.ndarray) -> np.ndarray:
    # 3 features per MQ: early slope, absolute area, peak relative signal.
    early = sequences[:, :, :32]
    axis = np.arange(32, dtype=np.float64)
    centered = axis - axis.mean()
    slope = (early @ centered) / (centered @ centered)
    area = np.trapezoid(np.abs(sequences), axis=-1) / (sequences.shape[-1] - 1)
    peak = (1.0 + sequences).max(axis=-1)
    return np.stack([slope, area, peak], axis=2).reshape(len(sequences), -1)


def calibrated_svm(train_x: np.ndarray, train_y: np.ndarray,
                   train_groups: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    oof = np.full(len(train_y), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for fit_idx, val_idx in cv.split(train_x, train_y, train_groups):
            model = make_pipeline(StandardScaler(), LinearSVC(
                C=1, class_weight="balanced", dual=False, max_iter=10000,
                random_state=SEED))
            model.fit(train_x[fit_idx], train_y[fit_idx])
            oof[val_idx] = model.decision_function(train_x[val_idx])
        assert np.isfinite(oof).all()
        sigmoid = LogisticRegression(random_state=SEED).fit(oof.reshape(-1, 1), train_y)
        final = make_pipeline(StandardScaler(), LinearSVC(
            C=1, class_weight="balanced", dual=False, max_iter=10000,
            random_state=SEED)).fit(train_x, train_y)
    return sigmoid.predict_proba(final.decision_function(test_x).reshape(-1, 1))[:, 1]


def base_probabilities(blocks: list[pd.DataFrame], labels: np.ndarray,
                       groups: np.ndarray, train_idx: np.ndarray,
                       test_idx: np.ndarray) -> tuple[np.ndarray, dict]:
    all_series = corrected_sequences(blocks, train_idx)
    channel_mean = all_series[train_idx].mean(axis=(0, 2), keepdims=True)
    channel_std = all_series[train_idx].std(axis=(0, 2), keepdims=True)
    normalized = (all_series - channel_mean) / np.maximum(channel_std, 1e-6)
    initial = train_tcn(normalized[train_idx], labels[train_idx], 20)
    start = gradcam_region(initial, normalized[train_idx], labels[train_idx])
    signal = normalized[:, :, start:start + WINDOW].copy()
    final_tcn = train_tcn(signal[train_idx], labels[train_idx], 40)
    with torch.no_grad():
        tcn_p = torch.softmax(final_tcn(torch.from_numpy(signal[test_idx])), dim=1)[:, 1].numpy()
    features = handcrafted_features(all_series[:, :, start:start + WINDOW])
    et = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2,
                              max_features="sqrt", random_state=SEED, n_jobs=1)
    et.fit(features[train_idx], labels[train_idx])
    et_p = et.predict_proba(features[test_idx])[:, 1]
    svm_p = calibrated_svm(features[train_idx], labels[train_idx],
                           groups[train_idx], features[test_idx])
    lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(
        solver="lsqr", shrinkage="auto"))
    lda.fit(features[train_idx], labels[train_idx])
    lda_p = lda.predict_proba(features[test_idx])[:, 1]
    probabilities = np.column_stack([tcn_p, et_p, svm_p, lda_p])
    assert np.isfinite(probabilities).all()
    return probabilities, {"start_256": start, "stop_256": start + WINDOW}


def reference_collection_scores() -> pd.DataFrame:
    saved = pd.read_csv(REFERENCE)
    assert (len(saved), saved.Coleta.nunique()) == (55946, 31)
    return saved.groupby(["Coleta", "Vaso_id", "Classe"], as_index=False).agg(
        et_p=("p1_Extra Trees", "mean"),
        ensemble5_p=("p1_ensemble", "mean"))


def raw_mean_baseline(blocks: list[pd.DataFrame], labels: np.ndarray,
                      groups: np.ndarray) -> np.ndarray:
    features = np.asarray([np.r_[block[MQ].mean().to_numpy(),
                                 block[ENV].mean().to_numpy()] for block in blocks])
    probability = np.full(len(blocks), np.nan)
    for train_idx, test_idx in LeaveOneGroupOut().split(features, labels, groups):
        model = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2,
                                     random_state=SEED, n_jobs=1)
        model.fit(features[train_idx], labels[train_idx])
        probability[test_idx] = model.predict_proba(features[test_idx])[:, 1]
    assert np.isfinite(probability).all()
    return probability


def add_metrics(rows: list[dict], data: pd.DataFrame, name: str,
                probability_column: str, reference_counts: pd.Series) -> None:
    true = data.Classe.to_numpy()
    predicted = (data[probability_column].to_numpy() >= .5).astype(int)
    rows.append({"modelo": name, "unidade": "coleta", "n": len(data),
                 "acertos": int((predicted == true).sum()),
                 "acuracia": accuracy_score(true, predicted),
                 "acuracia_balanceada": balanced_accuracy_score(true, predicted),
                 "f1_macro": f1_score(true, predicted, average="macro")})
    nreadings = data.Coleta.map(reference_counts).to_numpy()
    rows.append({"modelo": name, "unidade": "leitura; decisao propagada da coleta",
                 "n": int(nreadings.sum()),
                 "acertos": int(nreadings[predicted == true].sum()),
                 "acuracia": np.average(predicted == true, weights=nreadings),
                 "acuracia_balanceada": np.mean([
                     np.average(predicted[true == c] == c, weights=nreadings[true == c])
                     for c in (0, 1)]),
                 "f1_macro": f1_score(np.repeat(true, nreadings),
                                      np.repeat(predicted, nreadings), average="macro")})


def main() -> None:
    OUT.mkdir(exist_ok=True)
    frame, blocks, labels, groups = load_data()
    names = [b.Coleta.iloc[0] for b in blocks]
    outer = LeaveOneGroupOut()
    predictions = []
    diagnostics = []
    for fold, (train_idx, test_idx) in enumerate(outer.split(np.arange(len(blocks)), labels, groups), 1):
        checkpoint = OUT / f"fold_{fold:02d}.csv"
        if checkpoint.exists():
            predictions.append(pd.read_csv(checkpoint))
            print(f"fold {fold}/10 cached", flush=True)
            continue
        started = time.monotonic()
        meta_oof = np.full((len(train_idx), 3), np.nan)
        inner = LeaveOneGroupOut()
        for inner_fit, inner_val in inner.split(train_idx, labels[train_idx], groups[train_idx]):
            absolute_fit = train_idx[inner_fit]
            absolute_val = train_idx[inner_val]
            p, _ = base_probabilities(blocks, labels, groups, absolute_fit, absolute_val)
            meta_oof[inner_val] = p[:, :3]
        assert np.isfinite(meta_oof).all()
        meta = LogisticRegression(C=1, class_weight="balanced", random_state=SEED)
        vessel_counts = pd.Series(groups[train_idx]).value_counts()
        sample_weight = np.array([1 / vessel_counts[g] for g in groups[train_idx]])
        meta.fit(meta_oof, labels[train_idx], sample_weight=sample_weight)
        test_p, detail = base_probabilities(blocks, labels, groups, train_idx, test_idx)
        combined = meta.predict_proba(test_p[:, :3])[:, 1]
        part = pd.DataFrame({"Coleta": np.array(names)[test_idx],
                             "Vaso_id": groups[test_idx], "Classe": labels[test_idx],
                             "tcn_p": test_p[:, 0], "et_features_p": test_p[:, 1],
                             "svm_features_p": test_p[:, 2], "lda_features_p": test_p[:, 3],
                             "stack_p": combined})
        part.insert(0, "fold", fold)
        part.to_csv(checkpoint, index=False)
        predictions.append(part)
        detail.update({"fold": fold, "vaso_teste": groups[test_idx][0],
                       "n_treino_coletas": len(train_idx), "n_teste_coletas": len(test_idx),
                       "segundos": time.monotonic() - started})
        diagnostics.append(detail)
        print(f"fold {fold}/10 {detail['vaso_teste']} {detail['segundos']:.1f}s", flush=True)
    result = pd.concat(predictions, ignore_index=True)
    assert result.Coleta.nunique() == 31 and not result.Coleta.duplicated().any()
    reference = reference_collection_scores()
    result = result.merge(reference, on=["Coleta", "Vaso_id", "Classe"], validate="one_to_one")
    mean_p = raw_mean_baseline(blocks, labels, groups)
    result["et_raw_mean_p"] = result.Coleta.map(dict(zip(names, mean_p)))
    result.to_csv(OUT / "predicoes_por_coleta.csv", index=False)
    counts = frame.groupby("Coleta").size()
    metrics = []
    for name, column in [("ET original (5 MQ + ambiente)", "et_p"),
                         ("Ensemble original (5 modelos)", "ensemble5_p"),
                         ("ET medias brutas por coleta (6 MQ + ambiente)", "et_raw_mean_p"),
                         ("TCN proposta (6 MQ)", "tcn_p"),
                         ("ET descritores (6 MQ)", "et_features_p"),
                         ("SVM descritores (6 MQ)", "svm_features_p"),
                         ("LDA descritores (6 MQ)", "lda_features_p"),
                         ("Stack TCN+ET+SVM (meta-logistica)", "stack_p")]:
        add_metrics(metrics, result, name, column, counts)
        by_vessel = result.groupby(["Vaso_id", "Classe"], as_index=False)[column].mean()
        vessel_true = by_vessel.Classe.to_numpy()
        vessel_predicted = (by_vessel[column].to_numpy() >= .5).astype(int)
        metrics.append({"modelo": name, "unidade": "vaso; media das coletas",
                        "n": len(by_vessel),
                        "acertos": int((vessel_predicted == vessel_true).sum()),
                        "acuracia": accuracy_score(vessel_true, vessel_predicted),
                        "acuracia_balanceada": balanced_accuracy_score(vessel_true, vessel_predicted),
                        "f1_macro": f1_score(vessel_true, vessel_predicted, average="macro")})
    original_readings = pd.read_csv(REFERENCE)
    for name, column in [("ET original (5 MQ + ambiente)", "p1_Extra Trees"),
                         ("Ensemble original (5 modelos)", "p1_ensemble")]:
        true = original_readings.Classe.to_numpy()
        predicted = (original_readings[column].to_numpy() >= .5).astype(int)
        metrics.append({"modelo": name, "unidade": "leitura individual original",
                        "n": len(true), "acertos": int((predicted == true).sum()),
                        "acuracia": accuracy_score(true, predicted),
                        "acuracia_balanceada": balanced_accuracy_score(true, predicted),
                        "f1_macro": f1_score(true, predicted, average="macro")})
    pd.DataFrame(metrics).to_csv(OUT / "metricas.csv", index=False)
    if diagnostics:
        pd.DataFrame(diagnostics).to_csv(OUT / "janelas_criterio_gradcam.csv", index=False)
    method = {
        "base": "55.946 leituras; 31 coletas; 10 vasos; cinco coletas excluidas",
        "unidade_modelo_novo": "coleta; decisao por leitura somente por propagacao",
        "validacao": "LOSO externo por vaso e LOSO interno por vaso para treinar meta-classificador",
        "pressao": "recorte de pressao ja presente no dataset; sem novo alinhamento",
        "baseline": "mediana das 20 primeiras leituras MQ retidas por coleta; proxy de sinal, nao resistencia fisica R0",
        "correcao_ambiental": "regressao linear MQ~temperatura+indice de solo ajustada somente nos vasos de treino de cada fit",
        "sequencia": "6 MQ relativos, 256 pontos reamostrados por coleta; Grad-CAM no treino seleciona janela de 128 pontos",
        "tcn": "2 blocos, canais 16/24, dilatacoes 1/2, dropout 0.4, 20 epocas para Grad-CAM + 40 epocas finais",
        "descritores": "por sensor: derivada inicial, area absoluta, pico relativo; 18 colunas",
        "tabular": "ET 300 arvores min_samples_leaf=2; SVM linear calibrada em 3 folds agrupados; LDA shrinkage auto",
        "controle_ablação": "ET 300 arvores sobre media bruta de 6 MQ + temperatura e indice de solo por coleta",
        "meta": "regressao logistica C=1, class_weight=balanced, pesos iguais por vaso, entradas TCN/ET/SVM",
        "limite": "analise exploratoria sobre coletas e entradas selecionadas apos inspecao de resultados nos mesmos dez vasos; 31 coletas nao sao 55.946 unidades independentes",
    }
    (OUT / "metodo.json").write_text(json.dumps(method, indent=2, ensure_ascii=False), encoding="utf-8")
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
