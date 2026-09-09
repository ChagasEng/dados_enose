"""Treina uma X-TCN com o mesmo protocolo do Random Forest.

Split 70/30 estratificado por Coleta; MQ-only; explicabilidade por
Integrated Gradients agregada por sensor no conjunto de teste.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "06_07" / "4_polimento_inicial_modelagem" / "datasets_limpos" / "antes_dia_20_pressao_filtrada_estrito.csv"
OUT = ROOT / "x_tcn" / "resultados"
FEATURES = ["MQ2", "MQ3", "MQ7", "MQ8", "MQ135", "MQ138"]
SEED, WINDOW, STRIDE, EPOCHS, BATCH = 42, 128, 64, 60, 32


def seed_everything():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def group_split(df, ratio=0.70, seed=SEED):
    train, test, summary = [], [], []
    for klass, block in df.groupby("Classe"):
        groups = pd.Series(block.Coleta.unique()).sample(frac=1, random_state=seed).tolist()
        n_train = int(len(groups) * ratio)
        tr_groups = set(groups[:n_train])
        train.append(block[block.Coleta.isin(tr_groups)])
        test.append(block[~block.Coleta.isin(tr_groups)])
        summary.append({"classe": int(klass), "coletas_total": len(groups), "coletas_treino": n_train,
                        "coletas_teste": len(groups)-n_train, "linhas_treino": len(train[-1]), "linhas_teste": len(test[-1])})
    return pd.concat(train), pd.concat(test), pd.DataFrame(summary)


def windows(df, scaler):
    xs, ys, groups = [], [], []
    for name, block in df.sort_values(["Coleta", "Tempo"]).groupby("Coleta", sort=False):
        values = scaler.transform(block[FEATURES].to_numpy(np.float32))
        label = int(block.Classe.iloc[0])
        starts = list(range(0, max(1, len(values) - WINDOW + 1), STRIDE))
        if len(values) >= WINDOW and starts[-1] != len(values) - WINDOW: starts.append(len(values) - WINDOW)
        for start in starts:
            chunk = values[start:start + WINDOW]
            if len(chunk) < WINDOW: chunk = np.pad(chunk, ((0, WINDOW-len(chunk)), (0, 0)), mode="edge")
            xs.append(chunk.T); ys.append(label); groups.append(name)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys), np.asarray(groups)


class TemporalBlock(nn.Module):
    def __init__(self, c_in, c_out, dilation):
        super().__init__(); pad = 2 * dilation
        self.net = nn.Sequential(nn.Conv1d(c_in, c_out, 3, padding=pad, dilation=dilation), nn.ReLU(),
                                 nn.Conv1d(c_out, c_out, 3, padding=pad, dilation=dilation), nn.ReLU())
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
    def forward(self, x):
        y = self.net(x)[..., :x.shape[-1]]
        return torch.relu(y + self.skip(x))


class XTCN(nn.Module):
    def __init__(self):
        super().__init__(); self.blocks = nn.Sequential(TemporalBlock(6, 24, 1), TemporalBlock(24, 24, 2), TemporalBlock(24, 32, 4))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(.20), nn.Linear(32, 2))
    def forward(self, x): return self.head(self.blocks(x))


def metricas(y, p):
    return {"accuracy": float(accuracy_score(y,p)), "balanced_accuracy": float(balanced_accuracy_score(y,p)), "f1_macro": float(f1_score(y,p,average="macro"))}


def integrated_gradients(model, x, steps=32):
    baseline = torch.zeros_like(x); total = torch.zeros_like(x)
    for alpha in torch.linspace(0, 1, steps, device=x.device):
        point = (baseline + alpha * (x-baseline)).detach().requires_grad_(True)
        scores = model(point); score = scores.gather(1, scores.argmax(1, keepdim=True)).sum()
        grad = torch.autograd.grad(score, point)[0]; total += grad
    return (x-baseline) * total / steps


def main():
    seed_everything(); OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET).dropna(subset=["Coleta", "Tempo", "Classe", *FEATURES]).copy()
    df.Classe = df.Classe.astype(int)
    train_df, test_df, split = group_split(df)
    scaler = StandardScaler().fit(train_df[FEATURES])
    xtr, ytr, _ = windows(train_df, scaler); xte, yte, gte = windows(test_df, scaler)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr)), batch_size=BATCH, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model = XTCN().to(device)
    counts = np.bincount(ytr, minlength=2); weights = torch.tensor(len(ytr)/(2*counts), dtype=torch.float32, device=device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4); loss_fn = nn.CrossEntropyLoss(weight=weights)
    history=[]
    for epoch in range(1, EPOCHS+1):
        model.train(); losses=[]
        for xb, yb in train_loader:
            xb,yb=xb.to(device),yb.to(device); optim.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward(); optim.step(); losses.append(loss.item())
        history.append({"epoca":epoch,"loss_treino":float(np.mean(losses))})
    model.eval()
    with torch.no_grad():
        prob = torch.softmax(model(torch.from_numpy(xte).to(device)), 1)[:,1].cpu().numpy()
    pred=(prob>=.5).astype(int); window_metrics=metricas(yte,pred)
    by_group=pd.DataFrame({"Coleta":gte,"classe_real":yte,"probabilidade_saudavel":prob}).groupby("Coleta",as_index=False).agg({"classe_real":"first","probabilidade_saudavel":"mean"})
    by_group["classe_prevista"]=(by_group.probabilidade_saudavel>=.5).astype(int)
    group_metrics=metricas(by_group.classe_real,by_group.classe_prevista)
    cm=confusion_matrix(by_group.classe_real,by_group.classe_prevista,labels=[0,1])
    # Explicação: atribuição absoluta média dos sinais, em janelas de teste.
    attrs=[]
    for start in range(0,len(xte),BATCH): attrs.append(integrated_gradients(model, torch.from_numpy(xte[start:start+BATCH]).to(device)).detach().abs().cpu().numpy())
    importance=np.concatenate(attrs).mean(axis=(0,2)); importance/=importance.sum()
    pd.DataFrame({"sensor":FEATURES,"importancia_integrated_gradients":importance}).sort_values("importancia_integrated_gradients",ascending=False).to_csv(OUT/"importancia_sensores_x_tcn.csv",index=False)
    split.to_csv(OUT/"split_70_30_por_coleta_x_tcn.csv",index=False); pd.DataFrame(history).to_csv(OUT/"historico_treino_x_tcn.csv",index=False); by_group.to_csv(OUT/"predicoes_por_coleta_x_tcn.csv",index=False)
    pd.DataFrame(cm,index=["real_0_doente","real_1_saudavel"],columns=["previsto_0_doente","previsto_1_saudavel"]).to_csv(OUT/"matriz_confusao_por_coleta_x_tcn.csv")
    summary={"modelo":"X-TCN", "dataset":str(DATASET.relative_to(ROOT)), "features":FEATURES, "janela":WINDOW,"stride":STRIDE,"epocas":EPOCHS,"dispositivo":str(device),"metricas_por_janela":window_metrics,"metricas_por_coleta":group_metrics,"n_janelas_treino":int(len(ytr)),"n_janelas_teste":int(len(yte)),"n_coletas_teste":int(len(by_group))}
    (OUT/"resumo_avaliacao_x_tcn.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    fig,ax=plt.subplots(figsize=(6,5)); ax.imshow(cm,cmap="Blues"); ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["Doente","Saudável"],yticklabels=["Doente","Saudável"],xlabel="Previsto",ylabel="Real",title="X-TCN — matriz por Coleta")
    for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha="center",va="center"); fig.tight_layout(); fig.savefig(OUT/"matriz_confusao_por_coleta_x_tcn.png",dpi=180); plt.close(fig)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__ == "__main__": main()
