from pathlib import Path
import itertools
import json

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT = Path('.').resolve()
PROJECT = ROOT / 'razao_sensores'
DATASET = ROOT / 'sem pressao' / 'dataset_sem_pressao.csv'
OUT_METRICS = PROJECT / 'resultados' / 'metricas'
OUT_MATRICES = PROJECT / 'resultados' / 'matrizes'
OUT_REPORTS = PROJECT / 'resultados' / 'relatorios'
OUT_GRAPHS = PROJECT / 'graficos'
OUT_MODELS = PROJECT / 'modelos'
for d in [OUT_METRICS, OUT_MATRICES, OUT_REPORTS, OUT_GRAPHS, OUT_MODELS]:
    d.mkdir(parents=True, exist_ok=True)

GROUP = 'Coleta'
TARGET = 'Classe'
RANDOM_STATE = 42
EPS = 1e-9

def load_data():
    df = pd.read_csv(DATASET)
    mq = [c for c in df.columns if c.upper().startswith('MQ')]
    for c in mq + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=[GROUP, TARGET] + mq).copy()
    df[TARGET] = df[TARGET].astype(int)
    return df, mq

def add_ratio_features(df, mq):
    out = df.copy()
    ratio_cols = []
    # todas as razoes ordenadas: A/B e B/A carregam informacoes diferentes para modelos lineares
    for a, b in itertools.permutations(mq, 2):
        col = f'{a}_div_{b}'
        out[col] = out[a] / (out[b].abs() + EPS)
        ratio_cols.append(col)
    # razoes focadas no MQ7 como possivel sensor menos afetado pelo ambiente
    mq7_cols = [c for c in ratio_cols if c.startswith('MQ7_div_') or c.endswith('_div_MQ7')]
    return out, ratio_cols, mq7_cols

def evaluate_feature_set(df, feature_cols, name):
    x = df[feature_cols]
    y = df[TARGET]
    groups = df[GROUP]
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = {
        'extra_trees': ExtraTreesClassifier(n_estimators=700, random_state=RANDOM_STATE, n_jobs=-1, max_features='sqrt', min_samples_leaf=10, bootstrap=False),
        'random_forest': RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1, max_features='sqrt', min_samples_leaf=10, class_weight='balanced'),
        'logistic_scaled': Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(max_iter=3000, class_weight='balanced', random_state=RANDOM_STATE))]),
    }
    rows=[]
    preds={}
    for model_name, model in models.items():
        pred_all = np.full(len(df), -1, dtype=int)
        for fold, (tr, te) in enumerate(cv.split(x, y, groups), 1):
            fitted = clone(model)
            fitted.fit(x.iloc[tr], y.iloc[tr])
            pred = fitted.predict(x.iloc[te]).astype(int)
            pred_all[te] = pred
            rows.append({
                'feature_set': name,
                'modelo': model_name,
                'fold': fold,
                'accuracy': accuracy_score(y.iloc[te], pred),
                'balanced_accuracy': balanced_accuracy_score(y.iloc[te], pred),
                'f1_macro': f1_score(y.iloc[te], pred, average='macro'),
            })
        valid = pred_all >= 0
        rows.append({
            'feature_set': name,
            'modelo': model_name,
            'fold': 'media_oof',
            'accuracy': accuracy_score(y.iloc[valid], pred_all[valid]),
            'balanced_accuracy': balanced_accuracy_score(y.iloc[valid], pred_all[valid]),
            'f1_macro': f1_score(y.iloc[valid], pred_all[valid], average='macro'),
        })
        preds[(name, model_name)] = pred_all
    return pd.DataFrame(rows), preds

def save_matrix(y, pred, name):
    cm = confusion_matrix(y, pred, labels=[0,1])
    pd.DataFrame(cm, index=['real_doente','real_saudavel'], columns=['pred_doente','pred_saudavel']).to_csv(OUT_MATRICES / f'matriz_{name}.csv')
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0,1], ['doente','saudavel'])
    ax.set_yticks([0,1], ['doente','saudavel'])
    ax.set_title(name)
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUT_MATRICES / f'matriz_{name}.png', dpi=160)
    plt.close(fig)

def main():
    df, mq = load_data()
    df_ratio, ratio_cols, mq7_ratio_cols = add_ratio_features(df, mq)
    feature_sets = {
        'mq_originais': mq,
        'somente_razoes': ratio_cols,
        'mq_mais_razoes': mq + ratio_cols,
        'razoes_com_mq7': mq7_ratio_cols,
        'mq7_mais_razoes_mq7': ['MQ7'] + mq7_ratio_cols,
    }
    all_rows=[]
    all_preds={}
    for name, cols in feature_sets.items():
        rows, preds = evaluate_feature_set(df_ratio, cols, name)
        all_rows.append(rows)
        all_preds.update(preds)
    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(OUT_METRICS / 'comparacao_razao_sensores_cv.csv', index=False)
    means = result[result['fold'].astype(str)=='media_oof'].sort_values('balanced_accuracy', ascending=False)
    means.to_csv(OUT_METRICS / 'ranking_razao_sensores.csv', index=False)
    best = means.iloc[0].to_dict()
    best_key = (best['feature_set'], best['modelo'])
    pred = all_preds[best_key]
    y = df_ratio[TARGET].to_numpy()
    save_matrix(y, pred, f"melhor_{best['feature_set']}_{best['modelo']}")
    report = classification_report(y, pred, target_names=['doente','saudavel'], digits=4, zero_division=0)
    (OUT_REPORTS / 'relatorio_razao_sensores.txt').write_text('\n'.join([
        'Relatorio - features por razao entre sensores',
        '=============================================',
        '',
        'Hipotese testada:',
        'Dividir o sinal de um sensor pelo sinal de outro pode reduzir a influencia comum do ambiente/caixa, pois variacoes globais tendem a aparecer em varios sensores ao mesmo tempo.',
        '',
        'Feature sets testados:',
        '- mq_originais: sensores MQ puros.',
        '- somente_razoes: todas as razoes sensor_a / sensor_b.',
        '- mq_mais_razoes: sensores puros + razoes.',
        '- razoes_com_mq7: razoes envolvendo MQ7.',
        '- mq7_mais_razoes_mq7: MQ7 puro + razoes envolvendo MQ7.',
        '',
        'Melhor resultado:',
        f"- Feature set: {best['feature_set']}",
        f"- Modelo: {best['modelo']}",
        f"- Accuracy: {best['accuracy']:.6f}",
        f"- Balanced accuracy: {best['balanced_accuracy']:.6f}",
        f"- F1 macro: {best['f1_macro']:.6f}",
        '',
        'Relatorio de classificacao:',
        report,
        '',
        'Ranking completo em resultados/metricas/ranking_razao_sensores.csv',
    ]), encoding='utf-8')
    # treina melhor na base completa
    cols = feature_sets[best['feature_set']]
    model_defs = {
        'extra_trees': ExtraTreesClassifier(n_estimators=700, random_state=RANDOM_STATE, n_jobs=-1, max_features='sqrt', min_samples_leaf=10, bootstrap=False),
        'random_forest': RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1, max_features='sqrt', min_samples_leaf=10, class_weight='balanced'),
        'logistic_scaled': Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(max_iter=3000, class_weight='balanced', random_state=RANDOM_STATE))]),
    }
    final_model = clone(model_defs[best['modelo']])
    final_model.fit(df_ratio[cols], df_ratio[TARGET])
    joblib.dump({'modelo': final_model, 'feature_set': best['feature_set'], 'features': cols, 'metricas_cv': best}, OUT_MODELS / 'melhor_modelo_razao_sensores.joblib')
    # grafico ranking
    fig, ax = plt.subplots(figsize=(11,5))
    labels = means['feature_set'] + ' / ' + means['modelo']
    ax.bar(labels, means['balanced_accuracy'], color='#2874a6')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Balanced accuracy')
    ax.set_title('Comparacao: sensores puros vs razoes entre sensores')
    ax.tick_params(axis='x', rotation=35, labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT_GRAPHS / 'ranking_razao_sensores.png', dpi=170)
    plt.close(fig)
    print(means.to_string(index=False))
    print(OUT_REPORTS / 'relatorio_razao_sensores.txt')

if __name__ == '__main__':
    main()
