# Comparacao: melhor_acuracia vs dia_20_mais

Dataset do melhor_acuracia: `sem pressao/dataset_sem_pressao.csv`
Dataset novo: `dia_20_mais/dia_20_mais/dataset_dia_20_mais.csv`

## Resumo
- melhor_acuracia/sem_pressao: 70.894 linhas
- dia_20_mais: 27.195 linhas
- soma dos dois: 98.089 linhas
- dias sem_pressao: [11, 12, 13, 18, 19]
- dias dia_20_mais: [20, 21]

## Classes
- sem_pressao: classe 0 = 31.817, classe 1 = 39.076
- dia_20_mais: classe 0 = 12.333, classe 1 = 14.862

## Modelo ExtraTrees campeao aplicado no dia_20_mais
- Limiar usado no melhor resultado: 0.57
- Acuracia original melhor_acuracia: 0.9265
- Acuracia no dia_20_mais com limiar 0.50: 0.4535
- Acuracia no dia_20_mais com limiar 0.57: 0.4535
- Balanced accuracy no dia_20_mais com limiar 0.57: 0.5000
- F1 macro no dia_20_mais com limiar 0.57: 0.3120

## Arquivos gerados
- `resumo_geral_datasets.csv`
- `distribuicao_por_dia_classe.csv`
- `distribuicao_por_coleta.csv`
- `estatisticas_mq_por_classe.csv`
- `comparacao_medias_mq_por_classe.csv`
- `avaliacao_modelo_melhor_no_dia_20_mais.json`
- `predicoes_modelo_melhor_no_dia_20_mais.csv`
