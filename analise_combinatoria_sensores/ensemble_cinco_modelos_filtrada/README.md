# Ensemble de cinco modelos na base filtrada

O ensemble combina Extra Trees, Random Forest, TCN, MLP e k-NN, sem SVM ou LDA.
Em cada um dos dez folds LOSO, as probabilidades usadas para uma leitura foram
produzidas por modelos que não treinaram com o vaso dessa leitura. O script
alinha as previsões salvas por fold, coleta, vaso, classe e tempo; para o Extra
Trees, cujo arquivo não contém tempo, confirma a ordem original das linhas
antes de anexá-lo. Todos os cinco componentes reproduziram exatamente os
acertos individuais já publicados em ambos os cenários.

| Entradas | Acertos / leituras | Acurácia | Acurácia balanceada | Recall inoculado | F1 macro |
|---|---:|---:|---:|---:|---:|
| Cinco MQ | 51.828/55.946 | 92,64% | 92,21% | 84,83% | 92,52% |
| Cinco MQ + solo, temperatura e pressão | 53.357/55.946 | 95,37% | 95,24% | 93,04% | 95,34% |

A decisão usa a média não ponderada das cinco probabilidades da classe 1,
com limiar 0,5. As entradas são MQ2, MQ3, MQ8, MQ135 e MQ138, isolados ou
acompanhados de índice de solo, temperatura e pressão. São 31 coletas de dez
vasos após a exclusão de cinco coletas.

Arquivos: `metricas.csv`, `predicoes_mq.csv`, `predicoes_mq_ambiente.csv` e
`metodo.json`. Para reproduzir a agregação das probabilidades já salvas:

```powershell
.\.venv\Scripts\python.exe analise_combinatoria_sensores\comparar_ensemble_cinco_modelos_filtrada.py
```

As exclusões, a escolha dos cinco MQ e a composição do ensemble foram definidas
após inspeção de resultados obtidos nos mesmos dez vasos usados na avaliação.
Os percentuais são exploratórios e não estimam desempenho externo.
O ensemble ficou abaixo do Extra Trees individual nos dois cenários.
