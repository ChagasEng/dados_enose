# Piloto de fusão temporal e tabular na base filtrada

Experimento exploratório com **55.946 leituras, 31 coletas e dez vasos**, após
as mesmas cinco exclusões da comparação anterior. O LOSO externo reserva um
vaso inteiro por rodada. A regressão logística final recebe previsões
produzidas por LOSO interno nos outros nove vasos. A janela temporal é
selecionada por Grad-CAM usando apenas os vasos de treino de cada ajuste.

| Modelo | Vasos corretos | Coletas corretas | Leituras corretas |
|---|---:|---:|---:|
| ET original, cinco MQ + ambiente | 10/10 | 31/31 | 54.783/55.946 (97,92%) |
| Ensemble original de cinco modelos | 10/10 | 31/31 | 53.357/55.946 (95,37%) |
| ET, médias brutas por coleta de seis MQ + temperatura e solo | 10/10 | 25/31 (80,65%) | 46.513/55.946 (83,14%) |
| TCN proposta, seis MQ | 6/10 | 16/31 (51,61%) | 28.513/55.946 (50,97%) |
| ET com 18 descritores | 3/10 | 13/31 (41,94%) | 24.012/55.946 (42,92%) |
| SVM com 18 descritores | 1/10 | 3/31 (9,68%) | 4.101/55.946 (7,33%) |
| LDA com 18 descritores | 3/10 | 14/31 (45,16%) | 26.431/55.946 (47,24%) |
| **TCN + ET + SVM, meta-regressão logística** | **3/10** | **9/31 (29,03%)** | **16.945/55.946 (30,29%)** |

**Como ler a tabela:** para os dois modelos originais, a coluna de leituras
mostra suas previsões originais por leitura. Os modelos novos produzem uma
previsão por coleta; na última coluna essa decisão foi atribuída a todas
as leituras da coleta. As colunas de coletas e vasos são comparáveis entre
os modelos. Os vasos são as dez unidades biológicas.

O piloto operacionalizou parte da arquitetura proposta: seis sinais MQ,
correção linear por temperatura e índice de solo ajustada só no treino,
normalização relativa à mediana das primeiras 20 leituras retidas de cada
coleta, TCN de dois blocos com dropout 0,40, escolha de uma janela de 128
pontos por Grad-CAM de treino, e 18 descritores (inclinação inicial, área
absoluta e pico relativo por MQ). ET, SVM e LDA foram avaliados sobre esses
descritores; o meta-classificador combinou TCN, ET e SVM com probabilidades
obtidas dentro de cada treino externo. Na avaliação externa, o Grad-CAM
selecionou a segunda metade da série reamostrada em todas as dez rodadas.

O teste **não reproduz literalmente** a figura: não há resistência física
`R0` no arquivo de leituras, apenas sinais dos MQ; a pressão já foi usada
no recorte anterior e não foi aplicado um novo alinhamento pelo início da
queda de pressão. Portanto, o pico relativo é do sinal registrado, não
`R_pico/R0` elétrico. Os modelos novos usam uma decisão por coleta, enquanto
o ET anterior foi ajustado por leitura. O controle com médias brutas, bem
acima do stack, indica perda de informação na representação experimental;
não identifica qual etapa causou a perda. Os hiperparâmetros do piloto não
foram otimizados.

As exclusões de coletas e as entradas do comparador original foram
selecionadas após inspeção de resultados nos mesmos dez vasos. Assim, esta
comparação não estima desempenho em vasos externos. Os resultados baixos
do piloto não demonstram que toda arquitetura híbrida seja inferior.

Arquivos: `predicoes_por_coleta.csv`, `metricas.csv`, `janelas_criterio_gradcam.csv`
e `metodo.json`. Reproduzir a partir da raiz do projeto:

```powershell
.\.venv\Scripts\python.exe analise_combinatoria_sensores\testar_ensemble_fisica_xai_filtrada.py
```
