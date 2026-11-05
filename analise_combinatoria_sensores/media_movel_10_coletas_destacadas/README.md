# Media movel de 10 nas coletas destacadas por erro

Interpretei "arquivo destacado" como as **quatro coletas com maioria das leituras errada** no LOSO do melhor modelo com ambiente. Testei cada coleta separadamente e as quatro juntas, sempre mantendo os **dez vasos** e as **63.492 leituras** na base. Mantive o Extra Trees e as entradas MQ2, MQ3, MQ8, MQ135, MQ138, solo, temperatura e pressao.

Em cada coleta modificada, as posicoes **1 a 9 ficam originais**. A posicao 10 recebe a media dos valores originais 1 a 10; a 11 recebe a media original 2 a 11; a 12 recebe a media original 3 a 12, e assim sucessivamente. O calculo e feito separadamente para cada coluna de sensor, sem misturar coletas. Tempo, classe e identificacao do vaso permanecem intactos. `auditoria_valores_originais_e_medias.csv` mostra todos os pares original/suavizado.

| Coletas com media movel | Acertos LOSO | Acuracia normal | Variacao de acertos |
|---|---:|---:|---:|
| Nenhuma (original) | 55.916/63.492 | 88,07% | referencia |
| Saudavel vaso 2, dia 12 | 55.957/63.492 | 88,13% | +41 |
| Inoculado vaso 9, dia 19 | 56.072/63.492 | 88,31% | +156 |
| Saudavel vaso 2, dia 19 | 55.629/63.492 | 87,62% | -287 |
| Saudavel vaso 1, dia 19 | 56.030/63.492 | 88,25% | +114 |
| **As quatro juntas** | **56.060/63.492** | **88,29%** | **+144** |

Nas quatro coletas destacadas, os acertos passaram de **234/6.285** para **291/6.285** quando todas foram suavizadas. As coletas saudavel vaso 2/dia 12, inoculado vaso 9/dia 19 e saudavel vaso 2/dia 19 continuaram com **zero leituras corretas**. O ganho local ficou apenas na coleta saudavel vaso 1/dia 19 (234 para 291 acertos). Parte das mudancas na acuracia total vem do retreinamento do modelo com os sinais suavizados presentes nos folds de treino, e nao de corrigir os arquivos destacados no fold em que sao testados.

No **70/30 por leitura**, usando exatamente o mesmo split anterior, a aplicacao nas quatro juntas mudou **18.643/19.048** para **18.644/19.048** acertos: apenas **uma leitura**. Esse protocolo compartilha as mesmas coletas e vasos entre treino e teste.

Conclusao: a media movel localizada nas coletas escolhidas pelos erros produziu **ganho pequeno no LOSO (+0,23 ponto percentual)** e praticamente nenhum efeito no 70/30. Ela nao recuperou os tres arquivos inteiramente classificados de forma errada. Como as coletas foram escolhidas depois de observar seus erros, este ensaio e exploratorio e nao deve substituir a validacao original do artigo.

Arquivos: `comparacao_cenarios.csv`, `acertos_por_coleta_e_cenario.csv`, `auditoria_valores_originais_e_medias.csv`, `predicoes_loso_quatro_juntas.csv`, `comparacao_70_30_por_leitura.csv` e `metodo.json`.
