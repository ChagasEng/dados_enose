# Gráficos revisados das cinco coletas

Esta versão substitui, para apresentação, os gráficos anteriores com escala automática por painel ou MQ reescalados separadamente em cada coleta. A fonte é a base balanceada de 52.486 leituras e as previsões LOSO já calculadas do Extra Trees.

## Escalas

O eixo horizontal é o tempo registrado na sessão, fixado em **0–680 s** nas cinco coletas. Cada sensor/variável tem o **mesmo limite vertical nas cinco colunas e nos cinco gráficos individuais**:

| Variável | Limites verticais |
|---|---:|
| MQ-2 | 6.000–16.500 unidades brutas |
| MQ-3 | 11.000–19.500 unidades brutas |
| MQ-7 | 0–7.500 unidades brutas |
| MQ-8 | 2.500–9.000 unidades brutas |
| MQ-135 | 10.000–22.000 unidades brutas |
| MQ-138 | 4.500–14.000 unidades brutas |
| Índice operacional do solo | 0–1 |
| Temperatura | 30–48 °C |
| Pressão | 93,49–93,73 kPa |
| Probabilidade prevista da classe 1 | 0–1 |

Os limites foram fixados para abranger todos os valores registrados nessas cinco coletas. O MQ-7 aparece porque foi registrado, mas **não entrou** no ET de cinco MQ. A escala de pressão é estreita em termos absolutos; os valores do eixo devem ser lidos antes de interpretar a altura da curva.

## Linhas e destaque

- Linha cinza clara: sinal bruto de cada leitura.
- Linha colorida forte: média móvel centrada de **10 s**, calculada só para visualização, sem atravessar lacunas de aquisição maiores que 2 s.
- Sombreado vermelho: leituras em que o ET **usando apenas cinco MQ** previu classe 1 (controle), embora a classe real da coleta seja 0 (inoculado). Esse sombreado é calculado das previsões originais, **sem suavização**.
- No último painel, linha azul: probabilidade da classe 1 pelo ET só MQ; linha laranja: probabilidade pelo ET com MQ, condição binária do solo, temperatura e pressão; linha vermelha tracejada: limiar de decisão 0,5. As curvas exibidas têm média móvel, mas a contagem de erros usa as probabilidades originais.
- Espaços em branco no traçado: lacunas de aquisição maiores que 2 s, que não foram interpoladas nem conectadas.

Arquivos principais: `cinco_coletas_escalas_fixas_media_movel.png` e `.pdf`. Os cinco arquivos `*_escalas_fixas.png` permitem ampliar cada coleta mantendo as mesmas escalas.

Essas figuras descrevem associações entre sinais e erros. O destaque de um trecho não demonstra qual sensor causou a decisão do modelo. Para atribuição de importância seriam necessárias análises adicionais, e para explicação física seria necessário controlar/registrar o processo de aquisição.
