# Analise exploratoria sem tres vasos com erros concentrados

**Atencao:** este ensaio exclui vasos inteiros selecionados pelos erros do classificador. Para a pergunta sobre manter os vasos e retirar somente coletas ruidosas, consulte `../validacao_qualidade_por_coleta/README.md`: dez vasos, 34 coletas, LOSO refeito com acuracia de 89,66%.

Foram retirados **todos os dados** dos vasos `classe_1_vaso_1` (saudavel 1), `classe_1_vaso_2` (saudavel 2) e `classe_0_vaso_9` (inoculado 9), identificados no CSV. Restaram **7 vasos, 25 coletas e 43.485 leituras cruas**. O conjunto de sensores e os hiperparametros ficaram fixos: MQ2, MQ3, MQ8, MQ135, MQ138 + solo, temperatura e pressao; Extra Trees com 900 arvores.

| Avaliacao | Acertos por leitura | Acuracia normal |
|---|---:|---:|
| LOSO original, 10 vasos | 55.916/63.492 | 88,07% |
| Previsoes do LOSO original, olhando somente os 7 vasos restantes | 42.028/43.485 | **96,65%** |
| Novo LOSO treinado e testado somente nos 7 vasos restantes | 41.872/43.485 | **96,29%** |
| Novo 70/30 por leitura, somente nos 7 vasos restantes | 13.042/13.046 | 99,97% |

O novo LOSO fez **7 rodadas**; em cada uma, treinou com seis vasos e testou todas as leituras do setimo. O percentual total aumentou de 88,07% para 96,29% porque os vasos removidos eram os mais dificeis no teste. Na comparacao que mantem **as mesmas leituras dos sete vasos restantes**, retirar os tres vasos do treinamento reduziu a acuracia de 96,65% para 96,29% (menos 156 acertos). Assim, os dados nao mostram que descartar esses vasos melhora o modelo para os vasos restantes.

O 70/30 por leitura no conjunto reduzido compartilha as mesmas 25 coletas e sete vasos entre treino e teste. Seu 99,97% nao mede generalizacao a vaso novo. Como os vasos foram escolhidos para exclusao depois de observar seus erros, esta e uma **analise exploratoria**, inadequada como resultado principal de validacao do artigo.

Arquivos: `comparacao_metricas.csv`, `predicoes_loso_7_vasos.csv`, `predicoes_70_30_por_leitura_7_vasos.csv`, `acertos_loso_por_vaso.csv` e `metodo.json`.
