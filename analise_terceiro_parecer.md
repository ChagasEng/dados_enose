# Verificação do parecer sobre dia, período, pressão e entradas

As análises abaixo usam as **36 coletas e os dez vasos**, sem excluir coletas, exceto quando se indica explicitamente um recorte por dia ou tempo. O ET foi reajustado com os hiperparâmetros anteriores (900 árvores, semente 42). A unidade biológica continua sendo o vaso. Os números deste relatório pertencem à seção de **Resultados/Discussão**; o arquivo `materiais_metodos_artigo_pos_parecer.tex` descreve os procedimentos sem inseri-los como desempenho do método.

## Dia como preditor e transferência entre períodos

| Análise | Resultado por coleta | Interpretação |
|---|---:|---|
| Classe majoritária de cada dia calculada **nas próprias 36 coletas** | 24/36 = 66,67% | Referência descritiva com conhecimento dos rótulos avaliados; não é validação fora de treino. |
| Classe majoritária do mesmo dia **nos outros nove vasos** em cada LOSO; empate resolvido pela maioria global no treino | 14/36 = 38,89% | Referência fora de vaso. A queda decorre especialmente do empate 5 × 5 no dia 19: a retirada de um vaso muda a classe majoritária. |
| ET, nove entradas, LOSO somente 18–19 de maio | 15/19 = 78,95% | Dez vasos, dez coletas inoculadas e nove controles. |
| ET, seis MQ, LOSO somente 18–19 de maio | 13/19 = 68,42% | Mesmo recorte e partições. |
| ET, somente solo, temperatura e pressão, LOSO somente 18–19 de maio | 14/19 = 73,68% | Forte desempenho ambiental mesmo no recorte aproximadamente balanceado. |
| ET, nove entradas, treino 11–13 → teste 18–19 | 9/19 = 47,37% | Acurácia balanceada por coleta 50%; desempenho compatível com previsão de uma classe só no limiar de 0,5. |
| ET, nove entradas, treino 18–19 → teste 11–13 | 6/17 = 35,29% | Acurácia balanceada por coleta 30%. |

As duas separações entre períodos incluem vasos presentes no treino e no teste. Portanto, testam mudança temporal **nos vasos já observados**, não generalização para novas plantas. A irrigação entre 13 e 18 de maio, confirmada pelo pesquisador, ajuda a explicar a mudança do índice de solo; não elimina a associação entre período, umidade do solo, ambiente, composição das classes e possível deriva instrumental. Os dias foram **11, 12, 13, 18 e 19 de maio de 2026**.

## Número principal e contribuição aparente dos MQ

| ET, base completa | Acurácia por leitura | Por coleta | Por vaso |
|---|---:|---:|---:|
| Nove entradas fixas, seis MQ + ambiente | 55.136/63.492 = 86,84% | 31/36 | 9/10 |
| Cinco MQ selecionados entre 63 + ambiente | 55.916/63.492 = 88,07% | 32/36 | 9/10 |
| Somente seis MQ | 50.041/63.492 = 78,81% | 28/36 | 9/10 |
| Somente solo, temperatura e pressão | 53.787/63.492 = 84,71% | 29/36 | 10/10 |

O valor de **86,84% com nove entradas fixas** é o número de destaque mais defensável entre os ET apresentados. A diferença de 1,23 ponto percentual para o conjunto selecionado **não mede isoladamente** o viés de seleção: também mudam as entradas do classificador. Nas 63 combinações MQ + ambiente da base completa, a acurácia por leitura variou de 82,47% a 88,07% (mediana 86,39%; quartis 85,47% e 86,85%; desvio padrão entre configurações 1,18 ponto percentual). O melhor valor continua sujeito à seleção nos mesmos vasos.

A diferença pareada, por vaso, entre nove entradas e somente ambiente foi de +2,80 pontos percentuais de acurácia **por leitura dentro de vaso** em média (melhora em seis vasos, piora em três, empate em um). A enumeração bicaudal das 1.024 inversões de sinal deu $p=0{,}348$. Por acurácia **por coleta dentro de vaso**, a diferença média foi +6,67 pontos (melhora em dois vasos, empate em oito), com $p=0{,}50$. Essa enumeração é **exploratória**: a simetria exigida pelo teste não foi demonstrada e os ajustes LOSO compartilham vasos no treinamento. Não há evidência robusta aqui de ganho dos MQ além das variáveis ambientais.

O teste de permutação de rótulos por vaso, refazendo os modelos nas 252 alocações balanceadas, **não foi executado**. Sua interpretação causal depende da alocação experimental, que não foi documentada. Para corrigir também a busca de 63/511 configurações e a exclusão definida após observar erros, seria preciso repetir essas decisões dentro de cada permutação. Não atribuir valor-p a esse procedimento não realizado.

## Busca combinatória na base filtrada

Na base filtrada de 31 coletas e 55.946 leituras, a busca combinatória
do ET **somente MQ** teve como vencedor os **seis** sensores MQ2, MQ3,
MQ7, MQ8, MQ135 e MQ138: **53.388/55.946 = 95,43%**. A busca MQ com as
três variáveis ambientais fixas selecionou **cinco** sensores, sem MQ7:
MQ2, MQ3, MQ8, MQ135 e MQ138 + solo, temperatura e pressão,
**54.783/55.946 = 97,92%**. A busca das 511 combinações de nove variáveis
confirmou o mesmo vencedor com ambiente. Quando os **cinco MQ** foram
fixados para comparar ET, RF e TCN sem ambiente, o ET acertou
**53.015/55.946 = 94,76%**. Assim, 53.388 não pode ser atribuído ao
conjunto de cinco MQ.

## Cinco coletas excluídas na análise secundária

| Dia | Grupo | Vaso físico | Motivo |
|---:|---|---:|---|
| 12 | Controle | 2 | Maioria das leituras incorreta no LOSO original. |
| 19 | Inoculado | 10 (rotulado 1 na planilha derivada) | Ruído multissensor. |
| 19 | Inoculado | 9 | Maioria incorreta. |
| 19 | Controle | 1 | Maioria incorreta e ruído multissensor. |
| 19 | Controle | 2 | Maioria incorreta. |

Apenas a coleta controle do dia 12 pertence à **classe minoritária no próprio dia** (1 controle, 5 inoculados). No dia 19 havia 5 coletas de cada classe; portanto, as outras quatro exclusões não se concentram na minoria daquele dia. Isso não resolve o viés de exclusão pós-erro, mas refuta a concentração sugerida pelo parecer.

## Normalização por dia e fase temporal

Na padronização dos seis MQ separada por dia, as médias e os desvios padrão de cada fold foram calculados **somente nos outros nove vasos**. O LOSO completo resultou em 49.927/63.492 = 78,64% por leitura e 28/36 coletas, frente a 78,81% e 28/36 com seis MQ brutos. Essa sensibilidade específica mudou pouco o resultado de MQ isolados; não valida transferência entre períodos.

O recorte proporcional original reteve 11.604 leituras com tempo abaixo de 200 s (18,28% de 63.492) e 51.888 entre 200 e 800 s; nenhuma leitura retida ficou acima de 800 s. Após treinar novamente apenas com as 51.888 leituras da **janela fixa de 200–800 s**:

| Entradas ET no LOSO | Acurácia por leitura | Por coleta |
|---|---:|---:|
| Nove entradas | 46.053/51.888 = 88,75% | 32/36 |
| Seis MQ | 41.252/51.888 = 79,50% | 27/36 |
| Ambiente | 44.312/51.888 = 85,40% | 31/36 |

O corte por tempo real altera as métricas, mas preserva o desempenho elevado do ambiente. Como a base não contém uma marca verificada da abertura da válvula em cada série, **200–800 s não pode ser identificado com segurança como a fase de amostragem em todas as coletas**. As durações **originais**, calculadas nas 36 abas correspondentes do arquivo `coletas.xlsx` como tempo máximo menos mínimo, tiveram mediana de 598,37 s no grupo inoculado (449,65–935,39 s) e 674,15 s nos controles (541,65–1047,59 s). O cronograma do fluxograma não descreve rigidamente cada registro.

## Pressão e operação da câmara

O pesquisador confirmou que o ESP32 controlava a depressurização até o alvo aproximado de 70 kPa. O fluxograma enviado mostra que a **bomba era desligada**, a válvula de entrada era aberta e a pressão interna subia para cerca de 94–95 kPa. A válvula permanecia aberta durante **cinco minutos de aquisição e cinco minutos de análise das respostas**. Só depois era fechada; na desaturação, a bomba expulsava o ar da câmara por três minutos. Portanto, a afirmação anterior de que o ar ficava em uma câmara fechada durante toda a leitura estava incorreta. O fluxograma não descreve fluxo forçado contínuo durante a leitura.

Após o filtro de saltos de pressão, a mediana global usada na segunda etapa foi **93,69 kPa**. A faixa de mediana $\pm0{,}50$ kPa era **93,19–94,19 kPa**, próxima da pressão externa informada (~94 kPa). A segunda etapa excluiu mais 906/32.226 = 2,81% das leituras originais da classe 0 e 1.683/39.076 = 4,31% da classe 1. Após ambas as etapas, a remoção total foi 9,87% e 11,84%, respectivamente.

| Pressão na base final (kPa) | Classe 0, inoculado | Classe 1, controle |
|---|---:|---:|
| Média ± desvio padrão | 93,633 ± 0,085 | 93,672 ± 0,120 |
| Mediana | 93,69 | 93,71 |
| Percentis 5–95 | 93,52–93,73 | 93,51–93,86 |
| Mínimo–máximo | 93,47–93,75 | 93,49–93,88 |

A diferença de médias é 0,039 kPa, apenas descritiva: leituras sucessivas não são unidades independentes para testar essa diferença.

Os mínimos de pressão nas abas originais variaram entre 43,38 e 83,53 kPa, inclusive valores abaixo e acima do alvo aproximado de 70 kPa. Esse intervalo recomenda apresentar 70 kPa como **valor operacional de referência**, sem afirmar que foi alcançado de forma idêntica em todas as coletas. O tempo do mínimo também variou; a associação com as fases do fluxograma é aproximada.

## Limitações para a Discussão

Não há confirmação documentada, nesta base, de randomização da posição dos vasos ou da ordem de medição, aplicação de veículo sem J2 aos controles, confirmação nematológica individual, cultivar, estádio, composição do substrato, volume do vaso, condições de cultivo ou operação contínua dos MQ entre coletas. Essas lacunas devem ser relatadas como limitações, sem inventar procedimentos. A irrigação entre os dois períodos deve ser descrita porque altera a interpretação da umidade do solo. A AUC reunida por coleta combina probabilidades de modelos ajustados em folds distintos e pode refletir calibrações diferentes; por isso, tratá-la como descritiva.

Arquivos reproduzíveis: `analise_combinatoria_sensores/avaliar_confundimento_dia.py`, `avaliar_fase_temporal.py`, `resumir_coletas_originais.py`; resultados em `analise_combinatoria_sensores/confundimento_dia/`.
