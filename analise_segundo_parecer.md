# Verificação do segundo parecer

As análises posteriores sobre irrigação, dia, separação entre períodos,
fase temporal e pressão estão em `analise_terceiro_parecer.md`. Esse
relatório posterior deve prevalecer quando houver maior detalhe.

## Ordem documentada das análises

| Etapa | Base | Procedimento e decisão |
|---|---|---|
| 1 | 36 coletas; 63.492 leituras após filtro estrito | Preparação da base comum. |
| 2 | 36 coletas; divisão histórica por coleta | ET com seis MQ corrigidos e três variáveis ambientais; TCN; exame por faixa operacional do solo. |
| 3 | 36 coletas; LOSO por vaso | Busca de 63 subconjuntos MQ com as três variáveis ambientais fixas; o conjunto MQ2/MQ3/MQ8/MQ135/MQ138 foi escolhido por acurácia por leitura. **MQ7 saiu apenas por essa seleção baseada nos dados.** |
| 4 | 36 coletas; previsões da etapa 3 | Cada leitura foi predita no fold em que seu vaso estava fora do treino. Assim, todas as 36 coletas receberam um `E_c` fora de treino. As quatro coletas com maioria de erros e mais uma sinalizada por saltos compuseram a análise de sensibilidade. |
| 5 | 31 coletas; 55.946 leituras | ET, RF e TCN em dois conjuntos fixos; busca ET de todas as 511 combinações. A escolha e avaliação ocorreram nos mesmos dez vasos. |
| 6 | 36 coletas; base completa | Controles fixos ET com nove entradas, seis MQ, três ambientais e regressão logística com nove entradas, todos no mesmo LOSO. |

## Controles fixos na base completa

As métricas abaixo são **resultados** para a futura seção de Resultados, não para Materiais e Métodos. ET usa 900 árvores, semente 42 e os mesmos dez folds por vaso. Regressão logística usa padronização ajustada apenas no treino de cada fold.

| Entradas e modelo | Acurácia por leitura | Acurácia balanceada por leitura | AUC por coleta | Vasos corretos |
|---|---:|---:|---:|---:|
| ET, seis MQ + solo + temperatura + pressão | 86,84% | 87,08% | 0,864 | 9/10 |
| ET, somente seis MQ | 78,81% | 78,83% | 0,842 | 9/10 |
| ET, somente solo + temperatura + pressão | 84,71% | 84,77% | 0,873 | 10/10 |
| Regressão logística, nove entradas | 68,70% | 68,38% | 0,752 | 8/10 |

O controle só ambiental acertou todos os dez vasos pela regra de uma probabilidade média por vaso. Isso mostra que as variáveis ambientais carregam forte informação sobre os rótulos nesta base; não identifica a causa dessa associação. O ET com as nove entradas foi fixado sem seleção de subconjunto, mas os hiperparâmetros e o pré-processamento já provinham do trabalho anterior, de modo que não constitui validação externa independente.

Para a configuração de cinco MQ mais ambiente selecionada na base completa, a decisão agregada foi 9/10 vasos. O IC binomial exato de Clopper--Pearson para 9/10 é aproximadamente 55,5% a 99,7%; para 10/10, 69,2% a 100%. Esses intervalos não corrigem seleção de entradas nem confundimento.

## Dia, solo e classe

Faixa por coleta atribuída pela **mediana** do índice operacional de solo. Duas coletas controle do dia 13 têm algumas leituras individuais ligeiramente acima de 0,4, embora suas medianas sejam secas.

| Dia do mês | Seco: inoculado | Seco: controle | Úmido: inoculado | Úmido: controle |
|---:|---:|---:|---:|---:|
| 11 | 2 | 5 | 0 | 0 |
| 12 | 5 | 1 | 0 | 0 |
| 13 | 0 | 4 | 0 | 0 |
| 18 | 0 | 0 | 5 | 4 |
| 19 | 0 | 0 | 5 | 5 |

O índice de solo e o dia estão fortemente associados: todas as coletas classificadas como secas ocorreram nos dias 11–13 e todas as úmidas nos dias 18–19. O dia 13 incluiu apenas controles. A tabela não permite separar efeito de classe, dia, condição do solo e possível deriva do equipamento.

## Filtro de pressão

| Classe | Antes | Remoção pela janela de ±30 leituras | Remoção adicional pela faixa estrita | Remoção total |
|---|---:|---:|---:|---:|
| Inoculado (0) | 32.226 | 2.276 (7,06%) | 906 | 3.182 (9,87%) |
| Controle (1) | 39.076 | 2.945 (7,54%) | 1.683 | 4.628 (11,84%) |

## Pontos estatísticos ainda abertos

- Texto sugerido para **Discussão**, fora de Materiais e Métodos: "A posição dos vasos e a ordem de medição não foram documentadas para esta análise; assim, não é possível descartar efeitos de posição ou deriva instrumental associados aos grupos. Os controles não receberam juvenis de *H. glycines*, mas não foi informado se receberam o mesmo veículo de inoculação sem nematoides. Os rótulos refletem os grupos experimentais e não uma confirmação nematológica individual. Essas condições, somadas ao pequeno número de vasos e à associação entre dia e índice de umidade do solo, restringem a interpretação do desempenho como prova de conceito."
- O teste exato de permutação por vaso **não foi executado**. Ele exige refazer o treinamento para cada alocação de rótulos. Para corrigir a busca das 511 entradas e a exclusão definida por erros do modelo, seria necessário repetir também essas decisões dentro de cada permutação. Permutar só as previsões prontas não é válido para essa pergunta.
- Existem $\binom{10}{5}=252$ alocações balanceadas de rótulos, mas o limite de valor-p depende da estatística usada. Para acurácia simétrica à troca global dos nomes das classes, alocações complementares podem produzir a mesma estatística; não se deve prometer automaticamente valor-p mínimo de 1/252.
- Um valor-p causal também depende da justificativa de permutar os rótulos entre vasos, especialmente da alocação experimental original. A randomização dos tratamentos precisa ser confirmada.
- O campo receptivo **local** da TCN implementada é de 29 leituras: há duas convoluções de núcleo 3 por bloco, com dilatações 1, 2 e 4, resultando em $1+4(1+2+4)=29$. No intervalo mediano de 0,24 s, corresponde a cerca de 6,7 s. A média global no tempo agrega as saídas locais por toda a janela de 128 leituras.
- A subamostragem balanceada do teste histórico já existia, mas usa um único sorteio pós-predição. Ela foi mantida apenas como análise de sensibilidade; acurácia balanceada no teste completo é a referência mais estável.

Arquivos reproduzíveis: `analise_combinatoria_sensores/baselines_fixos_base_completa/metricas.csv`, `tabela_solo_classe_dia.csv`, `remocao_pressao_por_classe.csv`; `analise_combinatoria_sensores/comparacao_auc_modelos_filtrada/metricas_auc.csv`; `analise_combinatoria_sensores/avaliacao_parecer_base_completa/intervalos_exatos_por_vaso.csv`.
