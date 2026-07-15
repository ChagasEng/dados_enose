# Auditoria das coletas criticas

## Escopo

Coletas auditadas: C15, C16, C17, C24, C28, C29, C30, C31 e C32.
Foram verificados sinais crus, sinais corrigidos, variaveis ambientais, eventos abruptos e impacto no ExtraTrees.

## Resultado executivo

- C15 e C16 identicas em todas as colunas verificadas: `True`.
- Baseline recalculada no mesmo split: accuracy `0.9320`; balanced accuracy `0.9300`.
- Baseline do teste completo na auditoria: balanced accuracy `0.9300`.
- Maior associacao ambiental residual nas coletas criticas: `C28` / `MQ3` x `Soil_indice_0_1` = `-0.962`.

## Decisao por coleta

- `C15/C16`: duplicacao confirmada. Manter somente uma delas em qualquer treino/validacao.
- `C17`: quarentena. Ruido multissensor extremo sem variacao ambiental proporcional.
- `C24`: revisar canal MQ138 e log operacional; nao remover automaticamente antes de conferir a origem.
- `C28`: quarentena provisoria. Ha degrau multissensor e forte instabilidade em Soil.
- `C29-C31`: manter para auditoria biologica, mas repetir modelagem sem Soil e com validacao por coleta/dia.
- `C32`: quarentena. Ruido multissensor extremo sem correspondencia ambiental.

## Eventos e ambiente

- `C15`: 12 eventos nos MQ crus; 12 nos corrigidos; 3 linhas multissensor; amplitudes ambiente Soil=0.058, Temp=2.40 C, Pres=0.030 kPa.
- `C16`: 12 eventos nos MQ crus; 12 nos corrigidos; 3 linhas multissensor; amplitudes ambiente Soil=0.058, Temp=2.40 C, Pres=0.030 kPa.
- `C17`: 1480 eventos nos MQ crus; 1479 nos corrigidos; 359 linhas multissensor; amplitudes ambiente Soil=0.051, Temp=2.60 C, Pres=0.020 kPa.
- `C24`: 57 eventos nos MQ crus; 57 nos corrigidos; 0 linhas multissensor; amplitudes ambiente Soil=0.057, Temp=4.60 C, Pres=0.050 kPa.
- `C28`: 13 eventos nos MQ crus; 13 nos corrigidos; 3 linhas multissensor; amplitudes ambiente Soil=0.244, Temp=6.90 C, Pres=0.040 kPa.
- `C29`: 0 eventos nos MQ crus; 0 nos corrigidos; 0 linhas multissensor; amplitudes ambiente Soil=0.310, Temp=9.40 C, Pres=0.020 kPa.
- `C30`: 0 eventos nos MQ crus; 0 nos corrigidos; 0 linhas multissensor; amplitudes ambiente Soil=0.106, Temp=5.20 C, Pres=0.030 kPa.
- `C31`: 0 eventos nos MQ crus; 0 nos corrigidos; 0 linhas multissensor; amplitudes ambiente Soil=0.279, Temp=5.30 C, Pres=0.030 kPa.
- `C32`: 265 eventos nos MQ crus; 264 nos corrigidos; 83 linhas multissensor; amplitudes ambiente Soil=0.092, Temp=5.00 C, Pres=0.020 kPa.

## Coletas criticas presentes no teste original

- `C24`: accuracy por linha `1.0000` em `2121` linhas.
- `C28`: accuracy por linha `1.0000` em `2130` linhas.
- `C30`: accuracy por linha `1.0000` em `1546` linhas.
- `C32`: accuracy por linha `0.8875` em `1271` linhas.

C24, C28 e C30 tiveram 100% de acerto por linha no teste original. Como possuem artefatos ou forte assinatura ambiental, elas podem tornar o teste artificialmente facil. C32 teve desempenho inferior e adiciona ruido ao teste.

## Sensibilidade ao remover coletas do treino

- `baseline_recalculada`: accuracy `0.9320`; balanced accuracy `0.9300`; treino `41258` linhas.
- `sem_C16_duplicada`: accuracy `0.9312`; balanced accuracy `0.9290`; treino `39936` linhas.
- `sem_C17_ruidosa`: accuracy `0.9385`; balanced accuracy `0.9363`; treino `39997` linhas.
- `sem_C29_C31_soil_treino`: accuracy `0.7810`; balanced accuracy `0.7854`; treino `37062` linhas.
- `sem_C15_C16_duplicatas`: accuracy `0.9069`; balanced accuracy `0.9040`; treino `38614` linhas.
- `sem_criticas_do_treino`: accuracy `0.9319`; balanced accuracy `0.9293`; treino `34479` linhas.

Retirar apenas C16 quase nao altera o resultado, portanto uma das duplicatas pode ser removida com seguranca. Retirar C17 melhora a accuracy, confirmando que seu ruido prejudica o treino. Retirar C29 e C31 derruba fortemente o resultado porque elas sao as coletas saudaveis do dia 13 presentes no treino, enquanto C28 e C30, tambem do dia 13, estao no teste. Isso mostra dependencia de representacao por dia/condicao e exige validacao deixando um dia inteiro de fora.

## Ablacao ambiental com compensacao recalculada

- `corrigido_mais_ambiente`: accuracy `0.9320`; balanced accuracy `0.9300`.
- `recompensado_sem_soil`: accuracy `0.8974`; balanced accuracy `0.8941`.
- `recompensado_sem_temp`: accuracy `0.8925`; balanced accuracy `0.8891`.
- `recompensado_sem_pressao`: accuracy `0.9760`; balanced accuracy `0.9756`.
- `somente_mq_corrigido`: accuracy `0.8572`; balanced accuracy `0.8553`.
- `somente_mq_cru`: accuracy `0.8996`; balanced accuracy `0.9000`.

A variante recompensada sem pressao chegou a 97,60% neste split fixo, acima dos 93,20% originais. Isso nao deve ser anunciado como novo resultado final antes de validacao por coleta e por dia: o ganho pode refletir a assinatura de Soil/temperatura dos dias presentes no treino e teste. A queda para cerca de 89% sem Soil ou sem temperatura confirma que o classificador depende bastante do contexto ambiental.

## Efeito das coletas perfeitas no teste

Ao retirar C24, C28 e C30 apenas do calculo do teste, a accuracy cai de `0.9320` para `0.9079`. Essas tres coletas acrescentam aproximadamente `2.41` pontos percentuais a accuracy por linha.

## Arquivos

- `dados_auditoria/resumo_eventos_coletas_criticas.csv`
- `dados_auditoria/eventos_detalhados_coletas_criticas.csv`
- `dados_auditoria/correlacoes_ambiente_antes_depois.csv`
- `dados_auditoria/verificacao_duplicata_C15_C16.csv`
- `dados_auditoria/decisao_recomendada_por_coleta.csv`
- `auditoria_modelo/sensibilidade_remocao_treino.csv`
- `auditoria_modelo/desempenho_teste_por_coleta.csv`
- `auditoria_modelo/sensibilidade_exclusao_teste.csv`
- `auditoria_modelo/ablacao_variaveis_ambientais.csv`
- `graficos/C*_auditoria.png`
- `graficos/sensibilidade_modelo_coletas_criticas.png`

## Limite cientifico

A auditoria identifica anomalias matematicas e associacoes, mas a causa fisica final exige conferir logs de bomba, vedacao, alimentacao, ADC, troca de vaso e planilha bruta.
