# Auditoria das coletas criticas

## Escopo

Coletas auditadas: C15, C16, C17, C24, C28, C29, C30, C31 e C32.
Foram verificados sinais crus, sinais corrigidos, variaveis ambientais, eventos abruptos e impacto no ExtraTrees.

## Resultado executivo

- C15 e C16 identicas em todas as colunas verificadas: `False`.
- Baseline recalculada no mesmo split: accuracy `0.9083`; balanced accuracy `0.9073`.
- Baseline do teste completo na auditoria: balanced accuracy `0.9073`.
- Maior associacao ambiental residual nas coletas criticas: `C28` / `MQ3` x `Soil_indice_0_1` = `-0.963`.

## Decisao por coleta

- `C15/C16`: duplicacao resolvida; a C16 foi substituida e agora possui sinais diferentes da C15.
- `C17`: quarentena. Ruido multissensor extremo sem variacao ambiental proporcional.
- `C24`: revisar canal MQ138 e log operacional; nao remover automaticamente antes de conferir a origem.
- `C28`: quarentena provisoria. Ha degrau multissensor e forte instabilidade em Soil.
- `C29-C31`: manter para auditoria biologica, mas repetir modelagem sem Soil e com validacao por coleta/dia.
- `C32`: quarentena. Ruido multissensor extremo sem correspondencia ambiental.

## Eventos e ambiente

- `C15`: 12 eventos nos MQ crus; 12 nos corrigidos; 3 linhas multissensor; amplitudes ambiente Soil=0.058, Temp=2.40 C, Pres=0.030 kPa.
- `C16`: 0 eventos nos MQ crus; 0 nos corrigidos; 0 linhas multissensor; amplitudes ambiente Soil=0.071, Temp=5.10 C, Pres=0.030 kPa.
- `C17`: 1492 eventos nos MQ crus; 1491 nos corrigidos; 362 linhas multissensor; amplitudes ambiente Soil=0.051, Temp=2.60 C, Pres=0.020 kPa.
- `C24`: 57 eventos nos MQ crus; 57 nos corrigidos; 0 linhas multissensor; amplitudes ambiente Soil=0.057, Temp=4.60 C, Pres=0.050 kPa.
- `C28`: 13 eventos nos MQ crus; 13 nos corrigidos; 3 linhas multissensor; amplitudes ambiente Soil=0.244, Temp=6.90 C, Pres=0.040 kPa.
- `C29`: 0 eventos nos MQ crus; 0 nos corrigidos; 0 linhas multissensor; amplitudes ambiente Soil=0.310, Temp=9.40 C, Pres=0.020 kPa.
- `C30`: 0 eventos nos MQ crus; 0 nos corrigidos; 0 linhas multissensor; amplitudes ambiente Soil=0.106, Temp=5.20 C, Pres=0.030 kPa.
- `C31`: 0 eventos nos MQ crus; 0 nos corrigidos; 0 linhas multissensor; amplitudes ambiente Soil=0.279, Temp=5.30 C, Pres=0.030 kPa.
- `C32`: 269 eventos nos MQ crus; 269 nos corrigidos; 86 linhas multissensor; amplitudes ambiente Soil=0.092, Temp=5.00 C, Pres=0.020 kPa.

## Coletas criticas presentes no teste original

- `C24`: accuracy por linha `1.0000` em `2121` linhas.
- `C28`: accuracy por linha `1.0000` em `2130` linhas.
- `C30`: accuracy por linha `1.0000` em `1546` linhas.
- `C32`: accuracy por linha `0.4131` em `1271` linhas.

C24, C28 e C30 tiveram 100% de acerto por linha no teste original. Como possuem artefatos ou forte assinatura ambiental, elas podem tornar o teste artificialmente facil. C32 teve desempenho inferior e adiciona ruido ao teste.

## Sensibilidade ao remover coletas do treino

- `baseline_recalculada`: accuracy `0.9083`; balanced accuracy `0.9073`; treino `41377` linhas.
- `sem_C16`: accuracy `0.9312`; balanced accuracy `0.9290`; treino `39936` linhas.
- `sem_C17_ruidosa`: accuracy `0.8944`; balanced accuracy `0.8921`; treino `40116` linhas.
- `sem_C29_C31_soil_treino`: accuracy `0.7613`; balanced accuracy `0.7664`; treino `37181` linhas.
- `sem_C15_C16`: accuracy `0.9069`; balanced accuracy `0.9040`; treino `38614` linhas.
- `sem_criticas_do_treino`: accuracy `0.9319`; balanced accuracy `0.9293`; treino `34479` linhas.

C16 nao e mais duplicada; sua remocao agora serve apenas como teste de sensibilidade e nao como recomendacao de limpeza. Retirar C17 testa se seu ruido prejudica o treino. Retirar C29 e C31 mede a dependencia das coletas saudaveis do dia 13 presentes no treino, enquanto C28 e C30, tambem do dia 13, estao no teste. Isso mostra dependencia de representacao por dia/condicao e exige validacao deixando um dia inteiro de fora.

## Ablacao ambiental com compensacao recalculada

- `corrigido_mais_ambiente`: accuracy `0.9083`; balanced accuracy `0.9073`.
- `recompensado_sem_soil`: accuracy `0.8395`; balanced accuracy `0.8376`.
- `recompensado_sem_temp`: accuracy `0.8877`; balanced accuracy `0.8845`.
- `recompensado_sem_pressao`: accuracy `0.9502`; balanced accuracy `0.9503`.
- `somente_mq_corrigido`: accuracy `0.8648`; balanced accuracy `0.8637`.
- `somente_mq_cru`: accuracy `0.8986`; balanced accuracy `0.8990`.

A variante recompensada sem pressao chegou a `0.9502` neste split fixo, contra `0.9083` na baseline atual. Isso nao deve ser anunciado como novo resultado final antes de validacao por coleta e por dia. Sem Soil, a accuracy foi `0.8395`; sem temperatura, `0.8877`.

## Efeito das coletas perfeitas no teste

Ao retirar C24, C28 e C30 apenas do calculo do teste, a accuracy cai de `0.9083` para `0.8757`. Essas tres coletas acrescentam aproximadamente `3.26` pontos percentuais a accuracy por linha.

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
