# Lembrete: conjunto de sensores

- Valor mencionado para lembrar: **91,36%**. Nao encontrei esse valor nos resultados da analise combinatoria por coleta; falta confirmar a origem dele.
- Resultado verificado nesta analise: **91,67% de acuracia balanceada por coleta no teste**, equivalente a **11 de 12 coletas corretas**.
- Conjunto associado ao resultado verificado: **MQ2 + MQ135 + MQ138 + sensor de solo (`Soil_indice_0_1`) + temperatura (`Temp_C`) + pressao (`Pres_kPa`)**.
- Validacao cruzada por coleta no treino para esse conjunto: **74,13%**.
- Regra de escolha: entre as configuracoes com 11/12 coletas corretas no teste, foi a de maior acuracia balanceada na validacao cruzada do treino.

Fontes locais: `exploracao_teste_126_combinacoes.csv`, `validacao_cruzada_126_combinacoes.csv` e `predicoes_candidato_exploratorio.csv`.

## Atualizacao apos a orientacao sobre leituras

O resultado 91,67% acima usa **uma decisao por coleta**. Para a validacao que conta acertos e erros de cada leitura dentro de uma coleta, o mesmo conjunto MQ2 + MQ135 + MQ138 + solo + temperatura + pressao obteve **55.549/63.492 = 87,49%** com dados crus no LOSO por vaso e **13.959/15.861 = 88,01%** ao treinar e testar medias de quatro leituras. Consulte `validacao_por_leitura/README.md`.

Depois de refazer a **busca completa** com acuracia por leitura crua, o vencedor para o artigo mudou: **MQ2 + MQ3 + MQ8 + MQ135 + MQ138 + ambiente** obteve **55.916/63.492 = 88,07%** no LOSO. A media de quatro foi apenas um exemplo metodologico e nao integra a conclusao principal. Estes sao resultados exploratorios de LOSO nos mesmos dez vasos. Consulte `busca_mq_ambiente_loso_leituras/README.md`.

No **70/30 por leitura crua** para o mesmo conjunto, foram **18.643/19.048 = 97,87%**; sem ambiente, os mesmos cinco MQ obtiveram **18.586/19.048 = 97,57%**. Todas as coletas e todos os vasos aparecem em treino e teste nesse split por leitura. Consulte `validacao_70_30_por_leitura/README.md`.
