# Lembrete: conjunto de sensores

- Valor mencionado para lembrar: **91,36%**. Nao encontrei esse valor nos resultados da analise combinatoria por coleta; falta confirmar a origem dele.
- Resultado verificado nesta analise: **91,67% de acuracia balanceada por coleta no teste**, equivalente a **11 de 12 coletas corretas**.
- Conjunto associado ao resultado verificado: **MQ2 + MQ135 + MQ138 + sensor de solo (`Soil_indice_0_1`) + temperatura (`Temp_C`) + pressao (`Pres_kPa`)**.
- Validacao cruzada por coleta no treino para esse conjunto: **74,13%**.
- Regra de escolha: entre as configuracoes com 11/12 coletas corretas no teste, foi a de maior acuracia balanceada na validacao cruzada do treino.

Fontes locais: `exploracao_teste_126_combinacoes.csv`, `validacao_cruzada_126_combinacoes.csv` e `predicoes_candidato_exploratorio.csv`.
