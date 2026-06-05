# Grafico do split treino/teste

Esta pasta mostra exatamente como o dataset usado pelo Extra Trees foi repartido.

Dataset original:

```text
../sem pressao/dataset_sem_pressao.csv
```

Regra usada:

- 70% treino e 30% teste.
- Separacao por grupos de `Coleta` dentro de cada `Classe`.
- Mesmo `random_state = 42` do script de treino.
- `Coleta` e usada somente para separar treino/teste; ela nao entra no modelo.

Arquivos:

- `dataset_com_split_treino_teste.csv`: dataset completo filtrado para o modelo, com a coluna `Conjunto` indicando `treino` ou `teste`.
- `coletas_split_treino_teste.csv`: lista de cada `Coleta`, classe, conjunto e quantidade de linhas.
- `resumo_split_treino_teste.csv`: resumo por classe e conjunto.
- `grafico_dataset_treino_teste.png`: painel com os recortes treino/teste, as leituras MQ normalizadas e a quantidade de linhas por classe.
- `gerar_grafico_split_treino_teste.py`: script para recriar todos os arquivos desta pasta.
