# Algoritmo candidato de corte por pressao

## Ideia

Usar a coluna `Pres.` para detectar mudancas fisicas abruptas. Quando a diferenca absoluta de pressao entre duas linhas consecutivas ultrapassa um limiar robusto, o algoritmo marca uma janela ao redor do evento como candidata a corte.

## Limiar

Para cada coleta:

```text
delta_pressao = abs(Pres[t] - Pres[t-1])
limiar = max(percentil_99_5(delta_pressao), mediana(delta_pressao) + 8 * MAD(delta_pressao))
```

## Janela de corte

No prototipo atual:

```text
30 linhas antes do evento + linha do evento + 30 linhas depois
```

Isso gera uma janela de 61 linhas por evento candidato.

## Saida

Arquivo gerado:

```text
analises/candidatos_corte_por_pressao.csv
```

Cada linha contem:

- dataset;
- coleta;
- dia/vaso/classe;
- tempo do evento;
- delta de pressao;
- linhas inicial/final sugeridas para corte;
- tempo inicial/final do corte.

## Observacao

Este algoritmo ainda e candidato. Ele deve ser revisado visualmente antes de remover dados da base final.
