# 2. Filtragem de ruidos e anomalias fisicas

## Objetivo

Investigar trechos com ruido fisico e criar um corte automatico baseado em variacao abrupta de pressao antes de treinar novos modelos.

## C13 a C17

As coletas C13 a C17 pertencem ao recorte antes do dia 20 e estao ligadas a amostras com nematoide. A comparacao mostrou que C13 se comporta diferente de C14-C17 em alguns sensores, principalmente em `MQ135`.

Resumo observado:

- C13: `dia 19 - Soja Heterodera Vaso 6`.
- C14: `dia 19 - Soja Heterodera Vaso 7`.
- C15: `dia 19 - Soja Heterodera Vaso 8`.
- C16: `dia 19 - Soja Heterodera Vaso 9`.
- C17: `dia 19 - Soja Heterodera Vaso 1`.

A diferenca entre C13 e C14-C17 sugere que nao da para tratar essas coletas cegamente como equivalentes so por terem classe parecida. A hipotese fisica precisa ser conferida.

## C28

C28 corresponde a `dia 13 - Soja Saudavel Vaso 1`, sem nematoide. Ela foi separada porque apresentou comportamento anomalo, com variacao importante em `Pres.` e impacto simultaneo nos canais MQ.

## Corte por pressao

O algoritmo principal remove trechos onde `Pres.` varia abruptamente:

```text
abs(Pres[t] - Pres[t-1]) >= 0.1
```

Na versao estrita, alem da janela de corte, tambem sao removidos pontos fora de:

```text
mediana(Pres.) +/- 0.5
```

## Arquivos nesta pasta

- `analises/`: tabelas de C13-C17, C28 e eventos candidatos.
- `graficos/`: graficos das anomalias e antes/depois do corte por pressao.
- `scripts/`: scripts usados para detectar e aplicar cortes.
- `datasets_filtrados/`: datasets resultantes do corte por pressao.
- `documentos/`: explicacao tecnica do algoritmo e resumo C13-C17/C28.
- `modelagem/`: ExtraTrees, rede neural e importancia apos o corte por pressao.

## Decisao tomada ate aqui

Para o ExtraTrees foi usado o dataset estrito `antes_dia_20_pressao_filtrada_estrito.csv`, pois ele remove os trechos mais suspeitos de variacao fisica na pressao.

## Modelagem apos filtro

Usando somente sensores MQ:

- ExtraTrees accuracy: 89.96%.
- Rede neural accuracy: 79.12%.
- Feature mais importante no ExtraTrees: `MQ8`.
