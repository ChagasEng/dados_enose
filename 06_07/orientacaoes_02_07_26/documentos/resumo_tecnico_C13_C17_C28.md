# Resumo tecnico: C13-C17 e C28

## Mapa das coletas

No recorte antes do dia 20:

- C13: `dia 19 - Soja Heterodera Vaso 6`, com nematoide.
- C14: `dia 19 - Soja Heterodera Vaso 7`, com nematoide.
- C15: `dia 19 - Soja Heterodera Vaso 8`, com nematoide.
- C16: `dia 19 - Soja Heterodera Vaso 9`, com nematoide.
- C17: `dia 19 - Soja Heterodera Vaso 1`, com nematoide.
- C28: `dia 13 - Soja Saudavel Vaso 1`, sem nematoide.

## Achados iniciais

C13 se comporta de forma diferente de C14-C17 em alguns sensores. Em `MQ135`, por exemplo, C13 teve media aproximada de `10846`, enquanto C14 ficou em torno de `20113`, C15/C16 em torno de `18899`, e C17 em torno de `12285`. Isso confirma que ha discrepancia importante dentro do grupo C13-C17.

Em C14-C17 aparecem saltos grandes em `MQ2`, `MQ3` e `MQ135`, com deltas entre linhas acima de milhares de unidades em alguns pontos. Isso combina com a suspeita de ruido fisico ou mudanca abrupta de condicao.

C28 apresenta variacao forte em `Pres.` e tambem saltos relevantes em `MQ2`, mas nao mostra o mesmo padrao extremo de `MQ135` observado em C14-C17. Mesmo assim, C28 deve ser inspecionada porque foi apontada como coleta com pico anomalo.

## Proxima decisao

Nao remover ainda de forma definitiva. Primeiro usar `candidatos_corte_por_pressao.csv` para revisar visualmente os intervalos sugeridos e confirmar se os cortes coincidem com falhas fisicas reais.
