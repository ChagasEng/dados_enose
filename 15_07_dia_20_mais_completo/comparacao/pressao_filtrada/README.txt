Corte por variacao de pressao

Limiar usado: delta absoluto de Pres. >= 0.1
Janela removida: 30 linhas antes e 30 linhas depois de cada evento.
Versao estrita: tambem remove Pres. fora de mediana +/- 0.5.

Arquivos principais:
- antes_dia_20_pressao_filtrada.csv
- dia_20_mais_pressao_filtrada.csv
- eventos_variacao_pressao.csv
- resumo_corte_por_pressao.csv
- *_pressao_antes_depois.png
- *_curvas_apos_corte_pressao.png

Observacao: este corte remove variacoes abruptas de pressao, preservando pequenas oscilacoes normais.
