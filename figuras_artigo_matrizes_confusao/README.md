# Matrizes de confusão para o artigo

As figuras foram calculadas **por leitura** no conjunto de teste. Em cada matriz, as linhas são as classes reais e as colunas, as previstas: classe 0 = com nematoide; classe 1 = sem nematoide. O número em cada célula é a contagem de leituras, e o percentual usa como denominador o total da classe real (linha).

| Condição | Modelo | Real 0 → Prev. 0 | Real 0 → Prev. 1 | Real 1 → Prev. 0 | Real 1 → Prev. 1 |
|---|---|---:|---:|---:|---:|
| Solo seco | X-TCN | 6.251 | 256 | 0 | 10.185 |
| Solo seco | Random Forest (rótulo no artigo) | 5.864 | 643 | 0 | 10.185 |
| Solo úmido | X-TCN | 3.448 | 704 | 1.271 | 0 |
| Solo úmido | Random Forest (rótulo no artigo) | 3.513 | 639 | 746 | 525 |

## Onde inserir

- **Seção 4.2, após o parágrafo que termina com “recall de 90,12% para a classe com nematoide” e antes da seção 4.3:** inserir `matrizes_solo_seco.pdf` (ou PNG). Legenda sugerida: **Figura 6. Matrizes de confusão no teste em solo seco: (a) X-TCN e (b) Random Forest. Linhas: classes reais; colunas: classes previstas. Percentuais relativos à classe real. Fonte: Elaborado pelos autores (2026).**
- **Seção 4.3, após o parágrafo que termina com “não se manteve estável no solo úmido” e antes da seção 4.4:** inserir `matrizes_solo_umido.pdf` (ou PNG). Legenda sugerida: **Figura 7. Matrizes de confusão no teste em solo úmido: (a) X-TCN e (b) Random Forest. Linhas: classes reais; colunas: classes previstas. Percentuais relativos à classe real. Fonte: Elaborado pelos autores (2026).**

Os percentuais reproduzem as Tabelas 2 e 3 do PDF. A pedido do autor, o painel (b) usa o rótulo **Random Forest** do artigo. A implementação e os resultados locais que geraram essas contagens identificam **Extra Trees**; conferir a identificação do modelo antes da submissão.

Fontes das contagens: `06_07_melhor_modelo/analise_seco_molhado/matriz_confusao_{seco,molhado}.csv` e `08_10_2026_xtcn_solo_medio/resultados_tabela22_xtcn_mesmas_features/predicoes_por_leitura.csv`. As imagens podem ser recriadas com `gerar_matrizes.py` usando o Python de `.venv`.
