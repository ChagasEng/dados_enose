# Diagnostico C13-C17 e C28

## Arquivos gerados

- `graficos/coletas_por_nematoide_atualizado_estilo_original.png`
- `graficos/diagnostico_visual_C13_C17_C28.png`
- `graficos/ranking_ruido_por_coleta.png`
- `analise_C13_C17_C28/estatisticas_por_coleta.csv`
- `analise_C13_C17_C28/eventos_ruido_C13_C17_C28.csv`
- `analise_C13_C17_C28/duplicatas_exatas_coletas.csv`

## Mapa

- `C13`: dia 19 - Soja Heterodera Vaso 6, com nematoide.
- `C14`: dia 19 - Soja Heterodera Vaso 7, com nematoide.
- `C15`: dia 19 - Soja Heterodera Vaso 8, com nematoide.
- `C16`: dia 19 - Soja Heterodera Vaso 9, com nematoide.
- `C17`: dia 19 - Soja Heterodera Vaso 1, com nematoide.
- `C28`: dia 13 - Soja Saudavel Vaso 1, sem nematoide.

## C28

C28 nao foi a coleta mais ruidosa pelo criterio de saltos abruptos. Ela teve `14` eventos acima do limiar global, contra `1495` em C17. O que chama atencao em C28 e o tamanho dos degraus: `MQ2` teve amplitude de `8631` e maior salto de `6911`; `MQ138` teve amplitude de `3919` e maior salto de `3265`. A pressao em C28 ficou relativamente estavel, com amplitude de `0.04 kPa`.

Leitura provavel: C28 parece mais uma mudanca brusca de regime/saturacao/reacomodacao dos MQ do que ruido continuo causado por pressao. Como a pressao interna ja foi filtrada e ficou estavel, a causa mais provavel esta em transiente de gas, memoria/saturacao de sensor, fluxo interno, troca/manuseio ou efeito de contaminacao/residuo na camara.

## C13-C17

As coletas C13-C17 sao todas do dia 19 com nematoide, mas nao se comportam como repeticoes consistentes.

Nenhuma duplicata exata foi detectada.

`C17` e o ponto mais critico: ela teve `1495` saltos abruptos em Pres.+MQ, sendo `1491` nos MQ. Isso atingiu varios canais ao mesmo tempo (`MQ2`, `MQ3`, `MQ7`, `MQ8`, `MQ135` e `MQ138`), enquanto a pressao variou pouco (`0.02 kPa`).

Leitura provavel: C17 tem cara de instabilidade de aquisicao/sinal dos MQ, saturacao ou transiente eletrico/ADC, nao de variacao fisica de pressao.

## Acao recomendada

1. C15 e C16 agora possuem sinais diferentes; manter ambas e auditar cada uma separadamente.
2. Registrar a origem e a substituicao da C16 para manter a rastreabilidade.
3. Rodar outra versao removendo ou marcando C17 como coleta anomala.
4. Para C28, revisar o log experimental: troca de vaso, abertura/fechamento, tempo de estabilizacao, fluxo/bomba e possivel saturacao dos MQ.
5. Nao tratar esses pontos como resposta biologica sem validar a origem fisica/operacional.
