# Orientacoes 02/07/2026

Esta pasta consolida os arquivos para a etapa de calibracao dos sensores e pre-processamento antes de novos modelos.

## Estrutura

- `graficos_base/`: graficos ja usados para comparar coletas numeradas e coletas com/sem nematoide.
- `analises/`: tabelas geradas a partir dos datasets atuais.
- `documentos/`: plano de trabalho, checklist para conversa com Artur e resumo tecnico.
- `scripts/`: prototipos reproduziveis para corte por pressao e proximas rotinas.

## Arquivos principais

- `documentos/plano_trabalho_calibracao_preprocessamento.md`
- `documentos/checklist_conversa_artur.md`
- `documentos/resumo_tecnico_C13_C17_C28.md`
- `documentos/protocolo_compensacao_temperatura_umidade.md`
- `analises/inventario_colunas_hardware.csv`
- `analises/mapa_coletas_C13_C17_C28.csv`
- `analises/estatisticas_C13_C17_C28.csv`
- `analises/candidatos_corte_por_pressao.csv`

## Estado atual

A base atual nao possui coluna explicita de corrente, tensao ou alimentacao do microcontrolador. A coluna `V_ref_0` existe no `coletas.xlsx` e no dataset antigo `dataset_processado/dataset_unico_filtrado.csv`, mas nao substitui uma medicao direta de falha de alimentacao.

As coletas C13-C17 do recorte antes do dia 20 sao todas `Com nematoide`; C28 e `Sem nematoide`. Os primeiros resultados indicam mudancas fortes em `Pres.`, `MQ2`, `MQ3` e `MQ135` em pontos especificos, o que justifica implementar o corte por pressao antes de extrair features.
