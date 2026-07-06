# 1. Investigacao do hardware e banco de dados

## Objetivo

Confirmar se a base possui informacao suficiente para identificar falha de alimentacao, queda de corrente, tensao instavel ou interferencia fisica capaz de afetar o microcontrolador e os sensores MQ.

## O que foi encontrado

A planilha original `coletas.xlsx` possui as colunas:

`Tempo`, `Soil`, `Temp.`, `Pres.`, `V_ref_0`, `MQ2`, `MQ3`, `MQ7`, `MQ8`, `MQ135`, `MQ138`, `Classe`.

Nos datasets atuais de modelagem, a base usada possui:

`Coleta`, `Dia`, `Vaso`, `Tempo`, `Soil`, `Temp.`, `Pres.`, `MQ2`, `MQ3`, `MQ7`, `MQ8`, `MQ135`, `MQ138`, `Classe`.

Nao foi encontrada coluna direta de corrente, tensao de alimentacao, bateria, fonte, estado do microcontrolador, bomba, valvula ou evento fisico manual. A coluna `V_ref_0` existe em versoes antigas/originais, mas nao substitui uma medicao direta de corrente ou tensao de alimentacao.

## Arquivos nesta pasta

- `analises/inventario_colunas_hardware.csv`: inventario das colunas por arquivo.
- `documentos/checklist_conversa_artur.md`: perguntas para levantar com Artur.
- `modelagem/`: baseline antes do corte, com ExtraTrees, rede neural e importancia.

## Conclusao pratica

Com a base atual da para suspeitar de falha fisica por efeitos indiretos, principalmente `Pres.` e saltos simultaneos nos MQ, mas nao da para afirmar queda de alimentacao eletrica sem log de corrente/tensao. Para os proximos ensaios, o ideal e registrar tensao, corrente, estado de bomba/valvula e eventos manuais.

## Modelagem baseline

Foi rodado um baseline antes do corte por pressao usando MQ + ambiente. Resultado:

- ExtraTrees accuracy: 87.58%.
- Rede neural accuracy: 82.74%.
- Feature mais importante no ExtraTrees: `MQ8`.
