# Checklist para conversa com Artur

## Hardware e coleta

- Houve troca de fonte, cabo, notebook, microcontrolador ou regulador durante os ensaios?
- A alimentacao era constante? Qual tensao e corrente esperadas?
- Existe log de corrente, tensao, bateria/fonte ou queda de alimentacao?
- O sensor foi erguido/removido em quais momentos?
- A bomba, valvula ou fluxo de ar teve interrupcao?
- A caixa foi aberta, vedada novamente ou manipulada durante C13-C17 ou C28?
- Houve mudanca de silicone, vedacao, mangueira, furo, fluxo ou posicionamento?
- O tempo registrado em `Tempo` e confiavel para todos os ensaios?

## Banco de dados

- O que exatamente representa `V_ref_0`?
- `Soil` representa umidade do solo, umidade relativa ou outro canal?
- `Pres.` esta em qual unidade e onde o sensor fica fisicamente?
- Existem arquivos brutos adicionais com alimentacao/corrente?
- As abas `Pagina...` correspondem a quais coletas reais?

## Proximos ensaios

- Registrar tensao e corrente do microcontrolador.
- Registrar eventos fisicos durante a coleta.
- Marcar inicio/fim de exposicao, purga, abertura e troca de amostra.
- Criar flag manual de anomalia em tempo real.
