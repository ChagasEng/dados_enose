# Plano de trabalho: calibracao e pre-processamento

## Objetivo

Polir a base antes de treinar novos modelos, removendo falhas fisicas e efeitos ambientais evidentes. A prioridade e evitar conclusoes artificiais causadas por ruido, saturacao, queda de alimentacao, variacao de pressao, temperatura ou umidade.

## 1. Investigacao do hardware e banco de dados

1. Confirmar com o Artur se houve queda de alimentacao, mudanca de fonte, instabilidade no microcontrolador ou falha fisica durante os ensaios.
2. Verificar se os arquivos possuem informacao direta de corrente, tensao ou alimentacao.
3. Registrar quais colunas existem e quais faltam.
4. Definir para os proximos ensaios uma forma objetiva de marcar falhas fisicas: tensao de alimentacao, corrente, estado da bomba/valvula, tempo real e evento manual.

Resultado atual: nao ha coluna explicita de corrente/tensao/alimentacao. Existe `V_ref_0` em versoes antigas, mas os datasets atuais de modelagem nao carregam essa coluna.

## 2. Filtragem de ruidos e anomalias fisicas

1. Inspecionar C13-C17, principalmente C13 vs C14, porque sao coletas proximas e com mesma condicao de classe.
2. Inspecionar C28 porque ela foi apontada como ponto de pico anomalo.
3. Usar `Pres.` como gatilho primario para localizar falhas fisicas: quedas/subidas bruscas de pressao podem indicar abertura, movimentacao, vazamento, troca fisica ou instabilidade.
4. Criar um corte automatico candidato em torno desses eventos e validar visualmente antes de remover definitivamente.

Arquivos gerados:

- `analises/mapa_coletas_C13_C17_C28.csv`
- `analises/estatisticas_C13_C17_C28.csv`
- `analises/candidatos_corte_por_pressao.csv`
- `graficos_base/analise_C13_C17_antes_dia_20.png`
- `graficos_base/analise_C28_antes_dia_20.png`

## 3. Compensacao de temperatura e umidade

As leituras MQ dependem de temperatura e umidade/ambiente. Portanto, comparar sensores sem compensacao pode induzir interpretacao errada.

Abordagem 1: usar `Soil`, `Temp.` e `Pres.` como entradas junto com os sensores MQ no modelo. O modelo aprende a compensar o ambiente, mas pode tambem aprender vies experimental.

Abordagem 2: aplicar correcao matematica antes do modelo, usando curvas de sensibilidade dos datasheets. Essa abordagem e mais interpretavel, mas depende de calibracao correta e de uma variavel de umidade confiavel.

## 4. Polimento inicial da base

Antes de extrair features como area sob a curva:

1. Remover ou sinalizar trechos com falha de pressao.
2. Verificar saturacoes e saltos abruptos por sensor.
3. Padronizar as coletas e preservar o mapa C1, C2, C3...
4. So depois extrair features por coleta.
5. Validar sempre por grupo/coleta, evitando split aleatorio por linha.

## Produto esperado

Uma base limpa e rastreavel, com:

- coluna de status de falha/corte;
- intervalos removidos documentados;
- compensacao ambiental testada;
- graficos antes/depois do filtro;
- modelo treinado apenas apos o polimento.
