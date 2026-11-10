# Dados a confirmar antes da submissão

O arquivo `materiais_metodos_artigo_pos_parecer.tex` relata somente procedimentos verificáveis na seção enviada e nos arquivos de análise. Os itens abaixo precisam de consulta ao protocolo experimental, firmware ou caderno de campo. Não devem ser preenchidos por inferência.

Informações confirmadas pelo pesquisador nesta conversa: as coletas foram realizadas em 11, 12, 13, 18 e 19 de **maio de 2026**; a primeira ocorreu aproximadamente 35 dias após a inoculação. Houve **irrigação entre 13 e 18 de maio**, associada à mudança entre as faixas operacionais de solo seco e úmido. O experimento ocorreu na Estação de Pesquisas Agronema, Estrada do Carvoeiro, km 1,5, bairro Carvoeiro, distrito de Ribeirão Bonito, Grandes Rios, Paraná. Os vasos 1 a 5 eram controles não inoculados; os vasos 6 a 10 receberam cerca de 100 mil J2 de *H. glycines* por vaso.

O fluxograma enviado pelo pesquisador esclarece a operação: o ESP32 desligava a bomba após a depressurização até a referência próxima de 70 kPa e abria a válvula de entrada; a pressão interna subia para cerca de 94–95 kPa. A **válvula permanecia aberta** durante cinco minutos de aquisição e cinco de análise das respostas, com a bomba de extração desligada. Depois a válvula era fechada e a bomba expulsava o ar durante três minutos de desaturação. A descrição anterior de câmara fechada durante toda a leitura foi corrigida. O diagrama não documenta fluxo forçado contínuo nessa etapa.

1. Cultivar de soja, composição do substrato, tamanho dos vasos e condições de cultivo.
2. Origem e identificação da população de *Heterodera glycines* (raça/HG, se determinada); preparo do inóculo; eventual aplicação aos controles do mesmo veículo sem J2. A ausência de J2 nos controles foi confirmada, mas a existência de placebo não foi informada.
3. Não há confirmação de sorteio da posição dos vasos ou da ordem de aquisição nem registro da sequência de medição em cada dia. O pesquisador optou por não detalhar esse ponto. Não afirmar que houve randomização. A possibilidade de confusão por posição ou ordem de medição deve ser considerada na interpretação.
4. Confirmação individual da infestação, se realizada: cistos, ovos, J2, biomassa ou outro padrão de referência.
5. Alimentação e pré-aquecimento dos MQ entre coletas; operação elétrica específica do MQ7; unidades e transformação do sinal MQ armazenado (ADC, tensão ou outra); vazão da bomba; duração de purga; branco ou ar de referência.
6. Posição física do BMP280 e disponibilidade de medida de umidade relativa na câmara. O BMP280 registrado fornece temperatura e pressão, não umidade relativa.
7. Motivo experimental de usar somente o protocolo a vácuo na análise; versões do firmware e software a relatar.
8. Rastreabilidade do vaso 10: `sem pressao/dataset_sem_pressao.csv` chama três coletas inoculadas dos dias 12, 18 e 19 de `Vaso 10`, mas seu campo numérico `Vaso` contém `1`. O dataset derivado `06_07_melhor_modelo/dados/dataset_melhor_modelo_sensores_corrigidos.csv` também apresenta esses registros como vaso 1 inoculado. A chave `Classe + Vaso` os mantém separados do vaso 1 controle, porém a planilha derivada e os artefatos que dependem do nome da coleta devem ser reconciliados antes de publicação.

## Análises recomendadas que ainda não foram executadas neste ajuste

- Teste de permutação dos rótulos **por vaso**, com novo treinamento em cada permutação.
- Antes de interpretar esse teste como inferência causal, confirmar como foi feita a alocação dos dez vasos aos grupos. Para corrigir a seleção de sensores e as exclusões pós-erro, seria necessário repetir todo esse procedimento em cada permutação; permutar apenas as previsões salvas não produz o teste solicitado.
- Busca aninhada de sensores dentro dos folds LOSO, caso a intenção seja estimar desempenho após seleção automática das entradas.
- Repetição de toda a modelagem com normalização do índice de solo e limiar de pressão calculados apenas no treino de cada fold.
- Análise de sensibilidade por fase temporal real e avaliação da fração removida pelo filtro de pressão em cada classe.
- Comparação de MQ brutos e corrigidos com Huber no mesmo protocolo LOSO, com ajuste do Huber apenas no treino de cada fold.

As métricas por leitura, coleta e vaso e os intervalos bootstrap por vaso da base completa foram calculados em `analise_combinatoria_sensores/avaliacao_parecer_base_completa/` pelo script `analise_combinatoria_sensores/avaliar_base_completa_parecer.py`. O bootstrap mantém fixos o modelo e a escolha das entradas; não corrige o viés de seleção do subconjunto de sensores.

A acurácia balanceada das 511 combinações já avaliadas foi calculada sem novo treino pelo script `analise_combinatoria_sensores/reanalisar_ranking_balanceado.py`; o ranking está em `analise_combinatoria_sensores/busca_nove_variaveis_loso/ranking_511_acuracia_balanceada.csv`. A escolha original havia usado acurácia simples por leitura; o ranking balanceado é uma análise de sensibilidade com os mesmos vasos.

O LOSO dos controles fixos de ET com nove entradas, só seis MQ e só três variáveis ambientais, além da regressão logística com nove entradas, foi executado em `analise_combinatoria_sensores/baselines_fixos_base_completa/`. Esse diretório também contém a tabela solo × classe × dia e a proporção removida pelo filtro de pressão por classe. A comparação de AUC por coleta entre ET, RF e TCN nas 31 coletas está em `analise_combinatoria_sensores/comparacao_auc_modelos_filtrada/`. Os intervalos binomiais exatos por vaso estão em `analise_combinatoria_sensores/avaliacao_parecer_base_completa/intervalos_exatos_por_vaso.csv`.
