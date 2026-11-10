# Análise do parecer de Materiais e Métodos

## Prioridade para o artigo

1. **LOSO sem excluir coletas como referência principal.** O arquivo `analise_combinatoria_sensores/busca_mq_ambiente_loso_leituras/predicoes_vencedor_cru.csv` já contém previsões LOSO para as 36 coletas e os dez vasos, usando MQ2, MQ3, MQ8, MQ135 e MQ138 mais as três variáveis ambientais. Nesse cenário, há 55.916/63.492 acertos por leitura (88,07%); acurácia balanceada 88,32% e F1 macro 88,04%. Não é necessário afirmar que a validação não foi feita: ela existe. O vencedor foi selecionado entre 63 conjuntos de MQ usando os mesmos vasos, então seu desempenho continua exploratório como estimativa para vasos novos.
2. **Cenário com exclusões como sensibilidade.** Para a mesma configuração e modelo, após retirar cinco coletas, há 54.783/55.946 acertos (97,92%) e acurácia balanceada 97,88%. A diferença entre 88,07% e 97,92% é grande e deve aparecer claramente no artigo. Das cinco coletas excluídas, duas são da classe inoculada (2.702 leituras) e três da classe controle (4.844 leituras). Quatro foram escolhidas após observar erros do próprio modelo. A regra de saltos multissensor também foi definida depois de inspecionar os dados e pode sinalizar eventos reais, não apenas falhas instrumentais.
3. **A busca de 511 conjuntos não produz uma validação independente.** Extra Trees foi comparado nas 511 combinações e o vencedor foi escolhido e avaliado com os mesmos dez vasos após exclusões. O desempenho do vencedor é uma avaliação exploratória; não serve, sozinho, para afirmar superioridade generalizável dos sensores escolhidos. O cenário apenas ambiental é importante como controle da interpretação: na base filtrada, solo, temperatura e pressão sozinhos alcançaram 53.865/55.946 acertos (96,28%).
4. **A unidade independente é o vaso.** O split histórico 70/30 separa coletas, mas oito dos dez vasos aparecem nos dois lados. Use esse split como comparação histórica. As métricas por leitura, coleta e vaso e os intervalos por reamostragem de vasos sobre previsões LOSO salvas já foram calculados em `analise_combinatoria_sensores/avaliacao_parecer_base_completa/`; isso não corrige o viés da escolha de sensores nos mesmos vasos. Um teste de permutação por vaso exige refazer o treino com rótulos permutados por vaso. Seleção com validação aninhada ou novos vasos seriam mais fortes.

## Correções factuais no parecer e no texto

- O parecer elogia a compensação de MQ "dentro de cada rodada LOSO". **Isso não ocorreu na busca LOSO atual:** os scripts `buscar_nove_variaveis_loso.py` e `buscar_mq_loso_sem_quatro_coletas.py` usam colunas `MQ2` etc., ou seja, sinais MQ brutos. O `HuberRegressor` pertence à modelagem principal 70/30 descrita no texto. Não alegar compensação nos folds LOSO sem executar outra análise.
- A modelagem principal descrita no texto usa **seis MQ**. A comparação fixa de Random Forest e TCN no LOSO usa **cinco MQ**, e a busca de Extra Trees inclui todos os subconjuntos dos seis. Essa diferença deve ser explicitada, não tratada como omissão acidental do MQ7.
- `ExtraTreesClassifier` é **Extra Trees**, não Random Forest. O nome X-TCN também deve ser revisto, pois a implementação usada não contém o mecanismo de atenção ou explicabilidade citado no texto. Uma nota no fim da seção não substitui a correção das tabelas e legendas.
- A chave `Classe + Vaso` evita colisão no agrupamento dos dados derivados, mas registros inoculados aparecem com código numérico de vaso 1, embora a numeração física informada seja 6 a 10. A origem desse erro de identificador precisa ser rastreada e corrigida antes da publicação.
- O mínimo e o máximo do índice de solo, a faixa global de pressão e os percentis de salto foram calculados com a base inteira. Isso deve ser declarado como pré-processamento global. Para avaliar o efeito, seria preciso refazer a análise com parâmetros ajustados apenas nas coletas de treinamento de cada fold.

## Informações que o código não permite inventar

Confirmar com o protocolo e o caderno do experimento: randomização e ordem das medições; aplicação de veículo/placebo ao controle; cultivar, substrato, raça/HG do nematoide e confirmação individual de infestação; biometria; estado de alimentação e aquecimento dos MQ entre coletas; vazão; branco/ar de referência; unidade elétrica do sinal bruto; localização física do BMP280; taxa de amostragem e duração real das fases. Se alguma etapa não ocorreu ou não foi registrada, relatá-la como limitação.

## Sequência recomendada de revisão

1. Apresentar a validação LOSO na base completa antes do cenário pós-exclusão e manter a análise 70/30 como comparação histórica.
2. Incluir métricas por leitura, coleta e vaso, com intervalos de confiança por reamostragem de vasos quando adequados, sem tratar leituras correlacionadas como réplicas independentes.
3. Mostrar controle apenas ambiental e, se disponível, comparação de MQ brutos e compensados sob o mesmo protocolo. O ranking das 511 combinações pela acurácia balanceada foi calculado em `analise_combinatoria_sensores/busca_nove_variaveis_loso/ranking_511_acuracia_balanceada.csv` como análise de sensibilidade.
4. Padronizar nomes dos modelos e corrigir a rastreabilidade do vaso 10 no arquivo derivado.
5. Adicionar as informações experimentais verificadas; registrar as ausentes como limitações.

Os números deste documento foram conferidos nos arquivos salvos em `analise_combinatoria_sensores/`; nenhuma nova validação aninhada, permutação ou reanálise de pré-processamento foi executada aqui.
