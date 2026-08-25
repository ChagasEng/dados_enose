# 4.3 PROCESSAMENTO E PREPARAÇÃO DOS DADOS

Os dados obtidos pelo nariz eletrônico foram inicialmente armazenados em um arquivo no formato XLSX, no qual cada planilha correspondia a uma coleta experimental. Cada linha representava uma amostra da série temporal e continha os valores de tempo, umidade do solo (`Soil`), temperatura (`Temp.`), pressão (`Pres.`) e resposta dos seis sensores de gases empregados no protótipo (`MQ2`, `MQ3`, `MQ7`, `MQ8`, `MQ135` e `MQ138`), além do rótulo da condição fitossanitária. Durante a consolidação, foram acrescentadas as variáveis de identificação da coleta, do dia e do vaso. A classe 0 foi atribuída às amostras de soja inoculadas com *Heterodera glycines*, enquanto a classe 1 representou as plantas sadias.

Para a modelagem principal, foram consideradas as 36 coletas realizadas nos dias 11, 12, 13, 18 e 19, todas pertencentes ao mesmo protocolo experimental com controle de pressão. As coletas dos dias 20 e 21 foram mantidas separadamente, pois foram realizadas sob uma condição experimental distinta, sem o mesmo procedimento de pressurização, e sua inclusão poderia introduzir um fator de confusão na classificação. A identificação individual das coletas foi preservada durante todo o processamento, permitindo rastrear cada observação até o respectivo ensaio.

Inicialmente, foram removidos 15% dos registros do começo e 15% do final de cada série da base originalmente consolidada. Esse recorte teve como finalidade reduzir a influência dos transientes associados ao aquecimento e à estabilização inicial dos sensores, bem como do período final de dessaturação e recuperação do sistema. Em seguida, as colunas utilizadas foram convertidas para formato numérico e os registros sem valores válidos nas variáveis requeridas foram descartados. Durante a auditoria da base, verificou-se que a coleta referente ao dia 19, soja com *H. glycines*, vaso 9, havia sido incorporada com conteúdo inconsistente. Essa coleta foi substituída pelo registro correto, seu rótulo foi confirmado como classe 0 e todas as etapas posteriores foram novamente executadas. Após a consolidação e essa correção, a base submetida à filtragem por pressão continha 71.302 observações.

As séries também foram examinadas quanto à ocorrência de mudanças abruptas de pressão, indicativas de abertura da câmara, acionamento do sistema, perda de vedação ou outro transiente operacional. Para cada coleta, calculou-se a diferença absoluta entre duas leituras consecutivas de pressão, conforme a Equação (1):

$$
\Delta P_t = \left|P_t-P_{t-1}\right|.
$$

Foi considerado um evento de variação abrupta quando $\Delta P_t \geq 0{,}10$ kPa. Para evitar que a resposta transitória imediatamente anterior ou posterior ao evento permanecesse na base, foram excluídas 30 amostras antes e 30 amostras depois do ponto detectado, incluindo-se também a própria amostra do evento. Na versão estrita do procedimento, foram ainda removidas as observações cuja pressão se encontrava fora do intervalo definido pela mediana global mais ou menos 0,50 kPa. Como as janelas de eventos próximos podiam se sobrepor, cada linha foi contabilizada apenas uma vez na remoção. Esse procedimento excluiu 7.810 registros, equivalentes a 10,95% da base de entrada, e resultou em 63.492 observações distribuídas entre as 36 coletas. Desse total, 29.044 registros pertenciam à classe com nematoide e 34.448 à classe sadia.

Após a filtragem, as variáveis ambientais foram organizadas de acordo com a identificação do hardware. A temperatura e a pressão registradas pelo BMP280 foram mantidas, respectivamente, em graus Celsius e quilopascais, originando as variáveis `Temp_C` e `Pres_kPa`. A leitura do sensor capacitivo de umidade do solo foi transformada em um índice entre 0 e 1 por normalização mínimo-máximo, conforme a Equação (2):

$$
Soil_{0-1}=\frac{Soil-Soil_{\min}}{Soil_{\max}-Soil_{\min}}.
$$

Como os sensores da família MQ apresentam sensibilidade às condições ambientais e não havia medição de umidade relativa do ar, não foi aplicada uma calibração físico-química completa baseada na razão $R_s/R_0$. Em seu lugar, realizou-se uma compensação ambiental estatística. Para cada sensor MQ, ajustou-se um regressor robusto de Huber utilizando como preditores o índice de umidade do solo, a temperatura e a pressão. O ajuste foi realizado exclusivamente com as coletas destinadas ao treinamento. A componente ambiental estimada foi centralizada pela média das previsões no conjunto de treinamento e subtraída do sinal original, de acordo com a Equação (3):

$$
MQ_{s,t}^{corr}=MQ_{s,t}-\left[\widehat{MQ}_{s,t}^{amb}-\overline{\widehat{MQ}_{s}^{amb}}_{\,treino}\right],
$$

em que $MQ_{s,t}^{corr}$ é o valor corrigido do sensor $s$ no instante $t$, $\widehat{MQ}_{s,t}^{amb}$ é o valor estimado a partir das variáveis ambientais e $\overline{\widehat{MQ}_{s}^{amb}}_{\,treino}$ é a média dessas estimativas no conjunto de treinamento. Esse procedimento removeu a parcela linear associada às condições ambientais, preservando o nível médio do sinal observado no treinamento. Portanto, ele foi tratado como compensação estatística orientada pelas características dos sensores, e não como calibração física definitiva.

Para a avaliação supervisionada, a divisão dos dados foi realizada por coleta, e não por linhas individuais. Dentro de cada classe, as coletas foram embaralhadas com semente aleatória 42 e separadas em proporção nominal de 70% para treinamento e 30% para teste. Dessa forma, todas as amostras temporais de uma mesma coleta permaneceram em apenas um dos conjuntos, reduzindo o risco de vazamento de informação decorrente da elevada correlação entre observações consecutivas. Ao final, 24 coletas, correspondentes a 41.377 registros, foram destinadas ao treinamento, e 12 coletas, totalizando 22.115 registros, foram reservadas para o teste.

O conjunto preparado para a modelagem preservou as variáveis de identificação e os sinais brutos para fins de auditoria, além das seis respostas MQ compensadas e das três variáveis ambientais. A padronização empregada no processamento foi utilizada internamente apenas durante o ajuste dos regressores de Huber. Suavizações por médias ou medianas móveis e transformações em escore-z foram adotadas exclusivamente na elaboração dos gráficos exploratórios, não sendo fornecidas aos classificadores. Assim, as métricas de desempenho foram calculadas a partir das observações processadas, mas sem suavização temporal adicional.
