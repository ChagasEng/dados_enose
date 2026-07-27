# Rascunho para complementação do referencial teórico

> **Nota editorial — não inserir na dissertação:** este arquivo propõe a reescrita da
> Seção 3.4.2 e a inclusão das seções seguintes. As citações utilizam comandos
> LaTeX e correspondem ao arquivo `referencias_sugeridas_modelos.bib`.
>
> **Guia editorial das imagens:** os blocos “Sugestão de imagem” indicam onde cada
> figura se relaciona com o texto. Eles servem para a montagem da dissertação e devem
> ser retirados da versão final. As figuras propostas neste arquivo são esquemas
> conceituais elaborados para explicar os fundamentos apresentados no Referencial
> Teórico e não representam resultados experimentais.

## 3.4.2 Pré-processamento

O pré-processamento tem como finalidade reduzir componentes indesejadas dos sinais,
atenuar diferenças de escala e preservar as variações relacionadas à composição
volátil das amostras. Essa etapa é particularmente importante em matrizes de sensores
MOS, pois suas respostas podem ser influenciadas por ruído eletrônico, deriva,
temperatura, umidade, condições de amostragem e diferenças de sensibilidade entre os
elementos da matriz \cite{wilson2009applications,lee2025machine}.

Entre os procedimentos aplicáveis aos sinais de um nariz eletrônico estão a
filtragem passa-baixa, a suavização por média móvel, a correção da linha de base, a
remoção ou sinalização de observações anômalas e a normalização. A média móvel
substitui cada observação por uma medida de tendência local calculada em uma janela
temporal, reduzindo oscilações rápidas. Entretanto, janelas excessivamente amplas
podem atenuar transientes relevantes, como o início da resposta e a recuperação do
sensor. Por essa razão, a intensidade da suavização deve ser compatível com a taxa de
aquisição e com a dinâmica do ensaio.

A correção da linha de base busca expressar a resposta em relação a um estado de
referência, reduzindo diferenças de nível inicial entre sensores ou ciclos de
medição. Dependendo da natureza do sinal, podem ser utilizadas diferenças absolutas,
variações relativas ou razões envolvendo o valor de referência. Razões entre sensores
também podem ser construídas como atributos derivados para enfatizar respostas
relativas da matriz. Embora possam reduzir efeitos comuns a vários canais, essas
razões exigem cautela quando o denominador assume valores pequenos, pois podem
amplificar o ruído.

A normalização é necessária quando os atributos apresentam amplitudes e dispersões
distintas. A padronização pelo escore z centraliza os dados na média e os redimensiona
pelo desvio-padrão. A transformação mínimo–máximo mapeia os valores para um
intervalo previamente definido, enquanto a escala robusta utiliza estatísticas menos
sensíveis a valores extremos, como a mediana e a amplitude interquartil. Há ainda
transformações de potência, como a família de Yeo–Johnson, que pode reduzir
assimetrias mesmo quando há valores nulos ou negativos
\cite{yeo2000newfamily}. Transformações quantílicas, por sua vez, utilizam a
distribuição empírica para aproximar os atributos de uma distribuição de referência.
O método deve ser ajustado exclusivamente com os dados de treinamento e depois
aplicado, sem novo ajuste, aos conjuntos de validação e teste, evitando a transferência
indevida de informação entre as etapas.

A detecção de anomalias pode considerar mudanças abruptas, desvios em relação a
estatísticas locais ou afastamentos de uma faixa operacional. Em sistemas com
circulação forçada de ar, variações de pressão podem indicar abertura da câmara,
acionamento da bomba, alterações de vedação ou outros eventos físicos. A remoção
desses intervalos deve seguir critérios reproduzíveis e ser documentada, pois um
procedimento de filtragem também pode eliminar variações biologicamente relevantes
se for definido depois da observação dos rótulos.

A Figura \ref{fig:preprocessamento_sinais_conceitual} organiza as principais etapas
do pré-processamento, desde o sinal bruto até a obtenção de uma representação em
escala comparável. O esquema destaca a correção da linha de base, a filtragem, a
suavização e a normalização como operações destinadas a reduzir variações
indesejadas sem eliminar a dinâmica relevante da resposta sensorial.

> **SUGESTÃO DE IMAGEM:** etapas conceituais do pré-processamento de sinais.
>
> Caminho:
> `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\figuras_referencial\preprocessamento_sinais_conceitual.png`

Copiar de:
`C:\Users\Matheus Bastos Chaga\Documents\dados_enose\figuras_referencial\preprocessamento_sinais_conceitual.png`

Nome na pasta `imagens`:
`preprocessamento_sinais_conceitual.png`

```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{imagens/preprocessamento_sinais_conceitual.png}
    \caption{Representação conceitual das principais etapas do pré-processamento de sinais de sensores. Fonte: Elaborado pelo autor (2026).}
    \label{fig:preprocessamento_sinais_conceitual}
\end{figure}
```
## 3.4.3 Compensação de interferências ambientais

A resposta de sensores MOS não depende apenas dos gases presentes, mas também das
condições ambientais e operacionais. Temperatura, umidade e características do fluxo
gasoso podem modificar os processos de adsorção e reação na superfície sensível,
alterando a resistência elétrica do sensor \cite{wilson2009applications}. Em
experimentos com amostras de solo, variáveis como temperatura, pressão e teor de água
do substrato também podem afetar a liberação, o transporte e a concentração dos
compostos voláteis. Consequentemente, diferenças ambientais podem ser confundidas
com diferenças biológicas se não forem monitoradas.

Os datasheets dos sensores MQ2, MQ3, MQ7, MQ8, MQ135 e MQ138 apresentam condições
padronizadas de temperatura e umidade e curvas que relacionam a razão de resistência
dos sensores às variações ambientais
\cite{winsen2021mq2,winsen2018mq3b,winsen2021mq7b,winsen2021mq8,winsen2015mq135,winsen2015mq138}.
Essas informações demonstram que a resposta de cada modelo MQ possui dependência
ambiental própria e, portanto, deve ser tratada individualmente.

No presente trabalho, a compensação foi realizada separadamente para cada sensor MQ
e orientada pelas dependências ambientais descritas em seu respectivo datasheet.
Para cada sensor, uma regressão robusta de Huber estimou a parcela do sinal associada
à umidade do solo, à temperatura e à pressão; essa parcela estimada foi então
subtraída da resposta bruta. A variável \texttt{Soil} utilizada nessa compensação corresponde
a um índice de umidade do solo obtido pela leitura analógica do Capacitive Soil
Moisture Sensor V2.0, que realiza o sensoriamento do substrato por princípio
capacitivo; portanto, essa variável não representa a umidade relativa do ar
\cite{rajguru_soil_v20}. A temperatura e a pressão, por sua vez, foram fornecidas
pelo BMP280, que mede essas duas grandezas, mas não a umidade relativa do ar
\cite{bosch2021bmp280}. A regressão foi ajustada somente com os dados de treinamento,
evitando vazamento de informações. A função de perda de Huber é aproximadamente
quadrática para resíduos pequenos e linear para resíduos grandes, reduzindo a
influência de observações muito afastadas sem descartá-las
\cite{huber1964robust}.


## 3.4.4 Construção e seleção de atributos

Após o pré-processamento, os sinais podem ser representados por atributos extraídos
no domínio do tempo, tais como nível de estado estacionário, variação máxima, tempo
de resposta, inclinação, área sob a curva e tempo de recuperação. Também podem ser
empregadas as leituras simultâneas dos sensores e variáveis derivadas, como diferenças
e razões entre canais. O objetivo é produzir uma representação numérica que preserve
informações discriminantes e reduza redundâncias da série temporal
\cite{lee2025machine}.

A seleção de atributos busca identificar quais variáveis contribuem para a separação
das classes. O teste qui-quadrado avalia a associação entre variáveis não negativas e
o alvo categórico, enquanto o V de Cramér normaliza a estatística qui-quadrado para
expressar a intensidade da associação. A informação mútua mede a redução de
incerteza sobre uma variável proporcionada pelo conhecimento de outra, permitindo
captar relações que não precisam ser lineares \cite{shannon1948mathematical}.

Modelos baseados em árvores fornecem importâncias associadas à redução de impureza
obtida pelos atributos ao longo das divisões. Essa medida é computacionalmente
eficiente, porém pode favorecer variáveis com determinadas distribuições ou muitos
valores possíveis. A importância por permutação oferece uma avaliação complementar:
os valores de um atributo são embaralhados e mede-se a redução do desempenho
preditivo. Quanto maior a perda, maior a dependência do modelo em relação ao
atributo \cite{breiman2001random}. Em atributos correlacionados, contudo, a
informação de uma variável pode ser parcialmente substituída por outra, reduzindo a
importância individual observada.

Quando o número de sensores é pequeno, também é possível comparar 
subconjuntos de atributos. Essa abordagem verifica combinações de forma direta, mas
o número de possibilidades cresce exponencialmente. Por isso, a seleção deve ocorrer
dentro do processo de validação, sem utilizar o conjunto de teste na escolha do
subconjunto final.

A Figura \ref{fig:construcao_selecao_atributos_conceitual} representa a transformação
dos sinais temporais em características numéricas e a posterior seleção das variáveis
mais informativas para a etapa de reconhecimento de padrões.

> **SUGESTÃO DE IMAGEM:** fluxo conceitual de construção e seleção de atributos.
>
> Caminho:
> `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\figuras_referencial\construcao_selecao_atributos_conceitual.png`

Copiar de:
`C:\Users\Matheus Bastos Chaga\Documents\dados_enose\figuras_referencial\construcao_selecao_atributos_conceitual.png`

Nome na pasta `imagens`:
`construcao_selecao_atributos_conceitual.png`

```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{imagens/construcao_selecao_atributos_conceitual.png}
    \caption{Fluxo conceitual de extração e seleção de atributos em sinais de nariz eletrônico. Fonte: Elaborado pelo autor (2026).}
    \label{fig:construcao_selecao_atributos_conceitual}
\end{figure}
```
## 3.4.5 Redução de dimensionalidade

A análise de componentes principais (PCA) transforma os atributos originais em
componentes ortogonais ordenados segundo a variância explicada. Trata-se de um
método não supervisionado: os rótulos das classes não participam da construção dos
componentes. A PCA pode facilitar a visualização de assinaturas sensoriais, reduzir
colinearidade e condensar a informação, embora as direções de maior variância não
sejam necessariamente as mais discriminantes \cite{hotelling1933analysis}.

A análise discriminante linear (LDA) utiliza os rótulos e procura projeções que
aumentem a separação entre classes em relação à dispersão interna de cada grupo
\cite{fisher1936use}. Em um problema com \(C\) classes, a projeção discriminante
possui no máximo \(C-1\) componentes. A LDA pode atuar como classificador ou como
técnica supervisionada de redução de dimensionalidade. Como utiliza informação do
alvo, seu ajuste também deve ocorrer exclusivamente nos dados de treinamento de cada
partição de validação.


```
## 3.5 APRENDIZADO DE MÁQUINA APLICADO A NARIZES ELETRÔNICOS

O reconhecimento de padrões em narizes eletrônicos consiste em relacionar a resposta
conjunta da matriz sensorial a categorias ou estruturas presentes nas amostras. Os
algoritmos podem ser divididos em métodos supervisionados, nos quais cada amostra de
treinamento possui um rótulo conhecido, e não supervisionados, que investigam a
organização dos dados sem utilizar rótulos durante o ajuste. Diferentes famílias de
modelos impõem hipóteses distintas sobre a distribuição, a geometria e a
complexidade das relações entre sensores e classes. Por isso, a comparação entre
algoritmos é relevante em dados de e-nose, cujas respostas são multivariadas,
correlacionadas e frequentemente não lineares \cite{lee2025machine}.

### 3.5.1 Aprendizado supervisionado e não supervisionado

No aprendizado supervisionado, o conjunto de treinamento contém pares formados por
atributos e rótulos. O modelo estima uma função capaz de classificar observações ainda
não apresentadas. Em tarefas binárias, essa função pode produzir diretamente uma
classe ou uma pontuação interpretada como probabilidade. A capacidade de
generalização deve ser avaliada em amostras independentes do ajuste.

No aprendizado não supervisionado, os algoritmos identificam agrupamentos,
gradientes ou observações atípicas sem conhecer previamente as classes. Em um nariz
eletrônico, essa abordagem permite verificar se as assinaturas sensoriais formam
estruturas compatíveis com as condições experimentais. A concordância posterior
entre agrupamentos e rótulos conhecidos pode ser analisada, mas os rótulos não devem
orientar a formação dos grupos.

A Figura \ref{fig:aprendizado_supervisionado_nao_supervisionado} diferencia os dois
paradigmas de aprendizado. Nos métodos supervisionados, rótulos conhecidos orientam
o treinamento e a predição; nos métodos não supervisionados, a organização dos dados
é investigada sem o uso prévio das classes.

> **SUGESTÃO DE IMAGEM:** distinção conceitual entre aprendizado supervisionado e não supervisionado.
>
> Caminho:
> `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\figuras_referencial\aprendizado_supervisionado_nao_supervisionado.png`

Copiar de:
`C:\Users\Matheus Bastos Chaga\Documents\dados_enose\figuras_referencial\aprendizado_supervisionado_nao_supervisionado.png`

Nome na pasta `imagens`:
`aprendizado_supervisionado_nao_supervisionado.png`

```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{imagens/aprendizado_supervisionado_nao_supervisionado.png}
    \caption{Distinção conceitual entre aprendizado supervisionado e não supervisionado. Fonte: Elaborado pelo autor (2026).}
    \label{fig:aprendizado_supervisionado_nao_supervisionado}
\end{figure}
```
### 3.5.2 Modelos baseados em árvores

Árvores de decisão particionam recursivamente o espaço de atributos por meio de
regras. Embora sejam interpretáveis, árvores isoladas podem apresentar elevada
variância. Métodos de conjunto combinam diversas árvores para produzir previsões mais
estáveis.

O Random Forest treina árvores sobre amostras obtidas por reamostragem e considera
subconjuntos aleatórios de atributos em cada divisão. A combinação entre árvores
reduz a correlação dos erros e melhora a generalização em relação a uma única árvore
\cite{breiman2001random}. O método aceita relações não lineares, interações e
atributos em escalas distintas, características úteis para respostas de matrizes
sensoriais.

O Extra Trees, ou Extremely Randomized Trees, também combina múltiplas árvores, mas
introduz aleatoriedade adicional na escolha dos pontos de corte. Em sua formulação,
limiares candidatos são gerados aleatoriamente e a melhor divisão entre eles é
selecionada. Essa aleatorização pode reduzir a variância e o custo computacional, ao
preço de possível aumento do viés \cite{geurts2006extremely}. A diferença entre
Random Forest e Extra Trees está, portanto, principalmente no grau de aleatoriedade
utilizado na construção das árvores e na estratégia de amostragem.

### 3.5.3 Métodos de boosting

O boosting constrói modelos sequencialmente, de modo que cada novo estimador procure
corrigir erros ou reduzir a função de perda deixada pelo conjunto anterior. O
gradient boosting interpreta essa construção como uma otimização no espaço de
funções, adicionando árvores que aproximam o gradiente negativo da perda
\cite{friedman2001greedy}.

O XGBoost é uma implementação escalável de gradient boosting com árvores que inclui
regularização, encolhimento das contribuições, amostragem de atributos e tratamento
eficiente da construção das árvores \cite{chen2016xgboost}. Esses mecanismos
permitem controlar a complexidade do modelo e representar interações não lineares.

O HistGradientBoosting discretiza valores contínuos em intervalos, ou histogramas,
antes da busca por divisões. Essa aproximação reduz o custo de avaliar pontos de
corte e torna o treinamento eficiente em bases extensas. Assim como outros métodos de
boosting, requer controle de profundidade, taxa de aprendizagem, número de iterações
e regularização para limitar o sobreajuste.

A Figura \ref{fig:conjuntos_arvores_conceitual} compara as estratégias de diversidade
empregadas pelo Random Forest, pelo Extra Trees e pelos métodos de boosting. Nos dois
primeiros, múltiplas árvores são combinadas com diferentes níveis de aleatoriedade;
no boosting, as árvores são adicionadas sequencialmente para reduzir os erros do
conjunto anterior.

> **SUGESTÃO DE IMAGEM:** comparação conceitual das estratégias de conjuntos de árvores.
>
> Caminho:
> `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\figuras_referencial\conjuntos_arvores_conceitual.png`

Copiar de:
`C:\Users\Matheus Bastos Chaga\Documents\dados_enose\figuras_referencial\conjuntos_arvores_conceitual.png`

Nome na pasta `imagens`:
`conjuntos_arvores_conceitual.png`

```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{imagens/conjuntos_arvores_conceitual.png}
    \caption{Comparação conceitual entre Random Forest, Extra Trees e métodos de boosting. Fonte: Elaborado pelo autor (2026).}
    \label{fig:conjuntos_arvores_conceitual}
\end{figure}
```
### 3.5.4 Modelos probabilísticos e lineares

O Naive Bayes aplica o teorema de Bayes e assume independência condicional entre os
atributos dada a classe. Apesar de essa hipótese raramente ser satisfeita integralmente
em matrizes de sensores, o método pode apresentar bom desempenho e constitui uma
referência probabilística de baixa complexidade \cite{hand2001idiots}. No Gaussian
Naive Bayes, cada atributo contínuo é modelado por uma distribuição normal dentro de
cada classe. O Multinomial Naive Bayes utiliza distribuições multinomiais e requer
atributos não negativos. O Complement Naive Bayes estima pesos a partir do
complemento de cada classe e foi proposto para reduzir limitações do modelo
multinomial, especialmente em situações de desequilíbrio
\cite{rennie2003tackling}. A adequação dessas variantes a sinais contínuos deve ser
verificada empiricamente e acompanhada das transformações necessárias.

A regressão logística modela o logaritmo da razão de chances como combinação linear
dos atributos e converte o resultado em probabilidade por meio da função logística
\cite{cox1958regression}. Seus coeficientes permitem examinar direção e intensidade
das associações, desde que se considerem a escala e a colinearidade. Como fronteira
linear, também funciona como referência para verificar se modelos mais complexos
produzem ganho efetivo.

As máquinas de vetores de suporte (SVM) procuram uma fronteira que maximize a margem
entre as classes. Apenas observações próximas dessa fronteira, denominadas vetores de
suporte, determinam diretamente a solução \cite{cortes1995support}. A SVM linear é
adequada quando se deseja uma separação linear regularizada. Versões com funções
kernel podem representar fronteiras não lineares. Como o cálculo de distâncias e
margens depende da magnitude dos atributos, a normalização é especialmente
importante.

O método dos \(k\) vizinhos mais próximos (KNN) classifica uma observação conforme os
rótulos das amostras mais próximas no espaço de atributos
\cite{cover1967nearest}. A votação pode atribuir maior peso aos vizinhos de menor
distância. O KNN não estima uma função paramétrica durante o treinamento, mas sua
previsão é sensível à escala, à escolha da distância, ao número de vizinhos, à
presença de atributos irrelevantes e à densidade desigual das regiões do espaço.

A Figura \ref{fig:classificadores_conceitual} reúne representações dos princípios do
Naive Bayes, da regressão logística, da SVM e do KNN. Os esquemas destacam,
respectivamente, a modelagem probabilística, a conversão de uma combinação linear em
probabilidade, a maximização da margem e a votação baseada em vizinhança.

> **SUGESTÃO DE IMAGEM:** princípios conceituais dos classificadores probabilísticos, lineares e baseados em distância.
>
> Caminho:
> `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\figuras_referencial\classificadores_conceitual.png`

Copiar de:
`C:\Users\Matheus Bastos Chaga\Documents\dados_enose\figuras_referencial\classificadores_conceitual.png`

Nome na pasta `imagens`:
`classificadores_conceitual.png`

```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{imagens/classificadores_conceitual.png}
    \caption{Representação conceitual dos princípios do Naive Bayes, da regressão logística, da SVM e do KNN. Fonte: Elaborado pelo autor (2026).}
    \label{fig:classificadores_conceitual}
\end{figure}
```
### 3.5.5 Redes neurais artificiais

O perceptron multicamadas (MLP) é uma rede neural feedforward formada por uma camada
de entrada, uma ou mais camadas ocultas e uma camada de saída. Cada unidade combina
as entradas por pesos e aplica uma função de ativação não linear. Durante o
treinamento, o erro é propagado no sentido inverso para atualizar os pesos por
otimização numérica \cite{rumelhart1986learning}.

As camadas ocultas permitem ao MLP aproximar relações não lineares e interações
complexas entre sensores. Entretanto, o desempenho depende da arquitetura, das
funções de ativação, da regularização, da taxa de aprendizagem e do critério de
parada. A padronização dos atributos favorece a estabilidade da otimização. Em bases
com amostras correlacionadas, a avaliação deve impedir que registros do mesmo ciclo
experimental sejam distribuídos entre treinamento e teste.

A Figura \ref{fig:arquitetura_mlp_conceitual} representa uma arquitetura de MLP com
camada de entrada, duas camadas ocultas e camada de saída. As conexões simbolizam os
pesos ajustados durante o treinamento, enquanto o fluxo inverso representa a
retropropagação do erro.

> **SUGESTÃO DE IMAGEM:** arquitetura conceitual de um perceptron multicamadas.
>
> Caminho:
> `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\figuras_referencial\arquitetura_mlp_conceitual.png`

Copiar de:
`C:\Users\Matheus Bastos Chaga\Documents\dados_enose\figuras_referencial\arquitetura_mlp_conceitual.png`

Nome na pasta `imagens`:
`arquitetura_mlp_conceitual.png`

```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{imagens/arquitetura_mlp_conceitual.png}
    \caption{Arquitetura conceitual de um perceptron multicamadas e do processo de retropropagação do erro. Fonte: Elaborado pelo autor (2026).}
    \label{fig:arquitetura_mlp_conceitual}
\end{figure}
```
### 3.5.6 Métodos de agrupamento

O K-means particiona as observações em \(k\) grupos, minimizando a soma das distâncias
quadráticas entre cada observação e o centroide do grupo ao qual foi atribuída
\cite{lloyd1982least}. O método é simples e eficiente, mas exige a escolha prévia de
\(k\), é sensível à inicialização e à escala e tende a favorecer grupos
aproximadamente convexos e de dispersão semelhante.

O DBSCAN define agrupamentos como regiões de alta densidade separadas por regiões
menos densas. O algoritmo utiliza uma distância de vizinhança e um número mínimo de
observações para distinguir pontos centrais, pontos de borda e ruído. Dessa forma,
pode encontrar grupos de formatos irregulares e não exige a definição prévia da
quantidade de agrupamentos \cite{ester1996density}. Entretanto, é sensível à escala e
aos parâmetros de densidade, especialmente quando os grupos apresentam densidades
muito distintas.

A Figura \ref{fig:kmeans_dbscan_conceitual} compara a atribuição de observações aos
centroides realizada pelo K-means com a identificação de regiões densas e pontos de
ruído efetuada pelo DBSCAN.

> **SUGESTÃO DE IMAGEM:** diferença conceitual entre K-means e DBSCAN.
>
> Caminho:
> `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\figuras_referencial\kmeans_dbscan_conceitual.png`

Copiar de:
`C:\Users\Matheus Bastos Chaga\Documents\dados_enose\figuras_referencial\kmeans_dbscan_conceitual.png`

Nome na pasta `imagens`:
`kmeans_dbscan_conceitual.png`

```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{imagens/kmeans_dbscan_conceitual.png}
    \caption{Comparação conceitual entre o agrupamento por centroides do K-means e o agrupamento por densidade do DBSCAN. Fonte: Elaborado pelo autor (2026).}
    \label{fig:kmeans_dbscan_conceitual}
\end{figure}
```
### 3.5.7 Interpretação, análises estatísticas e seleção do modelo

A comparação dos modelos deve considerar desempenho preditivo, estabilidade entre
partições, complexidade e possibilidade de interpretação. Importâncias nativas de
árvores, importância por permutação, informação mútua, associações por
qui-quadrado/V de Cramér e buscas de subconjuntos oferecem perspectivas
complementares sobre a contribuição dos sensores. Nenhuma dessas medidas, de forma
isolada, demonstra causalidade ou identifica compostos químicos específicos.

Análises de correlação também auxiliam no diagnóstico. O coeficiente de Pearson mede
associação linear; o coeficiente de Spearman utiliza postos e mede associação
monotônica, sendo menos dependente da forma da distribuição. Para comparar grupos
quando as condições paramétricas não são atendidas, o teste de Kruskal–Wallis utiliza
postos para avaliar diferenças de localização. Essas análises podem revelar
redundâncias ou associações ambientais, mas não substituem a avaliação do modelo em
dados independentes.

### 3.5.8 Validação e métricas de desempenho

A matriz de confusão organiza as previsões segundo classes reais e previstas. Em uma
classificação binária, ela permite calcular verdadeiros positivos, verdadeiros
negativos, falsos positivos e falsos negativos. A acurácia corresponde à proporção
total de previsões corretas. Quando as classes possuem tamanhos diferentes ou
importâncias semelhantes, a acurácia balanceada, definida como a média do recall das
classes, reduz a predominância da classe majoritária
\cite{brodersen2010balanced}.

A precisão representa a proporção de previsões positivas que estão corretas, enquanto
o recall ou sensibilidade representa a proporção de elementos de uma classe que foi
identificada. A medida F1 é a média harmônica entre precisão e recall. Na versão
macro, a F1 é calculada para cada classe e posteriormente promediada, atribuindo peso
igual às classes. Por isso, a apresentação conjunta de acurácia, acurácia balanceada,
F1 macro e recall por classe é mais informativa que o uso isolado da acurácia.

A curva ROC relaciona a taxa de verdadeiros positivos à taxa de falsos positivos em
diferentes limiares. A área sob a curva resume a capacidade de ordenação do
classificador sem fixar um único limiar \cite{fawcett2006introduction}. Quando o
limiar de decisão é ajustado para equilibrar erros ou priorizar uma classe, ele deve
ser escolhido em dados de validação e avaliado posteriormente em dados não utilizados
nessa escolha.

Em dados obtidos ao longo do tempo dentro de uma mesma coleta, linhas consecutivas
não constituem observações plenamente independentes. Uma divisão aleatória por linha
pode colocar sinais quase idênticos em treinamento e teste, produzindo estimativas
otimistas. A separação por grupos mantém todos os registros de uma coleta em uma
única partição. Na validação cruzada estratificada por grupos, preserva-se, tanto
quanto possível, a proporção das classes sem romper os grupos. Estratégias em blocos
são recomendadas quando há dependência temporal, espacial ou hierárquica
\cite{roberts2017crossvalidation}.

Quando a seleção de atributos, o ajuste de hiperparâmetros ou a escolha do limiar
ocorrem durante a validação, uma estrutura aninhada oferece uma estimativa mais
conservadora. Os ciclos internos realizam as escolhas, enquanto os ciclos externos
avaliam o procedimento completo. Balanceamento por pesos ou reamostragem também deve
ser realizado apenas nos dados de treinamento de cada ciclo.

Para métodos não supervisionados, a inércia do K-means quantifica a dispersão interna
dos grupos, mas diminui à medida que \(k\) aumenta. O coeficiente de silhueta compara
a coesão de uma observação com seu grupo e a separação em relação ao grupo vizinho
\cite{rousseeuw1987silhouettes}. Quando rótulos externos estão disponíveis apenas
para avaliação, o índice de Rand ajustado mede a concordância entre duas partições e
corrige a semelhança esperada ao acaso \cite{hubert1985comparing}.

Em conjunto, esses procedimentos permitem avaliar não apenas o resultado médio, mas
também a estabilidade entre coletas, o comportamento em cada classe e a influência
das condições experimentais. Essa distinção é essencial para evitar que um
desempenho elevado em uma condição específica seja interpretado como desempenho
geral do sistema.

A Figura \ref{fig:validacao_metricas_conceitual} relaciona a separação por grupos à
avaliação do classificador. A manutenção de cada coleta em uma única partição reduz o
vazamento de informações, enquanto a matriz de confusão fornece os elementos usados
no cálculo das métricas de desempenho.

> **SUGESTÃO DE IMAGEM:** validação por grupos, matriz de confusão e métricas de desempenho.
>
> Caminho:
> `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\figuras_referencial\validacao_metricas_conceitual.png`

Copiar de:
`C:\Users\Matheus Bastos Chaga\Documents\dados_enose\figuras_referencial\validacao_metricas_conceitual.png`

Nome na pasta `imagens`:
`validacao_metricas_conceitual.png`

```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{imagens/validacao_metricas_conceitual.png}
    \caption{Representação conceitual da validação por grupos e da avaliação por matriz de confusão e métricas de desempenho. Fonte: Elaborado pelo autor (2026).}
    \label{fig:validacao_metricas_conceitual}
\end{figure}
```
