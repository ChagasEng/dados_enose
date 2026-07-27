# Rascunho para complementação do referencial teórico

> **Nota editorial — não inserir na dissertação:** este arquivo propõe a reescrita da
> Seção 3.4.2 e a inclusão das seções seguintes. As citações utilizam comandos
> LaTeX e correspondem ao arquivo `referencias_sugeridas_modelos.bib`.
> Parâmetros, divisões do conjunto de dados, valores de limiar e resultados obtidos
> foram deliberadamente reservados para os capítulos de metodologia e resultados.
>
> **Guia editorial das imagens:** os blocos “Sugestão de imagem” indicam onde cada
> figura se relaciona com o texto. Eles servem para a montagem da dissertação e devem
> ser retirados da versão final. Quando o gráfico apresenta dados do experimento, o
> bloco também informa que sua inserção definitiva deve ocorrer na Metodologia ou em
> Resultados, mesmo que o assunto seja explicado neste capítulo.

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

A Figura \ref{fig:mq7_cru_normalizado} apresenta a resposta do sensor MQ7 antes e
depois da normalização, permitindo observar a alteração da escala do sinal sem a
eliminação do seu comportamento relativo.

> **SUGESTÃO DE IMAGEM — inserir aqui, após o parágrafo sobre normalização:**
> comparação direta entre o sinal bruto e o sinal normalizado do sensor MQ7. É a
> figura existente que melhor ilustra visualmente o efeito da normalização.
>
> Caminho:
> `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\comparacao\cru_vs_normalizado\MQ7_cru_vs_normalizado.png`

Copiar de:
`C:\Users\Matheus Bastos Chaga\Documents\dados_enose\comparacao\cru_vs_normalizado\MQ7_cru_vs_normalizado.png`

Nome na pasta `imagens`:
`mq7_cru_normalizado.png`

```latex
A Figura \ref{fig:mq7_cru_normalizado} apresenta a comparação entre o sinal bruto e o sinal normalizado do sensor MQ7.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{imagens/mq7_cru_normalizado.png}
    \caption{Comparação entre os sinais bruto e normalizado do sensor MQ7. Fonte: Elaborado pelo autor (2026).}
    \label{fig:mq7_cru_normalizado}
\end{figure}
```
> **Destino recomendado na versão final:** Metodologia, na descrição do
> pré-processamento aplicado aos dados. No Referencial Teórico, pode ser mencionada
> apenas como exemplo elaborado pelo autor, caso o orientador concorde.

A detecção de anomalias pode considerar mudanças abruptas, desvios em relação a
estatísticas locais ou afastamentos de uma faixa operacional. Em sistemas com
circulação forçada de ar, variações de pressão podem indicar abertura da câmara,
acionamento da bomba, alterações de vedação ou outros eventos físicos. A remoção
desses intervalos deve seguir critérios reproduzíveis e ser documentada, pois um
procedimento de filtragem também pode eliminar variações biologicamente relevantes
se for definido depois da observação dos rótulos.

A Figura \ref{fig:pressao_antes_depois_filtragem} apresenta o comportamento do sinal
de pressão antes e depois da aplicação do critério de filtragem, possibilitando
visualizar os intervalos afetados pelo tratamento das anomalias.

> **SUGESTÃO DE IMAGEM — inserir aqui, após o parágrafo sobre anomalias:** pressão antes e depois
> da aplicação do corte para remoção de anomalias. A figura mostra de forma clara a
> consequência da filtragem.
>
> Caminho:
> `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\06_07\2_filtragem_ruidos_anomalias\graficos\antes_dia_20_estrito_pressao_antes_depois.png`

Copiar de:
`C:\Users\Matheus Bastos Chaga\Documents\dados_enose\06_07\2_filtragem_ruidos_anomalias\graficos\antes_dia_20_estrito_pressao_antes_depois.png`

Nome na pasta `imagens`:
`pressao_antes_depois_filtragem.png`

```latex
A Figura \ref{fig:pressao_antes_depois_filtragem} apresenta o comportamento do sinal de pressão antes e depois da filtragem de anomalias.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{imagens/pressao_antes_depois_filtragem.png}
    \caption{Comparação dos sinais de pressão antes e depois da filtragem de anomalias. Fonte: Elaborado pelo autor (2026).}
    \label{fig:pressao_antes_depois_filtragem}
\end{figure}
```
>
> **Destino recomendado na versão final:** Metodologia. Trata-se de uma demonstração
> do procedimento realmente aplicado, e não de uma figura teórica genérica.

## 3.4.3 Compensação de interferências ambientais

A resposta de sensores MOS não depende apenas dos gases presentes, mas também das
condições ambientais e operacionais. Temperatura, umidade e características do fluxo
gasoso podem modificar os processos de adsorção e reação na superfície sensível,
alterando a resistência elétrica do sensor \cite{wilson2009applications}. Em
experimentos com amostras de solo, variáveis como temperatura, pressão e teor de água
do substrato também podem afetar a liberação, o transporte e a concentração dos
compostos voláteis. Consequentemente, diferenças ambientais podem ser confundidas
com diferenças biológicas se não forem monitoradas.

Uma estratégia de compensação consiste em estimar a parcela da resposta sensorial
associada às variáveis ambientais e subtraí-la do sinal observado. Modelos lineares
expressam essa relação por coeficientes que representam a variação esperada do sensor
em função de cada variável explicativa. Quando há valores extremos, a regressão
robusta de Huber constitui uma alternativa aos mínimos quadrados. Sua função de perda
é aproximadamente quadrática para resíduos pequenos e linear para resíduos grandes,
reduzindo a influência de observações muito afastadas sem descartá-las
\cite{huber1964robust}. Para evitar vazamento de dados, tanto a compensação quanto
qualquer padronização associada devem ser estimadas somente no conjunto de
treinamento.

É necessário distinguir compensação estatística de calibração físico-química. A
primeira remove associações observadas entre o ambiente e o sinal dentro do domínio
experimental; a segunda requer modelo físico, condições controladas, gases de
referência e parâmetros compatíveis com as especificações dos sensores. Assim, uma
correção estatística pode melhorar a comparabilidade dos sinais, mas não substitui a
calibração instrumental nem garante seletividade molecular.

A Figura \ref{fig:correlacao_ambiente_correcao} apresenta as correlações das
variáveis ambientais com as respostas dos sensores antes e depois da compensação,
permitindo examinar as alterações produzidas pelo procedimento de correção.

> **SUGESTÃO DE IMAGEM — após este parágrafo:** mapas de correlação entre variáveis
> ambientais e sensores MQ antes e depois da correção. É a figura do projeto que
> melhor representa a compensação ambiental discutida nesta seção.
>
> Caminho:
> `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\06_07_melhor_modelo\graficos\correlacao_ambiente_antes_depois_correcao.png`

Copiar de:
`C:\Users\Matheus Bastos Chaga\Documents\dados_enose\06_07_melhor_modelo\graficos\correlacao_ambiente_antes_depois_correcao.png`

Nome na pasta `imagens`:
`correlacao_ambiente_antes_depois_correcao.png`

```latex
A Figura \ref{fig:correlacao_ambiente_correcao} apresenta as correlações das variáveis ambientais com as respostas dos sensores antes e depois da compensação.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{imagens/correlacao_ambiente_antes_depois_correcao.png}
    \caption{Correlação entre as variáveis ambientais e as respostas dos sensores antes e depois da compensação ambiental. Fonte: Elaborado pelo autor (2026).}
    \label{fig:correlacao_ambiente_correcao}
\end{figure}
```
>
> **Destino recomendado na versão final:** Metodologia ou Resultados, pois a figura
> apresenta o efeito observado nos dados do estudo. No Referencial, manter somente a
> explicação conceitual da compensação.

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

Quando o número de sensores é pequeno, também é possível comparar exaustivamente
subconjuntos de atributos. Essa abordagem verifica combinações de forma direta, mas
o número de possibilidades cresce exponencialmente. Por isso, a seleção deve ocorrer
dentro do processo de validação, sem utilizar o conjunto de teste na escolha do
subconjunto final.

A Figura \ref{fig:selecao_atributos_chi2} apresenta os valores obtidos na seleção de
atributos pelo teste qui-quadrado, possibilitando comparar a associação de cada
variável com as classes analisadas.

> **SUGESTÃO DE IMAGEM — após o parágrafo sobre qui-quadrado:** gráfico de seleção
> dos atributos pelo teste qui-quadrado. Ele se relaciona diretamente com os métodos
> descritos no segundo parágrafo desta seção.
>
> Caminho:
> `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\metodo_autor_henike\graficos\selecao_atributos_chi2.png`

Copiar de:
`C:\Users\Matheus Bastos Chaga\Documents\dados_enose\metodo_autor_henike\graficos\selecao_atributos_chi2.png`

Nome na pasta `imagens`:
`selecao_atributos_chi2.png`

```latex
A Figura \ref{fig:selecao_atributos_chi2} apresenta o resultado da seleção de atributos realizada pelo teste qui-quadrado.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\textwidth]{imagens/selecao_atributos_chi2.png}
    \caption{Seleção de atributos por meio do teste qui-quadrado. Fonte: Elaborado pelo autor (2026).}
    \label{fig:selecao_atributos_chi2}
\end{figure}
```
>
> **Destino recomendado na versão final:** Resultados, por apresentar o ranking
> encontrado no conjunto de dados.

A Figura \ref{fig:importancia_permutacao_melhor_modelo} apresenta a contribuição das
variáveis para o modelo selecionado, estimada pela redução do desempenho provocada
pela permutação individual de cada atributo.

> **SUGESTÃO DE IMAGEM — após o parágrafo sobre importância por permutação:** gráfico
> da importância das variáveis no melhor modelo. Esta imagem deve ser usada para
> interpretar o modelo que alcançou aproximadamente 95% de acurácia.
>
> Caminho:
> `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\06_07_melhor_modelo\importancia_sensores\grafico_importancia_permutacao_melhor_modelo.png`

Copiar de:
`C:\Users\Matheus Bastos Chaga\Documents\dados_enose\06_07_melhor_modelo\importancia_sensores\grafico_importancia_permutacao_melhor_modelo.png`

Nome na pasta `imagens`:
`importancia_permutacao_melhor_modelo.png`

```latex
A Figura \ref{fig:importancia_permutacao_melhor_modelo} apresenta a contribuição das variáveis estimada pela importância por permutação.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.90\textwidth]{imagens/importancia_permutacao_melhor_modelo.png}
    \caption{Importância das variáveis por permutação no melhor modelo de classificação. Fonte: Elaborado pelo autor (2026).}
    \label{fig:importancia_permutacao_melhor_modelo}
\end{figure}
```
>
> **Destino recomendado na versão final:** Resultados e Discussão. Não utilizar aqui
> como se fosse uma explicação genérica da técnica.

A Figura \ref{fig:ablacao_sensores} apresenta o efeito da retirada de sensores sobre
o desempenho preditivo e compara as melhores combinações de atributos identificadas
durante a análise de ablação.

> **IMAGEM COMPLEMENTAR PARA RESULTADOS:** gráfico da ablação de sensores e das
> melhores combinações de atributos.
>
> Caminho:
> `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\06_07_melhor_modelo\ablacao_sensores_sem_C16_treino\grafico_ablacao_e_melhores_combinacoes.png`

Copiar de:
`C:\Users\Matheus Bastos Chaga\Documents\dados_enose\06_07_melhor_modelo\ablacao_sensores_sem_C16_treino\grafico_ablacao_e_melhores_combinacoes.png`

Nome na pasta `imagens`:
`ablacao_sensores.png`

```latex
A Figura \ref{fig:ablacao_sensores} apresenta o efeito da retirada dos sensores e o desempenho das melhores combinações de atributos.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{imagens/ablacao_sensores.png}
    \caption{Análise de ablação dos sensores e desempenho das melhores combinações de atributos. Fonte: Elaborado pelo autor (2026).}
    \label{fig:ablacao_sensores}
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

A Figura \ref{fig:pca_2d_classes} apresenta a distribuição das observações no espaço
formado pelos dois primeiros componentes principais. A proporção da variância
representada por cada componente é apresentada na Figura
\ref{fig:pca_variancia_explicada}. A Figura \ref{fig:lda_1d_classes}, por sua vez,
apresenta a distribuição das classes na projeção supervisionada produzida pela LDA.

> **SUGESTÃO DE CONJUNTO DE IMAGENS — após esta seção:** utilizar as três figuras a
> seguir em sequência para mostrar (1) a projeção das classes, (2) a variância
> explicada pelos componentes e (3) a projeção supervisionada produzida pela LDA.
>
> PCA em duas dimensões:
> `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\metodo_autor_henike\graficos\pca_2d_classes.png`

Copiar de:
`C:\Users\Matheus Bastos Chaga\Documents\dados_enose\metodo_autor_henike\graficos\pca_2d_classes.png`

Nome na pasta `imagens`:
`pca_2d_classes.png`

```latex
A Figura \ref{fig:pca_2d_classes} apresenta a distribuição das classes no espaço formado pelos dois primeiros componentes principais.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\textwidth]{imagens/pca_2d_classes.png}
    \caption{Projeção bidimensional das classes obtida pela análise de componentes principais. Fonte: Elaborado pelo autor (2026).}
    \label{fig:pca_2d_classes}
\end{figure}
```
>
> Variância explicada pela PCA:
> `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\metodo_autor_henike\graficos\pca_variancia_explicada.png`

Copiar de:
`C:\Users\Matheus Bastos Chaga\Documents\dados_enose\metodo_autor_henike\graficos\pca_variancia_explicada.png`

Nome na pasta `imagens`:
`pca_variancia_explicada.png`

```latex
A Figura \ref{fig:pca_variancia_explicada} apresenta a proporção da variância explicada pelos componentes principais.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.80\textwidth]{imagens/pca_variancia_explicada.png}
    \caption{Proporção da variância explicada pelos componentes principais. Fonte: Elaborado pelo autor (2026).}
    \label{fig:pca_variancia_explicada}
\end{figure}
```
>
> LDA em uma dimensão:
> `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\metodo_autor_henike\graficos\lda_1d_classes.png`

Copiar de:
`C:\Users\Matheus Bastos Chaga\Documents\dados_enose\metodo_autor_henike\graficos\lda_1d_classes.png`

Nome na pasta `imagens`:
`lda_1d_classes.png`

```latex
A Figura \ref{fig:lda_1d_classes} apresenta a distribuição das classes na projeção discriminante obtida pela LDA.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\textwidth]{imagens/lda_1d_classes.png}
    \caption{Distribuição das classes na projeção discriminante produzida pela LDA. Fonte: Elaborado pelo autor (2026).}
    \label{fig:lda_1d_classes}
\end{figure}
```
>
> **Destino recomendado na versão final:** Resultados. Para evitar excesso de
> figuras, a PCA 2D e a LDA podem ser reunidas posteriormente em um único painel
> comparativo. No Referencial Teórico, o ideal é uma figura conceitual sem os
> resultados das classes do experimento.

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

A Figura \ref{fig:comparacao_modelos_acuracia_balanceada} apresenta a acurácia
balanceada obtida pelas combinações de transformação e classificador avaliadas,
possibilitando comparar o comportamento dos métodos supervisionados.

> **SUGESTÃO DE IMAGEM RELACIONADA:** comparação de desempenho entre diferentes
> combinações de transformação e modelo supervisionado. O gráfico demonstra que
> vários algoritmos foram avaliados, mas não explica conceitualmente a diferença
> entre aprendizado supervisionado e não supervisionado.
>
> Caminho:
> `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\metodo_autor_henike\graficos\comparacao_modelos_balanced_accuracy.png`

Copiar de:
`C:\Users\Matheus Bastos Chaga\Documents\dados_enose\metodo_autor_henike\graficos\comparacao_modelos_balanced_accuracy.png`

Nome na pasta `imagens`:
`comparacao_modelos_acuracia_balanceada.png`

```latex
A Figura \ref{fig:comparacao_modelos_acuracia_balanceada} apresenta a comparação da acurácia balanceada entre as combinações de transformação e modelo.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{imagens/comparacao_modelos_acuracia_balanceada.png}
    \caption{Comparação da acurácia balanceada obtida pelas diferentes combinações de transformação e modelo. Fonte: Elaborado pelo autor (2026).}
    \label{fig:comparacao_modelos_acuracia_balanceada}
\end{figure}
```
>
> **Destino recomendado na versão final:** Resultados, na comparação dos modelos.

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

A Figura \ref{fig:comparacao_metricas_extratrees} apresenta as métricas obtidas
pelas configurações avaliadas com o Extra Trees, permitindo comparar o efeito das
etapas de tratamento dos sinais sobre o desempenho do classificador.

> **SUGESTÃO DE IMAGEM RELACIONADA AOS MODELOS DE ÁRVORES:** gráfico comparativo das
> métricas das configurações de Extra Trees avaliadas no projeto.
>
> Caminho:
> `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\06_07_melhor_modelo\graficos\comparacao_metricas_extratrees.png`

Copiar de:
`C:\Users\Matheus Bastos Chaga\Documents\dados_enose\06_07_melhor_modelo\graficos\comparacao_metricas_extratrees.png`

Nome na pasta `imagens`:
`comparacao_metricas_extratrees.png`

```latex
A Figura \ref{fig:comparacao_metricas_extratrees} apresenta a comparação das métricas obtidas pelas configurações avaliadas com o Extra Trees.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{imagens/comparacao_metricas_extratrees.png}
    \caption{Comparação das métricas de desempenho das configurações avaliadas com o classificador Extra Trees. Fonte: Elaborado pelo autor (2026).}
    \label{fig:comparacao_metricas_extratrees}
\end{figure}
```
>
> **Destino recomendado na versão final:** Resultados. O projeto ainda não contém
> uma figura conceitual pronta que explique visualmente a construção de uma árvore,
> do Random Forest e do Extra Trees.

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

A Figura \ref{fig:matriz_confusao_xgboost} apresenta a distribuição das
classificações corretas e incorretas produzidas pelo XGBoost para as classes
consideradas no estudo.

> **SUGESTÃO DE IMAGEM RELACIONADA AO XGBoost:** matriz de confusão obtida na
> avaliação experimental do XGBoost.
>
> Caminho:
> `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\xboosty\resultados\matrizes\matriz_confusao_xgboost.png`

Copiar de:
`C:\Users\Matheus Bastos Chaga\Documents\dados_enose\xboosty\resultados\matrizes\matriz_confusao_xgboost.png`

Nome na pasta `imagens`:
`matriz_confusao_xgboost.png`

```latex
A Figura \ref{fig:matriz_confusao_xgboost} apresenta a distribuição das classificações corretas e incorretas produzidas pelo XGBoost.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.75\textwidth]{imagens/matriz_confusao_xgboost.png}
    \caption{Matriz de confusão obtida pelo classificador XGBoost. Fonte: Elaborado pelo autor (2026).}
    \label{fig:matriz_confusao_xgboost}
\end{figure}
```
>
> **Destino recomendado na versão final:** Resultados. Esta matriz não substitui uma
> figura conceitual sobre boosting, pois apresenta o desempenho do modelo nos dados.

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

As matrizes de confusão do Naive Bayes, da SVM linear e do KNN são apresentadas,
respectivamente, nas Figuras \ref{fig:matriz_confusao_naive_bayes},
\ref{fig:matriz_confusao_svm_linear} e \ref{fig:matriz_confusao_knn}, permitindo
comparar os padrões de acertos e erros produzidos por essas famílias de
classificadores.

> **SUGESTÕES DE IMAGENS RELACIONADAS A ESTA SEÇÃO:** as matrizes abaixo registram
> resultados de modelos probabilísticos e de modelos lineares/baseados em distância.
> Elas podem ser usadas posteriormente na comparação dos classificadores.
>
> Naive Bayes:
> `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\naive bayes\resultados\matrizes\matriz_confusao_naive_bayes.png`

Copiar de:
`C:\Users\Matheus Bastos Chaga\Documents\dados_enose\naive bayes\resultados\matrizes\matriz_confusao_naive_bayes.png`

Nome na pasta `imagens`:
`matriz_confusao_naive_bayes.png`

```latex
A Figura \ref{fig:matriz_confusao_naive_bayes} apresenta a matriz de confusão obtida pelo classificador Naive Bayes.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.75\textwidth]{imagens/matriz_confusao_naive_bayes.png}
    \caption{Matriz de confusão obtida pelo classificador Naive Bayes. Fonte: Elaborado pelo autor (2026).}
    \label{fig:matriz_confusao_naive_bayes}
\end{figure}
```
>
> SVM linear:
> `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\metodo_autor_henike\resultados\matrizes\matriz_confusao_minmax_svm_linear.png`

Copiar de:
`C:\Users\Matheus Bastos Chaga\Documents\dados_enose\metodo_autor_henike\resultados\matrizes\matriz_confusao_minmax_svm_linear.png`

Nome na pasta `imagens`:
`matriz_confusao_svm_linear.png`

```latex
A Figura \ref{fig:matriz_confusao_svm_linear} apresenta a matriz de confusão obtida pela máquina de vetores de suporte com kernel linear.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.75\textwidth]{imagens/matriz_confusao_svm_linear.png}
    \caption{Matriz de confusão obtida pela máquina de vetores de suporte com kernel linear. Fonte: Elaborado pelo autor (2026).}
    \label{fig:matriz_confusao_svm_linear}
\end{figure}
```
>
> KNN:
> `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\metodo_autor_henike\resultados\matrizes\matriz_confusao_minmax_knn.png`

Copiar de:
`C:\Users\Matheus Bastos Chaga\Documents\dados_enose\metodo_autor_henike\resultados\matrizes\matriz_confusao_minmax_knn.png`

Nome na pasta `imagens`:
`matriz_confusao_knn.png`

```latex
A Figura \ref{fig:matriz_confusao_knn} apresenta a matriz de confusão obtida pelo classificador dos k vizinhos mais próximos.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.75\textwidth]{imagens/matriz_confusao_knn.png}
    \caption{Matriz de confusão obtida pelo classificador dos \(k\) vizinhos mais próximos. Fonte: Elaborado pelo autor (2026).}
    \label{fig:matriz_confusao_knn}
\end{figure}
```
>
> **Destino recomendado na versão final:** Resultados. Para o Referencial, não há
> atualmente no projeto uma figura conceitual pronta sobre Naive Bayes, regressão
> logística, SVM e KNN.

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

A Figura \ref{fig:matriz_confusao_mlp} apresenta as classificações corretas e
incorretas produzidas pelo perceptron multicamadas para as duas classes avaliadas.

> **SUGESTÃO DE IMAGEM RELACIONADA À MLP:** matriz de confusão do perceptron
> multicamadas avaliado no projeto.
>
> Caminho:
> `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\multi layer perceptron\resultados\matrizes\matriz_confusao_mlp.png`

Copiar de:
`C:\Users\Matheus Bastos Chaga\Documents\dados_enose\multi layer perceptron\resultados\matrizes\matriz_confusao_mlp.png`

Nome na pasta `imagens`:
`matriz_confusao_mlp.png`

```latex
A Figura \ref{fig:matriz_confusao_mlp} apresenta a matriz de confusão obtida pelo perceptron multicamadas.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.75\textwidth]{imagens/matriz_confusao_mlp.png}
    \caption{Matriz de confusão obtida pelo perceptron multicamadas. Fonte: Elaborado pelo autor (2026).}
    \label{fig:matriz_confusao_mlp}
\end{figure}
```
>
> **Destino recomendado na versão final:** Resultados. O projeto ainda não contém um
> diagrama conceitual pronto com as camadas de entrada, camadas ocultas e camada de
> saída da rede.

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

Os critérios do cotovelo e da silhueta empregados na avaliação do número de grupos
são apresentados na Figura \ref{fig:kmeans_cotovelo_silhueta}, enquanto a Figura
\ref{fig:kmeans_pca} apresenta os agrupamentos produzidos pelo K-means no espaço da
PCA. Para o DBSCAN, a Figura \ref{fig:dbscan_kdistancia} apresenta a curva utilizada
na definição do parâmetro de vizinhança, e a Figura \ref{fig:dbscan_pca} apresenta os
agrupamentos e os pontos identificados como ruído.

> **SUGESTÃO DE CONJUNTO DE IMAGENS — após esta seção:** os gráficos abaixo formam um
> conjunto visual adequado para comparar o procedimento e o resultado do K-means com
> o DBSCAN.
>
> Escolha de \(k\) pelo cotovelo e pela silhueta:
> `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\nao supervisioando\kmeans\grafico_kmeans_cotovelo_silhouette.png`

Copiar de:
`C:\Users\Matheus Bastos Chaga\Documents\dados_enose\nao supervisioando\kmeans\grafico_kmeans_cotovelo_silhouette.png`

Nome na pasta `imagens`:
`kmeans_cotovelo_silhueta.png`

```latex
A Figura \ref{fig:kmeans_cotovelo_silhueta} apresenta os critérios do cotovelo e da silhueta empregados na avaliação do número de agrupamentos.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{imagens/kmeans_cotovelo_silhueta.png}
    \caption{Avaliação do número de agrupamentos do K-means pelos métodos do cotovelo e do coeficiente de silhueta. Fonte: Elaborado pelo autor (2026).}
    \label{fig:kmeans_cotovelo_silhueta}
\end{figure}
```
>
> Agrupamentos do K-means projetados pela PCA:
> `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\nao supervisioando\kmeans\grafico_kmeans_pca.png`

Copiar de:
`C:\Users\Matheus Bastos Chaga\Documents\dados_enose\nao supervisioando\kmeans\grafico_kmeans_pca.png`

Nome na pasta `imagens`:
`kmeans_pca.png`

```latex
A Figura \ref{fig:kmeans_pca} apresenta os agrupamentos produzidos pelo K-means e sua relação com as classes conhecidas.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{imagens/kmeans_pca.png}
    \caption{Comparação entre os agrupamentos obtidos pelo K-means e as classes reais na projeção da PCA. Fonte: Elaborado pelo autor (2026).}
    \label{fig:kmeans_pca}
\end{figure}
```
>
> Curva de k-distância utilizada na escolha de \(\varepsilon\) do DBSCAN:
> `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\nao supervisioando\dbscan\grafico_dbscan_kdistancia.png`

Copiar de:
`C:\Users\Matheus Bastos Chaga\Documents\dados_enose\nao supervisioando\dbscan\grafico_dbscan_kdistancia.png`

Nome na pasta `imagens`:
`dbscan_kdistancia.png`

```latex
A Figura \ref{fig:dbscan_kdistancia} apresenta a curva de k-distância utilizada na definição do parâmetro de vizinhança do DBSCAN.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\textwidth]{imagens/dbscan_kdistancia.png}
    \caption{Curva de k-distância utilizada como apoio para a definição do parâmetro \(\varepsilon\) do DBSCAN. Fonte: Elaborado pelo autor (2026).}
    \label{fig:dbscan_kdistancia}
\end{figure}
```
>
> Agrupamentos e ruídos do DBSCAN projetados pela PCA:
> `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\nao supervisioando\dbscan\grafico_dbscan_pca.png`

Copiar de:
`C:\Users\Matheus Bastos Chaga\Documents\dados_enose\nao supervisioando\dbscan\grafico_dbscan_pca.png`

Nome na pasta `imagens`:
`dbscan_pca.png`

```latex
A Figura \ref{fig:dbscan_pca} apresenta os agrupamentos e os pontos de ruído identificados pelo DBSCAN.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{imagens/dbscan_pca.png}
    \caption{Comparação entre os agrupamentos obtidos pelo DBSCAN e as classes reais na projeção da PCA. Fonte: Elaborado pelo autor (2026).}
    \label{fig:dbscan_pca}
\end{figure}
```
>
> **Destino recomendado na versão final:** Metodologia para os gráficos de escolha
> dos parâmetros; Resultados para as projeções dos agrupamentos. No Referencial,
> seria mais adequado usar somente um esquema conceitual de centroides e densidade.

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

A interpretação da contribuição individual das variáveis pode ser complementada
pela Figura \ref{fig:importancia_permutacao_melhor_modelo}, que apresenta a
importância por permutação calculada para o modelo selecionado.

> **SUGESTÃO DE IMAGEM — após esta seção:** importância por permutação do melhor
> modelo. Esta figura permite discutir quais sinais e variáveis ambientais mais
> influenciaram a classificação.
>
> Caminho:
> `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\06_07_melhor_modelo\importancia_sensores\grafico_importancia_permutacao_melhor_modelo.png`

Copiar de:
`C:\Users\Matheus Bastos Chaga\Documents\dados_enose\06_07_melhor_modelo\importancia_sensores\grafico_importancia_permutacao_melhor_modelo.png`

Nome na pasta `imagens`:
`importancia_permutacao_melhor_modelo.png`

```latex
A Figura \ref{fig:importancia_permutacao_melhor_modelo} apresenta a contribuição das variáveis estimada pela importância por permutação.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.90\textwidth]{imagens/importancia_permutacao_melhor_modelo.png}
    \caption{Importância das variáveis por permutação no melhor modelo de classificação. Fonte: Elaborado pelo autor (2026).}
    \label{fig:importancia_permutacao_melhor_modelo}
\end{figure}
```
>
> **Destino recomendado na versão final:** Resultados e Discussão. O mesmo arquivo
> foi relacionado à Seção 3.4.4; deve aparecer apenas uma vez na dissertação.

### 3.5.8 Validação e métricas de desempenho

A matriz de confusão organiza as previsões segundo classes reais e previstas. Em uma
classificação binária, ela permite calcular verdadeiros positivos, verdadeiros
negativos, falsos positivos e falsos negativos. A acurácia corresponde à proporção
total de previsões corretas. Quando as classes possuem tamanhos diferentes ou
importâncias semelhantes, a acurácia balanceada, definida como a média do recall das
classes, reduz a predominância da classe majoritária
\cite{brodersen2010balanced}.

A Figura \ref{fig:matriz_confusao_melhor_modelo} apresenta a distribuição das
classificações corretas e incorretas do modelo selecionado como o de melhor
desempenho.

> **SUGESTÃO DE IMAGEM — inserir aqui, após a explicação da matriz de confusão:**
> matriz de confusão do melhor modelo selecionado.
>
> Caminho:
> `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\06_07_melhor_modelo\matriz_confusao\matriz_confusao_melhor_modelo.png`

Copiar de:
`C:\Users\Matheus Bastos Chaga\Documents\dados_enose\06_07_melhor_modelo\matriz_confusao\matriz_confusao_melhor_modelo.png`

Nome na pasta `imagens`:
`matriz_confusao_melhor_modelo.png`

```latex
A Figura \ref{fig:matriz_confusao_melhor_modelo} apresenta a matriz de confusão do modelo selecionado como o de melhor desempenho.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.75\textwidth]{imagens/matriz_confusao_melhor_modelo.png}
    \caption{Matriz de confusão do melhor modelo de classificação. Fonte: Elaborado pelo autor (2026).}
    \label{fig:matriz_confusao_melhor_modelo}
\end{figure}
```
>
> **Destino recomendado na versão final:** Resultados. No Referencial, uma matriz
> genérica com VP, VN, FP e FN seria mais apropriada.

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

A Figura \ref{fig:limiares_extratrees_validacao} apresenta os limiares de decisão
selecionados nos folds externos, permitindo verificar sua variação ao longo da
validação cruzada.

> **SUGESTÃO DE IMAGEM — inserir aqui, junto à explicação da escolha do limiar:**
> valores de limiar selecionados nos folds externos da validação.
>
> Caminho:
> `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\extra trees\validacao_limiar_por_coleta\resultados\comparacoes\limiares_extra_trees_cv.png`

Copiar de:
`C:\Users\Matheus Bastos Chaga\Documents\dados_enose\extra trees\validacao_limiar_por_coleta\resultados\comparacoes\limiares_extra_trees_cv.png`

Nome na pasta `imagens`:
`limiares_extratrees_validacao.png`

```latex
A Figura \ref{fig:limiares_extratrees_validacao} apresenta os limiares de decisão selecionados nos folds externos da validação cruzada.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\textwidth]{imagens/limiares_extratrees_validacao.png}
    \caption{Limiares de decisão selecionados nos folds externos da validação cruzada. Fonte: Elaborado pelo autor (2026).}
    \label{fig:limiares_extratrees_validacao}
\end{figure}
```
>
> **Destino recomendado na versão final:** Metodologia ou Resultados da validação.

Em dados obtidos ao longo do tempo dentro de uma mesma coleta, linhas consecutivas
não constituem observações plenamente independentes. Uma divisão aleatória por linha
pode colocar sinais quase idênticos em treinamento e teste, produzindo estimativas
otimistas. A separação por grupos mantém todos os registros de uma coleta em uma
única partição. Na validação cruzada estratificada por grupos, preserva-se, tanto
quanto possível, a proporção das classes sem romper os grupos. Estratégias em blocos
são recomendadas quando há dependência temporal, espacial ou hierárquica
\cite{roberts2017crossvalidation}.

A Figura \ref{fig:divisao_treino_teste_coletas} apresenta a distribuição dos
registros e das coletas entre os conjuntos de treinamento e teste, evidenciando a
preservação dos blocos durante a divisão dos dados.

> **SUGESTÃO DE IMAGEM — inserir aqui, junto à explicação da separação por coletas:**
> representação dos recortes utilizados nos conjuntos de treinamento e teste.
>
> Caminho:
> `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\extra trees\grafico\grafico_dataset_treino_teste.png`

Copiar de:
`C:\Users\Matheus Bastos Chaga\Documents\dados_enose\extra trees\grafico\grafico_dataset_treino_teste.png`

Nome na pasta `imagens`:
`divisao_treino_teste_coletas.png`

```latex
A Figura \ref{fig:divisao_treino_teste_coletas} apresenta a distribuição dos registros e das coletas entre os conjuntos de treinamento e teste.

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{imagens/divisao_treino_teste_coletas.png}
    \caption{Distribuição dos registros e das coletas entre os conjuntos de treinamento e teste. Fonte: Elaborado pelo autor (2026).}
    \label{fig:divisao_treino_teste_coletas}
\end{figure}
```
>
> **Destino recomendado na versão final:** Metodologia, na subseção de divisão dos
> dados.

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

A Figura \ref{fig:metricas_seco_molhado} apresenta o desempenho do modelo nas
condições de solo seco e solo molhado, permitindo comparar a acurácia, a medida F1 e
o recall das classes em cada condição experimental.

> **SUGESTÃO DE IMAGEM — inserir aqui, após a discussão sobre estabilidade entre
> condições:**
> métricas do melhor modelo separadas entre solo seco e solo molhado. Esta figura é
> essencial para explicar que o valor próximo de 95% foi obtido na condição seca e
> que o comportamento foi diferente na condição molhada.
>
> Caminho:
> `C:\Users\Matheus Bastos Chaga\Documents\dados_enose\06_07_melhor_modelo\analise_seco_molhado_balanceado\grafico_metricas_seco_molhado_balanceado.png`

Copiar de:
`C:\Users\Matheus Bastos Chaga\Documents\dados_enose\06_07_melhor_modelo\analise_seco_molhado_balanceado\grafico_metricas_seco_molhado_balanceado.png`

Nome na pasta `imagens`:
`metricas_seco_molhado.png`

```latex
A Figura \ref{fig:metricas_seco_molhado} apresenta o desempenho do melhor modelo nas condições de solo seco e solo molhado.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{imagens/metricas_seco_molhado.png}
    \caption{Desempenho do melhor modelo nas condições de solo seco e solo molhado. Fonte: Elaborado pelo autor (2026).}
    \label{fig:metricas_seco_molhado}
\end{figure}
```
>
> **Destino recomendado na versão final:** Resultados e Discussão.
