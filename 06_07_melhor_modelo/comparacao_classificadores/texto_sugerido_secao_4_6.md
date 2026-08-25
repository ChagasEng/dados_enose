Como verificação complementar, o Extra Trees foi comparado a três classificadores
clássicos: Random Forest, máquina de vetores de suporte linear (SVM) e perceptron
multicamadas (MLP). Todos receberam os mesmos nove atributos e foram avaliados no
mesmo conjunto de teste, formado por coletas não utilizadas no treinamento. Para
evitar que a escolha dos hiperparâmetros incorporasse informação do teste, as
configurações do Extra Trees e do Random Forest foram selecionadas exclusivamente
no conjunto de treino, por validação cruzada estratificada em quatro partições e
agrupada por coleta. Os atributos foram padronizados nos pipelines da SVM e da MLP.

Conforme a Tabela~\ref{tab:comparacao_classificadores_classicos}, o Random Forest
apresentou o maior desempenho no teste, com acurácia balanceada de 92,20%, seguido
pela SVM linear (91,65%), pelo Extra Trees (90,73%) e pela MLP (87,56%). A diferença
entre Random Forest e Extra Trees foi de 1,47 ponto percentual nessa métrica. Na
validação cruzada interna por coleta, entretanto, o Extra Trees obteve a maior média
entre os dois ensembles avaliados (76,29%, contra 74,23% do Random Forest), o que
mostra que a ordem dos modelos depende das coletas usadas na avaliação.

Para o Extra Trees, os resultados foram obtidos diretamente da matriz de confusão
do melhor modelo, composta por 9.377 classificações corretas e 1.282 incorretas para
o grupo com nematoide, além de 10.710 classificações corretas e 746 incorretas para
o grupo saudável. Esses valores correspondem a 90,83% de acurácia, 90,73% de
acurácia balanceada e 90,80% de F1 macro.

Esses resultados não sustentam a afirmação de que o Extra Trees foi o classificador
de maior desempenho no holdout. Ele foi mantido como modelo focal para a análise de
importância dos atributos e para continuidade do protocolo previamente definido,
enquanto o Random Forest é apresentado como o melhor resultado da comparação
complementar. A divergência entre a validação interna e o teste também reforça a
necessidade de interpretar as métricas por coleta, e não apenas por leitura.

> Observação de edição: se for indispensável afirmar que o Extra Trees foi o melhor
> classificador, será necessário obter esse resultado em uma nova avaliação
> independente e pré-especificada. Alterar concorrentes, hiperparâmetros ou a
> divisão dos dados depois de observar o teste produziria uma comparação enviesada.
