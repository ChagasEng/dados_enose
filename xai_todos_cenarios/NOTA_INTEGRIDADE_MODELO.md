# Nota de integridade sobre o classificador

As três matrizes de confusão transcritas no capítulo foram reproduzidas antes
da aplicação de XAI. Os artefatos que as reproduzem exatamente são modelos
`ExtraTreesClassifier`, não `RandomForestClassifier`:

- cenário a: `[[5864, 643], [0, 10185]]`;
- cenário b: `[[3513, 639], [746, 525]]`;
- cenário c: `[[2968, 1456], [0, 4929]]`.

Além disso, o cenário c que gera a matriz publicada usa nove atributos (seis
MQ, `Soil`, `Temp.` e `Pres.`), embora o texto fornecido afirme que ele usa
somente os seis MQ.

Por isso, os resultados SHAP desta pasta explicam os modelos Extra Trees que
de fato originaram as matrizes. Não é metodologicamente correto chamar essas
explicações de explicações do Random Forest. Há duas opções para a versão
final da dissertação:

1. corrigir o nome do classificador e a descrição dos atributos no capítulo;
2. manter Random Forest como classificador final, mas então retreiná-lo nos
   mesmos splits e regenerar matrizes, métricas e resultados SHAP.

Não se deve apenas trocar o rótulo do modelo, pois Extra Trees e Random Forest
aprendem conjuntos de árvores por procedimentos diferentes e podem atribuir
importâncias diferentes às mesmas variáveis.
