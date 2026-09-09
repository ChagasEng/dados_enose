# XAI nos três cenários experimentais

Esta pasta aplica explicabilidade pós-hoc aos mesmos modelos e conjuntos de
teste que geraram as matrizes apresentadas no capítulo de resultados.

## Técnica adotada

Foi utilizado **SHAP com TreeExplainer**, adequado a modelos de árvores e a
dados tabulares. O artigo de Brar, Singh e Nanda (2025) usa Grad-CAM e LIME em
imagens classificadas por uma CNN 2D. Grad-CAM não é aplicável diretamente a
este estudo, pois as entradas são leituras tabulares de sensores MQ e
variáveis ambientais, e os modelos que geraram as matrizes são Extra Trees.
SHAP preserva o objetivo do artigo: tornar visíveis as variáveis que sustentam
as decisões do classificador.

Para cada cenário são gerados:

- importância global por média do valor SHAP absoluto;
- gráfico beeswarm com magnitude e direção dos efeitos;
- uma explicação local representativa de cada célula não vazia da matriz de
  confusão;
- tabelas CSV reprodutíveis;
- comparação percentual entre os três cenários.

A classe explicada é `0 = com nematoide`. Valor SHAP positivo desloca a saída
do modelo em direção a essa classe; valor negativo a desloca em direção à
classe `1 = sem nematoide`.

## Amostragem

A explicação global usa até 2.000 linhas do conjunto de teste por cenário,
amostradas de forma determinística e balanceada por coleta e classe. Isso
reduz a influência indevida de coletas com mais registros. A divisão entre
treino e teste não é alterada e nenhuma linha de treino é explicada como se
fosse teste.

## Execução

```bash
python xai_todos_cenarios/scripts/aplicar_xai_todos_cenarios.py
```

Os arquivos são gravados em `xai_todos_cenarios/resultados/`.

## Resultado obtido

| Cenário | 1ª variável | 2ª variável | 3ª variável | MQ em conjunto | Ambiente em conjunto |
|---|---:|---:|---:|---:|---:|
| Solo seco, com pressão | MQ-138 corrigido (17,67%) | Pressão (15,73%) | Umidade do solo (15,34%) | 63,39% | 36,61% |
| Solo úmido, com pressão | Temperatura (20,67%) | Pressão (15,50%) | Umidade do solo (13,37%) | 50,46% | 49,54% |
| Sem pressão, não estratificado | Pressão (24,65%) | Temperatura (21,56%) | MQ-2 (13,84%) | 52,26% | 47,74% |

O cenário seco foi o mais apoiado no conjunto dos sensores MQ. Nos cenários
úmido e sem pressão, aproximadamente metade da importância foi atribuída às
variáveis ambientais. Isso não estabelece causalidade; mostra quais
associações o modelo utilizou e recomenda controlar o contexto ambiental e
realizar ablações antes de atribuir o desempenho a uma assinatura biológica.

## Nota de integridade dos resultados

Embora o texto fornecido chame o classificador das três matrizes de
Random Forest, os artefatos que reproduzem exatamente essas matrizes são
modelos `ExtraTreesClassifier`:

- cenários a e b: um único modelo Extra Trees, com separação posterior do
  teste por umidade;
- cenário c: modelo Extra Trees do conjunto sem pressão.

Aplicar XAI a um Random Forest diferente não explicaria as previsões das
matrizes publicadas. Por isso, esta análise explica os modelos efetivamente
associados aos números do capítulo.

## Referência inspiradora

BRAR, D. S.; SINGH, B.; NANDA, V. An XAI-enabled 2D-CNN model for
non-destructive detection of natural adulterants in the wonder hot variety of
red chilli powder. *Sustainable Food Technology*, v. 3, p. 1099–1113, 2025.
DOI: 10.1039/D5FB00118H.
