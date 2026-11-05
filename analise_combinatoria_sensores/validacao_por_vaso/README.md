# Validacao 70/30 e LOSO por vaso

**Nota posterior:** esta pagina agrega cada coleta em uma unica previsao. Para a validacao solicitada pelo orientador, que conta acertos e erros dentro de cada coleta, use `../validacao_por_leitura/README.md`.

Unidade biologica: vaso. Unidade da metrica: coleta. O modelo aprende com leituras, mas cada coleta recebe **uma** predicao pela media das probabilidades de suas leituras. Todas as metricas abaixo sao por coleta.

Foram mantidas cinco configuracoes fixas da analise combinatoria anterior, com o mesmo Extra Trees de 900 arvores. Nenhum conjunto de sensores foi escolhido novamente com base nos resultados desta validacao.

| Configuracao | Original por coleta (12 coletas) | 70/30 por vaso (10 coletas) | LOSO por vaso (36 coletas) |
|---|---:|---:|---:|
| MQ2, MQ135, MQ138 + ambiente | 91,67% (11/12) | 91,67% (9/10) | **89,16% (32/36)** |
| MQ2, MQ3, MQ8, MQ135, MQ138 | 91,67% (11/12) | 91,67% (9/10) | 80,65% (29/36) |
| Seis MQ | 91,67% (11/12) | 91,67% (9/10) | 77,71% (28/36) |
| Seis MQ + ambiente | 83,33% (10/12) | 100,00% (10/10) | 86,22% (31/36) |
| MQ7 + ambiente | 75,00% (9/12) | 91,67% (9/10) | 86,53% (31/36) |

Percentuais: **acuracia balanceada por coleta**. Os acertos entre parenteses sao contagens simples; em 70/30 por vaso e no LOSO as classes nao tem necessariamente o mesmo numero de coletas, por isso percentual balanceado e fracao de acertos podem diferir.

## Protocolos

- **Split original por coleta:** 24 coletas de treino e 12 de teste (aproximadamente 67/33, chamado de 70/30 no projeto). Oito dos dez vasos aparecem em ambos os lados. Serve como referencia historica, mas nao mede generalizacao a vaso novo.
- **70/30 por vaso:** sete vasos no treino e tres no teste, estratificados por classe, semente 42. As dez coletas dos tres vasos de teste nao aparecem no treino. Uma unica particao de tres vasos gera uma estimativa muito sensivel aos vasos sorteados.
- **LOSO por vaso:** dez folds. Cada fold treina com nove vasos e testa todas as coletas do decimo. As 36 previsoes por coleta sao reunidas para calcular as metricas finais. Nao calculamos acuracia balanceada por fold isolado, porque cada fold contem apenas uma classe.

Os identificadores de vaso sao `Classe + Vaso`, porque o numero `1` aparece nas duas classes e representa unidades biologicas diferentes. **Conferir antes da publicacao:** o texto de materiais e metodos diz que os vasos inoculados sao 6 a 10, mas este CSV identifica os inoculados como 1, 6, 7, 8 e 9. Isso nao altera a separacao por unidade usada aqui, mas a nomenclatura deve ser reconciliada.

O candidato MQ2, MQ135, MQ138 + ambiente errou quatro coletas no LOSO: vaso inoculado 9 no dia 19 e vasos saudaveis 1 (dia 19) e 2 (dias 12 e 19). Sua matriz de confusao agregada foi: classe 0, 16/17 corretas; classe 1, 16/19 corretas.

Os sensores avaliados foram escolhidos previamente com estes dados, inclusive com consulta ao split original de teste. Portanto, a validacao LOSO e uma analise de robustez interna, nao uma confirmacao externa da escolha.

Arquivos: `metricas_por_coleta.csv`, `predicoes_por_coleta.csv`, `acertos_loso_por_vaso.csv`, `metodo.json` e `secao_validacao.tex`. Execute `..\validacao_por_vaso.py` a partir desta pasta ou o caminho completo a partir da raiz do projeto.
