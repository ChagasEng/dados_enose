# Busca completa das nove variaveis no LOSO

Depois da pergunta sobre variar os ambientais separadamente, esta busca avaliou **todas as 511 combinacoes nao vazias** das nove variaveis: seis MQ (`MQ2`, `MQ3`, `MQ7`, `MQ8`, `MQ135`, `MQ138`), indice operacional de umidade do solo (`Soil_indice_0_1`), temperatura (`Temp_C`) e pressao (`Pres_kPa`). As tres ultimas foram combinadas individualmente, incluindo sete configuracoes sem MQ. Destas 511, **126** ja haviam sido calculadas nos cenarios so MQ e MQ + ambiente completo; as outras **385** foram treinadas agora.

Todas as configuracoes usam a mesma base corrigida: **55.946 leituras, 31 coletas e dez vasos** apos retirar as cinco coletas listadas em `../validacao_sem_cinco_coletas/coletas_excluidas.csv`. O Extra Trees foi refeito em dez folds LOSO por vaso, com 900 arvores, `max_features=sqrt`, `min_samples_leaf=10`, `bootstrap=False` e semente 42. Cada leitura foi testada uma vez; o ranking usa acuracia por leitura. Em empate, prefere menos variaveis e depois mais MQ.

| Categoria | Melhor configuracao | Acertos / leituras | Acuracia |
|---|---|---:|---:|
| **Geral, entre 511** | MQ2 + MQ3 + MQ8 + MQ135 + MQ138 + solo + temperatura + pressao | **54.783 / 55.946** | **97,92%** |
| Melhor com 1 ou 2 ambientais | MQ2 + MQ3 + MQ135 + MQ138 + solo + temperatura | 54.625 / 55.946 | 97,64% |
| Melhor sem MQ | Solo + temperatura + pressao | 53.865 / 55.946 | 96,28% |
| Melhor so MQ | MQ2 + MQ3 + MQ7 + MQ8 + MQ135 + MQ138 | 53.388 / 55.946 | 95,43% |

**Conclusao da busca:** variar os ambientais separadamente **nao mudou o primeiro lugar nesta base**, mas mostrou alternativas proximas e um desempenho alto usando apenas as variaveis ambientais. A diferenca entre o primeiro lugar e o melhor com apenas parte dos ambientais foi de **158 leituras**, ou cerca de **0,28 ponto percentual**.

Os **511** casos contam variaveis de entrada, nao equipamentos fisicos. O BMP280 e um unico componente que fornece **temperatura e pressao**. Se elas forem tratadas como um bloco inseparavel para uma decisao de hardware, ha **255 combinacoes nao vazias** de seis MQ + BMP280 + sensor de solo. Esse subconjunto de 255 tambem tem o mesmo primeiro lugar.

**Limites:** a selecao das cinco coletas ocorreu apos inspecao da base e incluiu tres escolhidas pelos erros do classificador. A escolha do vencedor e sua avaliacao usaram os mesmos dez vasos. A acuracia de 97,92% e, portanto, descritiva e exploratoria, nao uma estimativa independente para novos vasos. O indice de solo e uma normalizacao operacional do sinal do sensor, nao uma porcentagem calibrada de agua; o BMP280 nao mede umidade relativa do ar.

Arquivos: `ranking_511.csv` contem todas as combinacoes ordenadas; `acertos_por_coleta_511.csv` permite verificar os 31 resultados por coleta de cada configuracao; `metodo.json` documenta o protocolo. Para reproduzir a partir da raiz do projeto:

```powershell
.\.venv\Scripts\python.exe analise_combinatoria_sensores\buscar_nove_variaveis_loso.py
```
