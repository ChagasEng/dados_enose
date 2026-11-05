# Divisao 70/30 por leitura crua

O split foi feito **nas 63.492 linhas de leitura crua** com `train_test_split(test_size=0.30, stratify=Classe, random_state=42)`: 44.444 leituras de treino e 19.048 de teste. A mesma divisao foi usada nas tres configuracoes abaixo. Cada leitura de teste recebeu uma previsao; a metrica e acuracia normal por leitura.

| Sensores | Acertos no teste | Acuracia normal |
|---|---:|---:|
| MQ2, MQ7, MQ8, MQ135, MQ138 | 18.578/19.048 | 97,53% |
| MQ2, MQ3, MQ8, MQ135, MQ138 | 18.586/19.048 | 97,57% |
| MQ2, MQ3, MQ8, MQ135, MQ138 + solo, temperatura e pressao | **18.643/19.048** | **97,87%** |

Esse split distribuiu leituras das **36 coletas e dos 10 vasos** entre treino e teste. Portanto, testa outras leituras de coletas e vasos que o modelo ja viu. Para verificar generalizacao a vaso nao visto, a comparacao e o LOSO por vaso. Com o mesmo conjunto de cinco MQ mais ambiente, o LOSO obteve **55.916/63.492 (88,07%)**. Os percentuais 70/30 e LOSO respondem a perguntas diferentes.

Arquivos: `split_por_leitura.csv` registra a divisao linha a linha; `metricas_por_leitura.csv` guarda os resultados; `predicoes_teste_por_leitura.csv` permite verificar cada previsao; `acertos_dentro_de_cada_coleta.csv` mostra os acertos e erros em cada coleta; `metodo.json` documenta os parametros.

Para reproduzir a partir da raiz do projeto:

```powershell
.\.venv\Scripts\python.exe analise_combinatoria_sensores\validar_70_30_leituras_selecionadas.py
```
