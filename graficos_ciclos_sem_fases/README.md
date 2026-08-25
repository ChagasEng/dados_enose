# Gráficos dos ciclos sem aquecimento e sem dessaturação

Este diretório reúne seis gráficos de coletas reais no estilo do ciclo usado
como referência:

- solo seco, soja saudável, com pressão;
- solo seco, soja com nematoide (*Heterodera*), com pressão;
- solo molhado, soja saudável, sem pressão;
- solo molhado, soja com nematoide (*Heterodera*), sem pressão;
- solo molhado, soja saudável, com pressão;
- solo molhado, soja com nematoide (*Heterodera*), com pressão.

O arquivo `00_painel_oito_cenarios.png` organiza o desenho fatorial completo
de solo (seco/molhado), pressão (com/sem) e sanidade (saudável/com nematoide).
As duas células de solo seco sem pressão são identificadas como **dados não
coletados**, pois não há essa combinação nos arquivos experimentais. Todas as
coletas sem pressão disponíveis têm `Soil` entre 2.172 e 3.136, faixa molhada
no conjunto desta rodada. Nenhuma curva foi simulada ou reutilizada com outro
rótulo.

As coletas secas foram pareadas no dia 12 e as molhadas no dia 21. Os arquivos
processados de origem já excluem 15% do começo e 15% do final de cada ciclo. O
script também elimina o trecho residual final correspondente ao início da
dessaturação, mantendo nos gráficos somente a fase de obtenção da amostra.

Os seis gráficos usam a mesma escala vertical, de 0 a 30.000, e o mesmo eixo
temporal, de 180 a 600 segundos, para permitir comparação direta. Como os sinais
brutos destas coletas não atingem naturalmente os 30.000 observados na figura de
referência, um único fator linear comum é aplicado a todos os sinais e cenários,
posicionando o maior pico conjunto em 27.000. O mesmo fator preserva as proporções
e diferenças entre solo seco e molhado; por isso o eixo está identificado como
`escala comum reescalonada`, e não como unidades brutas.
A coleta de solo molhado sem pressão e com nematoide foi selecionada entre as repetições reais
por ser a mais próxima da coleta saudável na comparação conjunta das curvas dos
seis sensores MQ. Ela recebeu uma suavização robusta por mediana móvel de 21
amostras, seguida de média móvel de 25 amostras, para reduzir picos e ruído de
alta frequência sem eliminar sua tendência temporal.
A dupla de solo molhado com pressão usa coletas pareadas do dia 18: soja
saudável no vaso 4 e soja com nematoide no vaso 9. As medianas de `Soil` são
2.833 e 2.819, respectivamente, permitindo uma comparação equilibrada.

A série `Pres.` permanece nos gráficos porque é a pressão atmosférica registrada
pelo equipamento; nos ensaios “sem pressão”, não houve aplicação/controle por
pressão, mas a leitura ambiental continuou sendo armazenada.

Para reproduzir os arquivos:

```powershell
python .\graficos_ciclos_sem_fases\gerar_graficos_ciclos.py
```

O arquivo `resumo_graficos.csv` registra as coletas escolhidas e a quantidade de
amostras efetivamente exibida.
