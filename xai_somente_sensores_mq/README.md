# X-TCN explicável — somente sensores MQ

Implementação PyTorch completa cuja entrada contém **somente os seis sensores
MQ** e os seis tratamentos de solo/soja. Temperatura, humidade ambiente,
temperatura do solo e pressão não entram como variáveis preditoras. Humidade e
pressão aparecem apenas na definição dos tratamentos/classes experimentais.

O exemplo incluído gera dados sintéticos, calcula `G/G0`, extrai os
oito descritores, constrói mapas RGB discretos de 224 × 224, treina a X-TCN com
atenção de canal e espacial e gera uma explicação Grad-CAM.

## Execução rápida

No PowerShell, a partir desta pasta:

```powershell
python -m pip install -r requirements.txt
python run_pipeline.py --epochs 5
```

Para uma validação rápida em CPU:

```powershell
python run_pipeline.py --epochs 1 --samples-per-class 6 --time-steps 96
python -m pytest -q
```

## Execução com a base real

```powershell
python run_real_data.py --epochs 40
```

Esta execução usa exclusivamente `MQ2`, `MQ3`, `MQ7`, `MQ8`, `MQ135` e
`MQ138`, faz a separação por coleta e grava o relatório integral em
`resultados_reais/RELATORIO_ACURACIAS_E_RESULTADOS.txt`. A base disponível
possui rótulos binários (com nemátode versus saudável), não seis tratamentos
multiclasse explicitamente rotulados.

## Teste alinhado ao artigo X-TCN

```powershell
python run_article_aligned.py --epochs 30 --random-epochs 30
```

Gera em `resultados_artigo_alinhado/` o teste principal com coletas disjuntas
e um diagnóstico aleatório 80/20. O segundo reproduz a lógica de divisão do
texto do artigo, mas não deve ser usado como estimativa científica porque
janelas sobrepostas da mesma coleta podem aparecer nos dois subconjuntos.

Todos os artefactos são escritos em `xai/results/`: checkpoint, métricas,
histórico, matriz de confusão, explicação Grad-CAM e tabela CSV de relevância
para cada combinação sensor × característica.

## Organização

- `xtcn/preprocessing.py`: resposta relativa `S = G/G0`.
- `xtcn/features.py`: AUC, ES, PSD, PW, Var, PP, DFC e FTF.
- `xtcn/fingerprint.py`: padronização sem fuga e mapa RGB em blocos discretos.
- `xtcn/model.py`: backbone dilatado, atenção dupla CA/SA e classificador.
- `xtcn/gradcam.py`: Grad-CAM na última convolução.
- `xtcn/synthetic.py`: seis classes sintéticas balanceadas.
- `xtcn/training.py`: treino Adam/Cross-Entropy e avaliação.
- `run_pipeline.py`: pipeline integral e exportação dos resultados.

## Convenções dos oito descritores

Cada descritor é um escalar por sensor e por janela. A análise usa a magnitude
dinâmica `abs(G/G0 - 1)`. `ES` é a energia total do espectro sem DC; `PSD` é o
pico do periodograma; `PW` é a largura normalizada à meia altura; `PP` é a
posição normalizada do máximo; `DFC` é a frequência dominante normalizada pela
frequência de Nyquist; e `FTF` é a entropia espectral normalizada.

O `FeatureStandardizer` deve sempre ser ajustado somente nas amostras de treino.
O mapa 224 × 224 usa 48 blocos constantes delimitados por índices inteiros, sem
redimensionamento interpolado. O Grad-CAM interpolado é usado apenas para
visualização e não volta a entrar no classificador.

## Utilização com dados reais

Substitua `generate_synthetic_enose` por um carregador que produza um array
`float32` de forma `(amostras, 6, tempo)` na ordem dos sensores definida em
`xtcn/constants.py`, e os rótulos inteiros de 0 a 5. Mantenha a separação por
vaso/dia/lote antes de ajustar o normalizador para impedir que medições
correlacionadas do mesmo ensaio apareçam em treino e teste.

Os dados sintéticos validam o software, mas não validam desempenho científico.
Em dados reais, recomenda-se validação agrupada por vaso ou campanha, repetição
por sementes aleatórias e análise das relevâncias Grad-CAM agregadas por classe.
