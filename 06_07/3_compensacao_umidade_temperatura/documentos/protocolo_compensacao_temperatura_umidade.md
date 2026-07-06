# Protocolo de compensacao ambiental

## Abordagem 1: modelo aprende a compensar

Entrada do modelo:

- `MQ2`, `MQ3`, `MQ7`, `MQ8`, `MQ135`, `MQ138`
- `Soil`, `Temp.`, `Pres.`

Vantagem: simples de testar e geralmente melhora a predicao.

Risco: o modelo pode usar ambiente como atalho experimental em vez de aprender o efeito biologico.

Validacao obrigatoria:

- split por coleta;
- teste separado por dia;
- comparacao com modelo MQ-only.

## Abordagem 2: correcao matematica

Fluxo:

1. Obter curvas de sensibilidade dos datasheets.
2. Definir uma referencia de temperatura/umidade.
3. Calcular fator de correcao por sensor.
4. Aplicar correcao antes de treinar o modelo.
5. Comparar graficos e metricas antes/depois.

Formula conceitual:

```text
sensor_corrigido = sensor_cru / fator_ambiente(temperatura, umidade)
```

Observacao: se `Soil` nao for umidade relativa do ar, ele nao deve ser usado diretamente como umidade do datasheet sem validacao.
