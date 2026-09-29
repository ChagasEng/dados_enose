# Escopo das entradas

Esta variante aceita exclusivamente séries temporais de condutância dos sensores:

1. MQ-2
2. MQ-3
3. MQ-5
4. MQ-7
5. MQ-8
6. MQ-135

Cada amostra tem formato `(6, tempo)`. As variáveis ambientais não são
concatenadas às características, ao fingerprint ou à camada fully connected.
A normalização `G/G0` é calculada separadamente para cada um dos seis sensores.

Os termos "seco", "húmido" e "pressão" continuam presentes nos nomes das
classes porque identificam os tratamentos experimentais, mas não são fornecidos
ao modelo como variáveis auxiliares.
