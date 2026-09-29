"""Constantes compartilhadas pelo pipeline X-TCN para E-nose."""

SENSOR_NAMES = ("MQ-2", "MQ-3", "MQ-5", "MQ-7", "MQ-8", "MQ-135")

FEATURE_NAMES = (
    "AUC",
    "ES",
    "PSD",
    "PW",
    "Var",
    "PP",
    "DFC",
    "FTF",
)

CLASS_NAMES = (
    "Soja normal (controlo)",
    "Solo seco COM nematodes (pressao)",
    "Solo seco SEM nematodes (pressao)",
    "Solo humido COM nematodes (pressao)",
    "Solo humido SEM nematodes (pressao)",
    "Protocolo de referencia",
)

