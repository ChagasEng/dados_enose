from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


OUT = Path(__file__).resolve().parent
BLUE = "#2166AC"
LIGHT_BLUE = "#D7EAF7"
GREEN = "#2A9D6F"
LIGHT_GREEN = "#DDF2E8"
ORANGE = "#E6862D"
LIGHT_ORANGE = "#FCE7D2"
PURPLE = "#7251A3"
LIGHT_PURPLE = "#E8DDF3"
RED = "#C94C4C"
LIGHT_RED = "#F6DDDD"
DARK = "#243447"
GRAY = "#667788"
LIGHT_GRAY = "#EEF2F5"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 16,
        "figure.facecolor": "white",
    }
)


def canvas(figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def box(ax, xy, width, height, text, facecolor=LIGHT_BLUE, edgecolor=BLUE, fontsize=11):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=1.7,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        color=DARK,
        fontsize=fontsize,
        fontweight="semibold",
    )
    return patch


def arrow(ax, start, end, color=GRAY, width=1.8, connectionstyle="arc3,rad=0"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=width,
        color=color,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(patch)
    return patch


def save(fig, filename):
    fig.savefig(OUT / filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def compensacao_ambiental():
    fig, ax = canvas()
    ax.set_title("Compensação conceitual de interferências ambientais", pad=18, color=DARK)
    box(ax, (0.04, 0.61), 0.18, 0.21, "Temperatura\nUmidade do solo\nPressão", LIGHT_ORANGE, ORANGE)
    box(ax, (0.31, 0.61), 0.18, 0.21, "Resposta bruta\ndos sensores", LIGHT_BLUE, BLUE)
    box(ax, (0.58, 0.61), 0.18, 0.21, "Modelo de\ncompensação", LIGHT_PURPLE, PURPLE)
    box(ax, (0.81, 0.61), 0.15, 0.21, "Sinal\ncorrigido", LIGHT_GREEN, GREEN)
    box(
        ax,
        (0.58, 0.85),
        0.18,
        0.075,
        "Datasheet de cada sensor MQ\n(orientação da correção)",
        LIGHT_GRAY,
        GRAY,
        8.2,
    )
    arrow(ax, (0.22, 0.715), (0.31, 0.715), ORANGE)
    arrow(ax, (0.49, 0.715), (0.58, 0.715), BLUE)
    arrow(ax, (0.76, 0.715), (0.81, 0.715), PURPLE)
    arrow(ax, (0.67, 0.85), (0.67, 0.82), GRAY, 1.4)

    x = np.linspace(0, 1, 160)
    raw = 0.48 + 0.18 * np.sin(2 * np.pi * x) + 0.13 * x
    corrected = 0.48 + 0.18 * np.sin(2 * np.pi * x)
    ax.plot(0.13 + 0.28 * x, 0.18 + 0.22 * raw, color=ORANGE, lw=2.2)
    ax.plot(0.59 + 0.28 * x, 0.18 + 0.22 * corrected, color=GREEN, lw=2.2)
    ax.text(0.27, 0.15, "Sinal com influência ambiental", ha="center", color=GRAY)
    ax.text(0.73, 0.15, "Assinatura sensorial compensada", ha="center", color=GRAY)
    arrow(ax, (0.43, 0.27), (0.57, 0.27), PURPLE)
    ax.text(
        0.5,
        0.05,
        "Correção estatística individual por sensor, orientada pelos datasheets",
        ha="center",
        color=DARK,
        fontsize=11,
    )
    ax.text(
        0.5,
        0.012,
        "Sem umidade relativa do ar, o procedimento não equivale à calibração física completa por Rs/R0",
        ha="center",
        color=GRAY,
        fontsize=9.2,
    )
    save(fig, "compensacao_ambiental_conceitual.png")


def preprocessamento_sinais():
    fig, ax = canvas()
    ax.set_title("Etapas conceituais do pré-processamento de sinais", pad=18, color=DARK)
    stages = [
        (0.03, "Sinal bruto", "Ruído e diferenças\nde linha de base", LIGHT_RED, RED),
        (0.23, "Correção de\nbaseline", "Referência comum\nentre medições", LIGHT_ORANGE, ORANGE),
        (0.44, "Filtragem e\nsuavização", "Atenuação de\noscilações rápidas", LIGHT_BLUE, BLUE),
        (0.66, "Normalização", "Escalas\ncomparáveis", LIGHT_PURPLE, PURPLE),
        (0.84, "Sinal\npré-processado", "Pronto para\na modelagem", LIGHT_GREEN, GREEN),
    ]
    widths = [0.15, 0.16, 0.17, 0.14, 0.13]
    for i, ((x, title, subtitle, face, edge), width) in enumerate(zip(stages, widths)):
        box(ax, (x, 0.56), width, 0.22, f"{title}\n{subtitle}", face, edge, 9.5)
        if i < len(stages) - 1:
            next_x = stages[i + 1][0]
            arrow(ax, (x + width, 0.67), (next_x, 0.67), edge)

    x = np.linspace(0, 1, 220)
    rng = np.random.default_rng(4)
    raw = 0.48 + 0.11 * np.sin(3 * np.pi * x) + 0.14 * x + rng.normal(0, 0.018, x.size)
    processed = 0.48 + 0.14 * np.sin(3 * np.pi * x)
    ax.plot(0.06 + 0.32 * x, 0.17 + 0.23 * raw, color=RED, lw=1.8)
    ax.plot(0.62 + 0.32 * x, 0.17 + 0.23 * processed, color=GREEN, lw=2.2)
    ax.text(0.22, 0.13, "Sinal com ruído, deriva e escala original", ha="center", color=GRAY)
    ax.text(0.78, 0.13, "Sinal em escala comparável", ha="center", color=GRAY)
    arrow(ax, (0.41, 0.25), (0.59, 0.25), PURPLE)
    ax.text(
        0.5,
        0.05,
        "As transformações devem preservar a dinâmica relevante da resposta sensorial",
        ha="center",
        color=DARK,
    )
    save(fig, "preprocessamento_sinais_conceitual.png")


def construcao_atributos():
    fig, ax = canvas()
    ax.set_title("Construção e seleção de atributos em sinais de nariz eletrônico", pad=18, color=DARK)
    x = np.linspace(0, 1, 200)
    for offset, color, phase in [(0.0, BLUE, 0), (0.08, GREEN, 0.7), (-0.07, ORANGE, 1.3)]:
        y = 0.67 + offset + 0.1 * np.sin(2 * np.pi * x + phase) + 0.16 * np.exp(-((x - 0.55) / 0.18) ** 2)
        ax.plot(0.03 + 0.23 * x, y, color=color, lw=2)
    ax.text(0.145, 0.49, "Sinais temporais", ha="center", fontweight="semibold", color=DARK)
    arrow(ax, (0.27, 0.66), (0.34, 0.66), BLUE)

    features = ["Máximo", "Área sob\na curva", "Inclinação", "Tempo de\nresposta", "Razões entre\nsensores"]
    y_positions = [0.78, 0.64, 0.50, 0.36, 0.22]
    for label, y in zip(features, y_positions):
        box(ax, (0.35, y - 0.055), 0.17, 0.11, label, LIGHT_BLUE, BLUE, 9.5)
    ax.text(0.435, 0.09, "Extração de características", ha="center", fontweight="semibold", color=DARK)
    arrow(ax, (0.53, 0.5), (0.60, 0.5), BLUE)

    box(ax, (0.61, 0.64), 0.17, 0.13, "Qui-quadrado", LIGHT_PURPLE, PURPLE)
    box(ax, (0.61, 0.43), 0.17, 0.13, "Informação\nmútua", LIGHT_PURPLE, PURPLE)
    box(ax, (0.61, 0.22), 0.17, 0.13, "Importância por\npermutação", LIGHT_PURPLE, PURPLE, 9.5)
    ax.text(0.695, 0.09, "Seleção", ha="center", fontweight="semibold", color=DARK)
    arrow(ax, (0.79, 0.5), (0.84, 0.5), PURPLE)
    box(ax, (0.85, 0.38), 0.12, 0.24, "Vetor de\natributos\nselecionados", LIGHT_GREEN, GREEN)
    save(fig, "construcao_selecao_atributos_conceitual.png")


def pca_lda():
    rng = np.random.default_rng(12)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle("Diferença conceitual entre PCA e LDA", fontsize=17, color=DARK, y=0.98)
    a = rng.multivariate_normal([-1.2, -0.2], [[1.0, 0.72], [0.72, 0.75]], 80)
    b = rng.multivariate_normal([1.2, 0.25], [[1.0, 0.72], [0.72, 0.75]], 80)
    for ax, title in zip(axes, ["PCA: preservação da variância", "LDA: separação entre classes"]):
        ax.scatter(a[:, 0], a[:, 1], s=24, alpha=0.72, color=BLUE, label="Classe A")
        ax.scatter(b[:, 0], b[:, 1], s=24, alpha=0.72, color=ORANGE, label="Classe B")
        ax.axhline(0, color="#D9E0E6", lw=1)
        ax.axvline(0, color="#D9E0E6", lw=1)
        ax.set_title(title, color=DARK, pad=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#CDD6DE")
    axes[0].arrow(-2.6, -2.0, 4.7, 3.8, width=0.025, head_width=0.22, color=PURPLE, length_includes_head=True)
    axes[0].text(0.25, 1.85, "Direção de maior variância", color=PURPLE, fontweight="semibold")
    axes[1].arrow(-2.4, -0.25, 4.8, 0.5, width=0.025, head_width=0.22, color=GREEN, length_includes_head=True)
    axes[1].text(-0.55, 0.72, "Direção discriminante", color=GREEN, fontweight="semibold")
    axes[1].legend(frameon=False, loc="lower right")
    fig.text(0.5, 0.02, "A PCA não utiliza rótulos; a LDA utiliza as classes conhecidas.", ha="center", color=GRAY)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    save(fig, "pca_lda_conceitual.png")


def supervisionado_nao_supervisionado():
    fig, ax = canvas()
    ax.set_title("Aprendizado supervisionado e não supervisionado", pad=18, color=DARK)
    ax.plot([0.5, 0.5], [0.10, 0.86], color="#D5DDE4", lw=2)
    ax.text(0.25, 0.86, "Supervisionado", ha="center", fontsize=15, fontweight="bold", color=BLUE)
    ax.text(0.75, 0.86, "Não supervisionado", ha="center", fontsize=15, fontweight="bold", color=PURPLE)
    box(ax, (0.06, 0.60), 0.18, 0.13, "Dados com\nrótulos", LIGHT_BLUE, BLUE)
    box(ax, (0.29, 0.60), 0.15, 0.13, "Treinamento", LIGHT_BLUE, BLUE)
    arrow(ax, (0.24, 0.665), (0.29, 0.665), BLUE)
    box(ax, (0.17, 0.30), 0.20, 0.15, "Predição de\nclasses conhecidas", LIGHT_GREEN, GREEN)
    arrow(ax, (0.365, 0.60), (0.27, 0.45), BLUE)
    box(ax, (0.56, 0.60), 0.18, 0.13, "Dados sem\nrótulos", LIGHT_PURPLE, PURPLE)
    box(ax, (0.79, 0.60), 0.15, 0.13, "Agrupamento", LIGHT_PURPLE, PURPLE)
    arrow(ax, (0.74, 0.665), (0.79, 0.665), PURPLE)
    box(ax, (0.66, 0.30), 0.20, 0.15, "Estruturas,\ngrupos e ruídos", LIGHT_ORANGE, ORANGE)
    arrow(ax, (0.865, 0.60), (0.76, 0.45), PURPLE)
    ax.text(0.25, 0.18, "Exemplos: classificação e regressão", ha="center", color=GRAY)
    ax.text(0.75, 0.18, "Exemplos: K-means e DBSCAN", ha="center", color=GRAY)
    save(fig, "aprendizado_supervisionado_nao_supervisionado.png")


def conjuntos_arvores():
    fig, ax = canvas(figsize=(13, 6.5))
    ax.set_title("Estratégias de conjuntos de árvores", pad=18, color=DARK)
    box(ax, (0.03, 0.41), 0.13, 0.18, "Dados de\nentrada", LIGHT_GRAY, GRAY)
    branches = [
        (0.24, 0.70, "Random Forest", "Reamostragem +\nsubconjuntos de atributos", BLUE, LIGHT_BLUE),
        (0.24, 0.41, "Extra Trees", "Cortes mais\naleatórios", GREEN, LIGHT_GREEN),
        (0.24, 0.12, "Boosting", "Árvores sequenciais\ncorrigem erros", ORANGE, LIGHT_ORANGE),
    ]
    for x, y, title, subtitle, color, light in branches:
        arrow(ax, (0.16, 0.50), (x, y + 0.08), color, connectionstyle="arc3,rad=0.08")
        box(ax, (x, y), 0.20, 0.17, f"{title}\n{subtitle}", light, color, 10)
        for i in range(3):
            cx = 0.53 + i * 0.085
            cy = y + 0.085
            ax.plot([cx, cx], [cy - 0.035, cy + 0.035], color=color, lw=2)
            ax.plot([cx, cx - 0.022], [cy + 0.01, cy + 0.045], color=color, lw=1.7)
            ax.plot([cx, cx + 0.022], [cy + 0.01, cy + 0.045], color=color, lw=1.7)
            if title == "Boosting" and i < 2:
                arrow(ax, (cx + 0.025, cy), (cx + 0.06, cy), ORANGE, 1.3)
        arrow(ax, (0.74, y + 0.085), (0.82, 0.50), color, connectionstyle="arc3,rad=-0.08")
    box(ax, (0.83, 0.41), 0.14, 0.18, "Predição\ncombinada", LIGHT_PURPLE, PURPLE)
    ax.text(0.60, 0.04, "A diversidade entre árvores reduz erros; o modo de gerar essa diversidade distingue os métodos.", ha="center", color=GRAY)
    save(fig, "conjuntos_arvores_conceitual.png")


def classificadores():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Princípios de classificadores probabilísticos, lineares e baseados em distância", fontsize=16, color=DARK, y=0.98)
    rng = np.random.default_rng(7)
    titles = ["Naive Bayes", "Regressão logística", "SVM", "KNN"]
    for ax, title in zip(axes.ravel(), titles):
        ax.set_title(title, color=DARK, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#CDD6DE")
    x = np.linspace(-4, 4, 300)
    axes[0, 0].plot(x, np.exp(-0.5 * (x + 1.2) ** 2), color=BLUE, lw=2.5)
    axes[0, 0].plot(x, 0.9 * np.exp(-0.5 * (x - 1.3) ** 2), color=ORANGE, lw=2.5)
    axes[0, 0].text(
        0.5, 0.04, "Probabilidade dos atributos em cada classe",
        ha="center", transform=axes[0, 0].transAxes, color=GRAY, fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
    )
    axes[0, 1].plot(x, 1 / (1 + np.exp(-x)), color=GREEN, lw=3)
    axes[0, 1].axhline(0.5, ls="--", color=GRAY)
    axes[0, 1].text(
        0.5, 0.04, "Combinação linear convertida em probabilidade",
        ha="center", transform=axes[0, 1].transAxes, color=GRAY, fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
    )
    p1 = rng.normal([-1.3, -0.6], 0.45, (25, 2))
    p2 = rng.normal([1.3, 0.6], 0.45, (25, 2))
    axes[1, 0].scatter(p1[:, 0], p1[:, 1], color=BLUE, s=22)
    axes[1, 0].scatter(p2[:, 0], p2[:, 1], color=ORANGE, s=22)
    axes[1, 0].plot([-2.5, 2.5], [1.7, -1.7], color=PURPLE, lw=2.6)
    axes[1, 0].plot([-2.5, 2.5], [2.2, -1.2], color=PURPLE, lw=1.3, ls="--")
    axes[1, 0].plot([-2.5, 2.5], [1.2, -2.2], color=PURPLE, lw=1.3, ls="--")
    axes[1, 0].text(
        0.5, 0.04, "Fronteira com margem máxima",
        ha="center", transform=axes[1, 0].transAxes, color=GRAY, fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
    )
    pts = rng.uniform(-2, 2, (38, 2))
    cls = pts[:, 0] + 0.5 * pts[:, 1] > 0
    axes[1, 1].scatter(pts[~cls, 0], pts[~cls, 1], color=BLUE, s=24)
    axes[1, 1].scatter(pts[cls, 0], pts[cls, 1], color=ORANGE, s=24)
    axes[1, 1].scatter([0.1], [-0.1], marker="*", s=180, color=GREEN, edgecolor=DARK)
    axes[1, 1].add_patch(Circle((0.1, -0.1), 0.85, fill=False, color=GREEN, lw=2))
    axes[1, 1].text(
        0.5, 0.04, "Votação entre os vizinhos mais próximos",
        ha="center", transform=axes[1, 1].transAxes, color=GRAY, fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    save(fig, "classificadores_conceitual.png")


def arquitetura_mlp():
    fig, ax = canvas(figsize=(12, 6.5))
    ax.set_title("Arquitetura conceitual de um perceptron multicamadas", pad=18, color=DARK)
    layers = [
        (0.12, 5, "Camada de entrada", BLUE),
        (0.38, 6, "Camada oculta 1", PURPLE),
        (0.64, 4, "Camada oculta 2", ORANGE),
        (0.88, 2, "Camada de saída", GREEN),
    ]
    positions = []
    for x, count, label, color in layers:
        ys = np.linspace(0.25, 0.78, count)
        layer_pos = []
        for y in ys:
            ax.add_patch(Circle((x, y), 0.025, facecolor="white", edgecolor=color, lw=2.2))
            layer_pos.append((x, y))
        positions.append(layer_pos)
        ax.text(x, 0.14, label, ha="center", color=DARK, fontweight="semibold")
    for left, right in zip(positions[:-1], positions[1:]):
        for p1 in left:
            for p2 in right:
                ax.plot([p1[0] + 0.025, p2[0] - 0.025], [p1[1], p2[1]], color="#C9D2DA", lw=0.65, zorder=0)
    ax.text(0.12, 0.88, "Sensores e\natributos", ha="center", color=BLUE)
    ax.text(0.51, 0.88, "Combinações ponderadas +\nfunções de ativação", ha="center", color=PURPLE)
    ax.text(0.88, 0.88, "Probabilidade\ndas classes", ha="center", color=GREEN)
    arrow(ax, (0.83, 0.92), (0.20, 0.92), RED, connectionstyle="arc3,rad=0.25")
    ax.text(0.51, 0.98, "Retropropagação do erro", ha="center", color=RED, fontweight="semibold")
    save(fig, "arquitetura_mlp_conceitual.png")


def kmeans_dbscan():
    rng = np.random.default_rng(21)
    c1 = rng.normal([-1.4, -0.2], [0.45, 0.6], (55, 2))
    c2 = rng.normal([1.3, 0.4], [0.5, 0.42], (55, 2))
    noise = np.array([[-2.6, 1.7], [2.7, -1.4], [0.0, 2.0], [-0.1, -2.1]])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle("Diferença conceitual entre K-means e DBSCAN", fontsize=17, color=DARK, y=0.98)
    axes[0].scatter(c1[:, 0], c1[:, 1], color=BLUE, alpha=0.75, s=25)
    axes[0].scatter(c2[:, 0], c2[:, 1], color=ORANGE, alpha=0.75, s=25)
    axes[0].scatter([-1.4, 1.3], [-0.2, 0.4], marker="X", s=180, color=[PURPLE, PURPLE], label="Centroides")
    axes[0].set_title("K-means: distância aos centroides", color=DARK)
    axes[0].legend(frameon=False, loc="lower right")
    axes[1].scatter(c1[:, 0], c1[:, 1], color=BLUE, alpha=0.75, s=25)
    axes[1].scatter(c2[:, 0], c2[:, 1], color=ORANGE, alpha=0.75, s=25)
    axes[1].scatter(noise[:, 0], noise[:, 1], marker="x", s=80, color=RED, label="Ruído")
    axes[1].set_title("DBSCAN: regiões de alta densidade", color=DARK)
    axes[1].legend(frameon=False, loc="lower right")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-3.1, 3.1)
        ax.set_ylim(-2.5, 2.5)
        for spine in ax.spines.values():
            spine.set_color("#CDD6DE")
    fig.text(0.5, 0.03, "O K-means exige k; o DBSCAN identifica grupos densos e pode sinalizar ruídos.", ha="center", color=GRAY)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    save(fig, "kmeans_dbscan_conceitual.png")


def validacao_metricas():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Validação por grupos e avaliação do desempenho", fontsize=17, color=DARK, y=0.98)
    ax = axes[0]
    ax.set_title("Separação por coletas", color=DARK)
    colors = [BLUE, GREEN, ORANGE, PURPLE, RED]
    for i in range(5):
        y = 4 - i
        for j in range(5):
            color = LIGHT_RED if i == j else LIGHT_BLUE
            edge = RED if i == j else BLUE
            rect = Rectangle((j, y), 0.88, 0.72, facecolor=color, edgecolor=edge, lw=1.3)
            ax.add_patch(rect)
            ax.text(j + 0.44, y + 0.36, f"C{j + 1}", ha="center", va="center", color=DARK)
        ax.text(5.05, y + 0.36, f"Fold {i + 1}: C{i + 1} em validação", va="center", color=GRAY, fontsize=9)
    ax.text(2.3, -0.45, "Uma coleta permanece inteira em uma única partição", ha="center", color=GRAY)
    ax.set_xlim(-0.2, 7.1)
    ax.set_ylim(-0.7, 5.1)
    ax.axis("off")

    ax = axes[1]
    ax.set_title("Matriz de confusão e métricas", color=DARK)
    matrix_colors = [[LIGHT_GREEN, LIGHT_RED], [LIGHT_RED, LIGHT_GREEN]]
    labels = [["VP", "FN"], ["FP", "VN"]]
    for row in range(2):
        for col in range(2):
            rect = Rectangle((col, 1 - row), 1, 1, facecolor=matrix_colors[row][col], edgecolor="white", lw=3)
            ax.add_patch(rect)
            ax.text(col + 0.5, 1.5 - row, labels[row][col], ha="center", va="center", fontsize=18, fontweight="bold", color=DARK)
    ax.text(1, 2.25, "Classe prevista", ha="center", color=GRAY)
    ax.text(-0.25, 1, "Classe real", ha="center", va="center", rotation=90, color=GRAY)
    metrics = ["Acurácia", "Acurácia balanceada", "Precisão", "Recall", "F1"]
    for i, metric in enumerate(metrics):
        box(ax, (2.35, 1.72 - i * 0.39), 1.25, 0.27, metric, LIGHT_PURPLE, PURPLE, 9.5)
    arrow(ax, (2.05, 1.0), (2.33, 1.0), PURPLE)
    ax.set_xlim(-0.5, 3.8)
    ax.set_ylim(-0.25, 2.55)
    ax.axis("off")
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    save(fig, "validacao_metricas_conceitual.png")


if __name__ == "__main__":
    preprocessamento_sinais()
    compensacao_ambiental()
    construcao_atributos()
    pca_lda()
    supervisionado_nao_supervisionado()
    conjuntos_arvores()
    classificadores()
    arquitetura_mlp()
    kmeans_dbscan()
    validacao_metricas()
    print(f"Figuras geradas em: {OUT}")
