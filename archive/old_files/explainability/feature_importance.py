import numpy as np
import matplotlib.pyplot as plt

# Re-using the same theme constants
BG      = "#ecf0f3"
CARD    = "#f7f9fb"
TXT     = "#333333"
GRID    = "#d1d5db"
PRIMARY = "#3b82f6"
MUTED   = "#64748b"
RL_FEATURES = ["SoH", "Temp", "Cycle", "Current"]

def _light_fig(figsize=(7, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=TXT, labelsize=9)
    ax.xaxis.label.set_color(TXT)
    ax.yaxis.label.set_color(TXT)
    ax.title.set_color(TXT)
    plt.tight_layout()
    return fig, ax

def generate_feature_importance_plot(shap_values, chosen_action=0):
    feature_names = RL_FEATURES
    # This helper is needed or we extract logic. 
    # For simplicity, assuming caller passes processed values or we re-implement logic
    from explainability.shap_analysis import _safe_shap_for_action
    vals = np.abs(_safe_shap_for_action(shap_values, chosen_action))
    order = np.argsort(vals)

    fig, ax = _light_fig((7, 3.5))
    colors = [PRIMARY, "#6366f1", "#8b5cf6", "#a78bfa"]
    ax.barh(np.array(feature_names)[order], vals[order],
            color=[colors[i % 4] for i in order], edgecolor="#c5cbd3", lw=0.5)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=10)
    ax.set_title("SoH Prediction Feature Importance", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.2, color=GRID)
    fig.tight_layout()
    return fig

def generate_feature_ranking_plot(shap_values):
    feature_names = RL_FEATURES
    from explainability.shap_analysis import _all_actions_shap
    matrix = np.abs(_all_actions_shap(shap_values))
    agg = matrix.mean(axis=0)
    order = np.argsort(agg)[::-1]

    fig, ax = _light_fig((7, 3.5))
    ranks = np.arange(len(feature_names))
    colors = [PRIMARY, "#6366f1", "#8b5cf6", "#a78bfa"]
    ax.barh(ranks, agg[order],
            color=[colors[i % 4] for i in range(len(feature_names))],
            edgecolor="#c5cbd3", lw=0.5)
    ax.set_yticks(ranks)
    ax.set_yticklabels(np.array(feature_names)[order], fontsize=10, color=TXT)
    ax.invert_yaxis()
    ax.set_xlabel("Aggregate |SHAP| (all actions)", fontsize=10)
    ax.set_title("Feature Ranking", fontsize=13, fontweight="bold")
    for i, v in enumerate(agg[order]):
        ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9, color=MUTED)
    ax.grid(axis="x", alpha=0.15, color=GRID)
    fig.tight_layout()
    return fig
