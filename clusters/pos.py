import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Získání adresáře aktuálního skriptu
script_dir = os.path.dirname(os.path.abspath(__file__))

# Sestavení úplné cesty k Excel souboru
excel_path = os.path.join(script_dir, 'cl_pos_with_std.xlsx')
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/clusters')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'cluster_pos_contours.png')

# Načtení dat z Excel souboru
df = pd.read_excel(excel_path)

# Definování barev pro každý model
colors = {
    'llama3-8B': 'blue',
    'mistral-7B': 'orange',
    'claude-3.5': 'green',
    'gemini-1.0': 'red',
    'gpt-4': 'purple'
}

# Definování markerů pro každou kategorii
markers = {
    'make statement': 'o',
    'cooperate': 'X',
    'yield': 's',
    'investigate': 'D',
    'demand': '^',
    'disapprove': '<',
    'reject': '>',
    'threaten': 'v',
    'protest': '*',
    'exhibit force': 'P',
    'reduce relations': 'H',
    'coerce': '8',
    'assault': 'h',
    'fight': 'p',
    'mass violence': 'd'
}

# Odstranění prefixu "vanilla-" z názvů modelů
df['model'] = df['model'].str.replace('vanilla-', '')

# Odstranění sufixu "neg" z názvů kategorií
df['category'] = df['category'].str.replace(' pos', '')

# Normalizace std_dev pro škálování průhlednosti
df['std_dev_norm'] = df['std_dev'] / df['std_dev'].max()

# Parametry Gaussiánu
grid_size = 100
w = 25# Experimentální šířka Gaussiánu
threshold = 1.1  # Spodní hranice pro vykreslení vrstevnic

# Vytvoření scatter plotu
plt.figure(figsize=(12, 8))
for model in df['model'].unique():
    model_data = df[df['model'] == model]

    # Vykreslení bodů
    for _, row in model_data.iterrows():
        alpha_value = max(0.5, 1 - row['std_dev_norm'])  # Zajištění minimální hodnoty průhlednosti 0.5
        plt.scatter(row['mean_ua'], row['mean_ru'],
                    marker=markers[row['category']],
                    color=colors[model],
                    alpha=alpha_value,  # Nastavení průhlednosti na základě std_dev_norm
                    edgecolor='black', linewidth=0.5)

    # Vytvoření gridu
    x_min, x_max = model_data['mean_ua'].min() - 10, model_data['mean_ua'].max() + 10
    y_min, y_max = model_data['mean_ru'].min() - 10, model_data['mean_ru'].max() + 10
    X, Y = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                       np.linspace(y_min, y_max, grid_size))

    # Vytvoření Z jako součtu Gaussiánů
    Z = np.zeros_like(X)
    for _, row in model_data.iterrows():
        x_i, y_i = row['mean_ua'], row['mean_ru']
        Z += np.exp(-((X - x_i) ** 2 + (Y - y_i) ** 2) / w)

    # Vykreslení vrstevnic pouze pro hodnoty Z >= threshold
    max_Z = np.nanmax(Z)
    if not np.all(np.isnan(Z)) and max_Z >= threshold:
        levels = np.linspace(threshold, max_Z, 5)  # Úrovně vrstevnic
        plt.contour(X, Y, Z, levels=levels, colors=colors[model], linewidths=0.8)

# Přidání tenké černé diagonální čáry
plt.plot([0, max(df['mean_ua'].max(), df['mean_ru'].max())],
         [0, max(df['mean_ua'].max(), df['mean_ru'].max())],
         color='black', linestyle='-', linewidth=1)
# Vytvoření vlastních legend pro kategorie a modely
handles, labels = plt.gca().get_legend_handles_labels()

# Vytvoření legendy pro kategorie
category_handles = [plt.Line2D([0], [0], marker=markers[category], color='w', label=category,
                               markersize=10, markerfacecolor='k', markeredgecolor='black', markeredgewidth=0.5) for
                    category in markers.keys()]
category_labels = list(markers.keys())

# Vytvoření legendy pro modely
model_handles = [plt.Line2D([0], [0], marker='o', color='w', label=model,
                            markersize=10, markerfacecolor=colors[model], markeredgecolor='black', markeredgewidth=0.5)
                 for model in colors.keys()]
model_labels = list(colors.keys())

# Přidání legend do grafu s tituly "Category" a "Models"
plt.legend(handles=[plt.Line2D([0], [0], color='w', label='Category')] + category_handles +
                   [plt.Line2D([0], [0], color='w', label='Models')] + model_handles,
           labels=['Category'] + category_labels + ['Models'] + model_labels,
           bbox_to_anchor=(1.05, 1), loc='upper left')


plt.xlabel('sentiment_UA')
plt.ylabel('sentiment_RU')
plt.grid(True)
plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
