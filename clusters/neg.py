import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Získání adresáře aktuálního skriptu
script_dir = os.path.dirname(os.path.abspath(__file__))

# Sestavení úplné cesty k Excel souboru
excel_path = os.path.join(script_dir, 'cl_neg_with_std.xlsx')
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/clusters')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'cluster_neg_contours.png')

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
df['category'] = df['category'].str.replace(' neg', '')

# Normalizace std_dev pro škálování průhlednosti
df['std_dev_norm'] = df['std_dev'] / df['std_dev'].max()

# Vytvoření scatter plotu
plt.figure(figsize=(12, 8))
for model in df['model'].unique():
    for category in df['category'].unique():
        subset = df[(df['model'] == model) & (df['category'] == category)]
        for _, row in subset.iterrows():
            alpha_value = max(0.45, 1 - row['std_dev_norm'])  # Zajištění minimální hodnoty průhlednosti 0.5
            plt.scatter(row['mean_ua'], row['mean_ru'],
                        label=f'{model} - {category}',
                        marker=markers[category],
                        color=colors[model],
                        alpha=alpha_value,  # Úprava průhlednosti na základě std_dev_norm
                        edgecolor='black', linewidth=0.5)

# Přidání vrstevnic hustoty pro každý model pomocí sns.kdeplot s upravenými parametry
for model in df['model'].unique():
    subset = df[df['model'] == model]

    x = subset['mean_ua']
    y = subset['mean_ru']

    if len(x) > 1:
        if model == "gpt-4":
            sns.kdeplot(
                x=x, y=y,
                levels=4,  # Méně vrstevnic pro fialový model (gpt-4)
                color=colors[model],
                linewidths=0.3,
                bw_adjust=0.5,  # Ještě detailnější shluky
                thresh=0.3
            )
        elif model == "claude-3.5":
            sns.kdeplot(
                x=x, y=y,
                levels=4,  # Více vrstevnic pro zelený model (claude-3.5)
                color=colors[model],
                linewidths=0.3,
                bw_adjust=0.5,  # Ještě detailnější shluky
                thresh=0.3
            )
        else:
            sns.kdeplot(
                x=x, y=y,
                levels=4,  # Zůstává počet vrstevnic pro ostatní modely
                color=colors[model],
                linewidths=0.3,
                bw_adjust=0.4,  # Ještě detailnější shluky
                thresh=0.4
            )

# Přidání tenké černé diagonální čáry od 0 do 100 na obou osách
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