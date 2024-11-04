import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Načtení dat z xlsx souboru
file_path = 'heatmap_neg.xlsx'
data = pd.read_excel(file_path)

# Nastavení sloupce 'category' jako index
data.set_index('category', inplace=True)

# Generování heatmapy s převrácenou barevnou škálou
plt.figure(figsize=(12, 8))
sns.heatmap(data, annot=True, cmap='coolwarm_r', center=0)
plt.title('neg model')
plt.tight_layout()
plt.savefig('heatmaps/heatmap_n.png')
plt.show()