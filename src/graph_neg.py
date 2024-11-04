import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Windows execution
# file_path = '../data/heatmap_neg.xlsx'
# data = pd.read_excel(file_path)

# macOS path
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, '/Users/martin/Documents/projekt/data/heatmap_neg.xlsx')

# Načtení dat z xlsx souboru
data = pd.read_excel(file_path)

# Nastavení sloupce 'category' jako index
data.set_index('category', inplace=True)

# Generování heatmapy s převrácenou barevnou škálou
plt.figure(figsize=(11, 8))
sns.heatmap(data, annot=True, cmap='coolwarm_r', center=0)
plt.title('neg model')
plt.tight_layout()

# Windows execution
#plt.savefig('heatmaps/heatmap_neg.png')

# macOS execution
output_path = os.path.join(base_dir, '/Users/martin/Documents/projekt/heatmaps/heatmap_neg.png')
plt.savefig(output_path)

plt.show()