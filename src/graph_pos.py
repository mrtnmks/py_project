import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data from the xlsx file
file_path = 'heatmap_pos.xlsx'
data = pd.read_excel(file_path)

# Set the 'category' column as the index
data.set_index('category', inplace=True)

# Generate the heatmap with a blueish color palette
plt.figure(figsize=(12, 8))
sns.heatmap(data, annot=True, cmap='coolwarm_r', center=0)
plt.title('pos model')
plt.tight_layout()
plt.savefig('heatmaps/heatmap_p')
plt.show()