import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the Excel file
df = pd.read_excel('cl_pos_with_std.xlsx')

# Define the markers for each model
markers = {
    'llama3-8B': 'o',
    'mistral-7B': 'X',
    'claude-3.5': 's',
    'gemini-1.0': 'D',
    'gpt-4': '^'
}

# Remove "vanilla-" prefix from model names
df['model'] = df['model'].str.replace('vanilla-', '')

# Remove "neg" suffix from category names
df['category'] = df['category'].str.replace(' pos', '')

# Use a seaborn color palette for better aesthetics and ensure unique colors
palette = sns.color_palette("tab20", len(df['category'].unique()))
base_colors = dict(zip(df['category'].unique(), palette))

# Normalize std_dev for transparency scaling
df['std_dev_norm'] = df['std_dev'] / df['std_dev'].max()

# Create the scatter plot
plt.figure(figsize=(12, 8))
for model in df['model'].unique():
    for category in df['category'].unique():
        subset = df[(df['model'] == model) & (df['category'] == category)]
        for _, row in subset.iterrows():
            alpha_value = max(0.2, 1 - row['std_dev_norm'])  # Ensure minimum alpha value of 0.2
            plt.scatter(row['mean_ua'], row['mean_ru'],
                        label=f'{model} - {category}',
                        marker=markers[model],
                        color=base_colors[category],
                        alpha=alpha_value,  # Adjust transparency based on std_dev_norm
                        edgecolor='black', linewidth=0.5)

# Add a thin black diagonal line
plt.plot([0, max(df['mean_ua'].max(), df['mean_ru'].max())],
         [0, max(df['mean_ua'].max(), df['mean_ru'].max())],
         color='black', linestyle='-', linewidth=1)

# Create custom legends for categories and models
handles, labels = plt.gca().get_legend_handles_labels()

# Create legend for categories
category_handles = [plt.Line2D([0], [0], marker='o', color='w', label=category,
                               markersize=10, markerfacecolor=base_colors[category], markeredgecolor='black', markeredgewidth=0.5) for category in base_colors.keys()]
category_labels = list(base_colors.keys())

# Create legend for models
model_handles = [plt.Line2D([0], [0], marker=markers[model], color='w', label=model,
                            markersize=10, markerfacecolor='k', markeredgecolor='black', markeredgewidth=0.5) for model in markers.keys()]
model_labels = list(markers.keys())

# Add legends to the plot with titles "Category" and "Models"
plt.legend(handles=[plt.Line2D([0], [0], color='w', label='Category')] + category_handles +
           [plt.Line2D([0], [0], color='w', label='Models')] + model_handles,
           labels=['Category'] + category_labels + ['Models'] + model_labels,
           bbox_to_anchor=(1.05, 1), loc='upper left')

plt.xlabel('sentiment_UA')
plt.ylabel('sentiment_RU')
plt.grid(True)
plt.tight_layout()
plt.savefig('test1.png', dpi=300, bbox_inches='tight')
plt.show()
