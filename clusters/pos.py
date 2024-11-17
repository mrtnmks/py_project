import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data from the Excel file
df = pd.read_excel('cl_pos_with_std.xlsx')

# Define the colors for each model
colors = {
    'llama3-8B': 'blue',
    'mistral-7B': 'orange',
    'claude-3.5': 'green',
    'gemini-1.0': 'red',
    'gpt-4': 'purple'
}

# Define the markers for each category
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

# Remove "vanilla-" prefix from model names
df['model'] = df['model'].str.replace('vanilla-', '')

# Remove "pos" suffix from category names
df['category'] = df['category'].str.replace(' pos', '')

# Normalize std_dev for transparency scaling
df['std_dev_norm'] = df['std_dev'] / df['std_dev'].max()

# Initialize the plot
plt.figure(figsize=(14, 10))

# Add scatter plots and density contours for each model
for model in df['model'].unique():
    subset = df[df['model'] == model]

    # Scatter plot for model points
    for category in subset['category'].unique():
        category_data = subset[subset['category'] == category]
        for _, row in category_data.iterrows():
            alpha_value = max(0.5, 1 - row['std_dev_norm'])  # Ensure minimum alpha value of 0.5
            plt.scatter(row['mean_ua'], row['mean_ru'],
                        label=f'{model} - {category}',
                        marker=markers[category],
                        color=colors[model],
                        alpha=alpha_value,
                        edgecolor='black', linewidth=0.5)

    # Add 2D density contours for the model
    # Use numpy to calculate kernel density estimation on the subset of data
    x = subset['mean_ua']
    y = subset['mean_ru']

    # Calculate 2D density estimate
    kde = sns.kdeplot(x=x, y=y, levels=7, color=colors[model], linewidths=0.3, alpha=0.7)
    kde.collections[0].set_edgecolor('black')  # Optional: change contour edge color

# Add a thin black diagonal line
plt.plot([0, max(df['mean_ua'].max(), df['mean_ru'].max())],
         [0, max(df['mean_ua'].max(), df['mean_ru'].max())],
         color='black', linestyle='-', linewidth=1)

# Create custom legends for categories and models
handles, labels = plt.gca().get_legend_handles_labels()

# Create legend for categories
category_handles = [plt.Line2D([0], [0], marker=markers[category], color='w', label=category,
                               markersize=10, markerfacecolor='k', markeredgecolor='black', markeredgewidth=0.5) for
                    category in markers.keys()]
category_labels = list(markers.keys())

# Create legend for models
model_handles = [plt.Line2D([0], [0], marker='o', color='w', label=model,
                            markersize=10, markerfacecolor=colors[model], markeredgecolor='black', markeredgewidth=0.5)
                 for model in colors.keys()]
model_labels = list(colors.keys())

# Add legends to the plot with titles "Category" and "Models"
plt.legend(handles=[plt.Line2D([0], [0], color='w', label='Category')] + category_handles +
                   [plt.Line2D([0], [0], color='w', label='Models')] + model_handles,
           labels=['Category'] + category_labels + ['Models'] + model_labels,
           bbox_to_anchor=(1.05, 1), loc='upper left')

# Add labels, grid, and save the plot
plt.xlabel('sentiment_UA')
plt.ylabel('sentiment_RU')
#plt.title('Scatter Plot with 2D Density Contours')
plt.grid(True)
plt.tight_layout()
plt.savefig('cluster_pos.png', dpi=300, bbox_inches='tight')
plt.show()
