import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the Excel file
excel_path = os.path.join(script_dir, 'cl_pos_with_std.xlsx')

# Load the data from the Excel file
df = pd.read_excel(excel_path)


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

# Create the scatter plot
plt.figure(figsize=(12, 8))
for model in df['model'].unique():
    for category in df['category'].unique():
        subset = df[(df['model'] == model) & (df['category'] == category)]
        for _, row in subset.iterrows():
            alpha_value = max(0.5, 1 - row['std_dev_norm'])  # Ensure minimum alpha value of 0.5
            plt.scatter(row['mean_ua'], row['mean_ru'],
                        label=f'{model} - {category}',
                        marker=markers[category],
                        color=colors[model],
                        alpha=alpha_value,  # Adjust transparency based on std_dev_norm
                        edgecolor='black', linewidth=0.5)

# Add a thin black diagonal line
plt.plot([0, max(df['mean_ua'].max(), df['mean_ru'].max())],
         [0, max(df['mean_ua'].max(), df['mean_ru'].max())],
         color='black', linestyle='-', linewidth=1)

# Create custom legends for categories and models
handles, labels = plt.gca().get_legend_handles_labels()

# Create legend for categories
category_handles = [plt.Line2D([0], [0], marker=markers[category], color='w', label=category,
                               markersize=10, markerfacecolor='k', markeredgecolor='black', markeredgewidth=0.5) for category in markers.keys()]
category_labels = list(markers.keys())

# Create legend for models
model_handles = [plt.Line2D([0], [0], marker='o', color='w', label=model,
                            markersize=10, markerfacecolor=colors[model], markeredgecolor='black', markeredgewidth=0.5) for model in colors.keys()]
model_labels = list(colors.keys())

# Add legends to the plot with titles "Category" and "Models"
plt.legend(handles=[plt.Line2D([0], [0], color='w', label='Category')] + category_handles +
           [plt.Line2D([0], [0], color='w', label='Models')] + model_handles,
           labels=['Category'] + category_labels + ['Models'] + model_labels,
           bbox_to_anchor=(1.05, 1), loc='upper left')

plt.xlabel('sentiment_UA')
plt.ylabel('sentiment_RU')
plt.grid(True)
plt.tight_layout()
#plt.savefig('test_pos.png', dpi=300, bbox_inches='tight')
plt.show()
