# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Setting up the parameters for the random data
np.random.seed(42)
models = ['llama3-8B', 'mistral-7B', 'daude-3.5', 'gemini-1.0', 'gpt-4']
categories = ['statement pos', 'statement neg', 'cooperation pos', 'cooperation neg', 'retreat neg', 
              'retreat pos', 'investigation neg', 'investigation pos', 'demand neg', 'demand pos',
              'disapproval neg', 'disapproval pos', 'rejection neg', 'rejection pos', 'threat neg',
              'threat pos', 'protest neg', 'protest pos', 'force neg', 'force pos', 'relation neg',
              'relation pos', 'coercion neg', 'coercion pos', 'assault neg', 'assault pos', 'hybrid attack neg',
              'hybrid attack pos']

# Creating a DataFrame for the data
data = {
    'mean': np.random.uniform(-10, 40, len(models) * len(categories)),
    'std': np.random.uniform(0, 35, len(models) * len(categories)),
    'category': np.tile(categories, len(models)),
    'model': np.repeat(models, len(categories))
}

df = pd.DataFrame(data)

# Setting up the plot
plt.figure(figsize=(14, 10))
sns.set(style="whitegrid")
palette = sns.color_palette("husl", len(categories))

# Plotting the scatter plot
sns.scatterplot(
    data=df,
    x='mean',
    y='std',
    hue='category',
    style='model',
    palette=palette,
    s=100
)

# Customizing the plot
plt.title("Mean vs Standard Deviation for Different Categories and Models")
plt.xlabel("mean")
plt.ylabel("std")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Legend")
plt.show()
