import numpy as np 
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt


Index= ['aaa', 'bbb', 'ccc', 'ddd', 'eee']
Cols = ['A', 'B', 'C', 'D']
df = DataFrame(abs(np.random.randn(5, 4)), index=Index, columns=Cols)

sns.heatmap(df, annot=True)
plt.show()

