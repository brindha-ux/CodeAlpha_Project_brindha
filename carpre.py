import numpy as np # linear algebra
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
df = pd.read_csv('C:/Users/Admin/Downloads/car/car data.csv')
print(df.head())
df.tail()
df.columns
df.info()

df.shape
df.duplicated().any()
duplicate_values = df.duplicated().sum()
duplicate_values
null_values = df.isna().sum()
null_values
df.drop_duplicates(inplace= True)
df.hist(figsize  = (12,12))
numerical_columns = ['Year', 'Selling_Price', 'Present_Price', 'Driven_kms', 'Owner']

numerical_df = df[numerical_columns]

corr_matrix = numerical_df.corr()
corr_matrix
plt.figure(figsize=(6,6))
sns.heatmap(corr_matrix, annot=True, cmap='magma',  linewidths=1,fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.show()
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
sns.pairplot(df[['Present_Price', 'Year', 'Driven_kms', 'Selling_Price']])
plt.show()

