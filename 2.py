import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import libraries for classification task
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

# Remove warnings
import warnings
warnings.filterwarnings('ignore')

# Load Dataset
df = pd.read_csv('C:/Users/Admin/Intern/Iris.csv')
print(df.head())

# Drop id column
df.drop('Id', axis=1, inplace=True)
print(df.head())

# Select only numerical columns
numerical_columns = df.select_dtypes(include=['number'])

# Compute correlation matrix
correlation_matrix = numerical_columns.corr()
print(correlation_matrix)

plt.figure(figsize=(10, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='rainbow')
plt.title("Correlation of Iris dataset")
plt.show()  # No need to print plt.show()
