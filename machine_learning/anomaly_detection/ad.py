from sklearn.preprocessing import MinMaxScaler
import pandas as pd

df = pd.read_csv('../../datasets/per_field/usl/clustering/Mall_Customers.csv')

# Data Preprocessing:
df = df.drop('CustomerID', axis=1)
df = df.rename(columns={'Annual Income (k$)': 'Income', 'Spending Score (1-100)': 'Spend_Score'})

# Encode categorical variables into dummy \ indicator variables:
df = pd.get_dummies(df)

# Scale numerical variables:
scaler = MinMaxScaler()
numerical_vars = ['Age', 'Income', 'Spend_Score']
df[numerical_vars] = scaler.fit_transform(df[numerical_vars])
