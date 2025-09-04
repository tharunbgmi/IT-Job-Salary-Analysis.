import pandas as pd
df = pd.read_csv("salary_prediction_data.csv")
print(df.head())
df.drop_duplicates()
df.isnull().sum()
df.dropna()
df.describe()
df.groupby('Job_Title')['Salary'].mean()

import pandas as pd
import scipy.stats as stats

# Load your dataset CSV
df = pd.read_csv('salary_prediction_data.csv')

# Group salaries by job title for ANOVA
groups = [group['Salary'].values for name, group in df.groupby('Job_Title')]

# Run ANOVA test
f_stat, p_value = stats.f_oneway(*groups)

print(f"F-statistic: {f_stat:.2f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Result: Significant differences exist between job title salaries.")
else:
    print("Result: No significant difference in salaries by job title.")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('salary_prediction_data.csv')

# Create boxplot
plt.figure(figsize=(10,6))
sns.boxplot(x='Job_Title', y='Salary', data=df)
plt.title('Salary Distribution by Job Title')
plt.xlabel('Job Title')
plt.ylabel('Salary')
plt.show()
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Load dataset
df = pd.read_csv('salary_prediction_data.csv')

# Perform Tukey's HSD test
tukey = pairwise_tukeyhsd(endog=df['Salary'], groups=df['Job_Title'], alpha=0.05)

# Print results
print(tukey.summary())

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('salary_prediction_data.csv')

# Experience vs Salary
print("Average salary by Experience (years):")
print(df.groupby('Experience')['Salary'].mean())

plt.figure(figsize=(8,5))
sns.scatterplot(x='Experience', y='Salary', data=df)
plt.title('Salary vs Experience')
plt.show()

# Location vs Salary (Categorical)
print("\nAverage salary by Location:")
print(df.groupby('Location')['Salary'].mean())

plt.figure(figsize=(8,5))
sns.boxplot(x='Location', y='Salary', data=df)
plt.title('Salary Distribution by Location')
plt.show()

# Education vs Salary (Categorical)
print("\nAverage salary by Education:")
print(df.groupby('Education')['Salary'].mean())

plt.figure(figsize=(8,5))
sns.boxplot(x='Education', y='Salary', data=df)
plt.title('Salary Distribution by Education Level')
plt.show()

# Age vs Salary
print("\nCorrelation between Age and Salary:")
print(df[['Age', 'Salary']].corr())

plt.figure(figsize=(8,5))
sns.scatterplot(x='Age', y='Salary', data=df)
plt.title('Salary vs Age')
plt.show()

# Gender vs Salary (Categorical)
print("\nAverage salary by Gender:")
print(df.groupby('Gender')['Salary'].mean())

plt.figure(figsize=(8,5))
sns.boxplot(x='Gender', y='Salary', data=df)
plt.title('Salary Distribution by Gender')
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('salary_prediction_data.csv')

# Experience vs Salary
print("Average Salary by Experience:")
print(df.groupby('Experience')['Salary'].mean())
plt.figure(figsize=(8,5))
sns.scatterplot(x='Experience', y='Salary', data=df)
plt.title('Salary vs Experience')
plt.show()

# Location vs Salary
print("\nAverage Salary by Location:")
print(df.groupby('Location')['Salary'].mean())
plt.figure(figsize=(8,5))
sns.boxplot(x='Location', y='Salary', data=df)
plt.title('Salary Distribution by Location')
plt.show()

# Education vs Salary
print("\nAverage Salary by Education:")
print(df.groupby('Education')['Salary'].mean())
plt.figure(figsize=(8,5))
sns.boxplot(x='Education', y='Salary', data=df)
plt.title('Salary Distribution by Education')
plt.show()

# Age vs Salary
print("\nCorrelation between Age and Salary:")
print(df[['Age', 'Salary']].corr())
plt.figure(figsize=(8,5))
sns.scatterplot(x='Age', y='Salary', data=df)
plt.title('Salary vs Age')
plt.show()

# Gender vs Salary
print("\nAverage Salary by Gender:")
print(df.groupby('Gender')['Salary'].mean())
plt.figure(figsize=(8,5))
sns.boxplot(x='Gender', y='Salary', data=df)
plt.title('Salary Distribution by Gender')
plt.show()


import pandas as pd
import statsmodels.api as sm
import numpy as np

# Check for missing values
print(df.isnull().sum())

# Drop or fill missing values (example: drop)
df = df.dropna()

# One-hot encode categorical variables again after dropping NA
df_encoded = pd.get_dummies(df, columns=['Job_Title', 'Location', 'Education', 'Gender'], drop_first=True)

# Define features and target
X = df_encoded.drop(['Salary'], axis=1)
y = df_encoded['Salary']

# Confirm all features are numeric
print(X.dtypes.unique())  # Should show only numeric types

# Convert all to float if needed
X = X.astype(float)

# Add constant (intercept)
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X).fit()

print(model.summary())

import matplotlib.pyplot as plt
import seaborn as sns

# Predicted values and residuals
predictions = model.predict(X)
residuals = y - predictions

# Residual plot (residuals vs predicted)
plt.figure(figsize=(8,5))
sns.scatterplot(x=predictions, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Salary')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Salary')
plt.show()

# Histogram of residuals (normality check)
plt.figure(figsize=(8,5))
sns.histplot(residuals, kde=True)
plt.title('Distribution of Residuals')
plt.show()

# Create new data in same format as X (with dummy variables)
import numpy as np

new_data = pd.DataFrame({
    'const': 1,
    'Experience': [10],
    'Age': [35],
    'Job_Title_Director': [0],
    'Job_Title_Engineer': [1],
    'Job_Title_Manager': [0],
    'Location_Suburban': [0],
    'Location_Urban': [1],
    'Education_High School': [0],
    'Education_Master': [1],
    'Education_PhD': [0],
    'Gender_Male': [1]
})

predicted_salary = model.predict(new_data)
print(f"Predicted Salary: {predicted_salary.values[0]:.2f}")




