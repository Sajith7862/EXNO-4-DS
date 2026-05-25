# EXNO:4-DS
## Name : Mohamed Hameem Sajith J
## Roll : 212223240090
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
- STEP 1:Read the given Data.
- STEP 2:Clean the Data Set using Data Cleaning Process.
- STEP 3:Apply Feature Scaling for the feature in the data set.
- STEP 4:Apply Feature Selection for the feature in the data set.
- STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```python
import pandas as pd
import numpy as np

df = pd.read_csv("/content/bmi.csv")
print(df.head())

```
<img width="385" height="122" alt="image" src="https://github.com/user-attachments/assets/0e701381-43cd-44e9-a0ff-fd0ebaf1d0b6" />

```python
print(df.info())
```
<img width="382" height="220" alt="image" src="https://github.com/user-attachments/assets/26fffdac-2dfe-479f-a6a8-8f2b053ab3e7" />

<img width="476" height="563" alt="image" src="https://github.com/user-attachments/assets/6b82be32-6762-4978-b78d-84fe6fac2305" />
<img width="478" height="329" alt="image" src="https://github.com/user-attachments/assets/c97748cf-5a15-4186-924e-a80385496ef2" />

## FEATURE SCALING :


```python
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler

# Select Numeric Columns
num_cols = df.select_dtypes(include=np.number).columns

```
<img width="527" height="367" alt="image" src="https://github.com/user-attachments/assets/fffe8c1d-86ae-4fb5-98f3-8f02d39dc721" />

<img width="533" height="361" alt="image" src="https://github.com/user-attachments/assets/5db21cad-eea0-4977-a557-c6930f68a274" />

<img width="531" height="359" alt="image" src="https://github.com/user-attachments/assets/04600c35-ff22-4a60-8bfd-7605834756e8" />

<img width="533" height="366" alt="image" src="https://github.com/user-attachments/assets/5ea80507-c063-4f4f-a8eb-f32cb79b4824" />

## FEATURE SELECTION :
```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import OneHotEncoder

# Example:
# Last column considered as target variable
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

# Apply one-hot encoding to the 'Gender' column
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
gender_encoded = encoder.fit_transform(X[['Gender']])
gender_df = pd.DataFrame(gender_encoded, columns=encoder.get_feature_names_out(['Gender']))

# Drop the original 'Gender' column and concatenate the one-hot encoded columns
X_encoded = X.drop('Gender', axis=1)
X_encoded = pd.concat([X_encoded.reset_index(drop=True), gender_df], axis=1)


# Select Top 3 Features
best_features = SelectKBest(score_func=chi2, k=3)

fit = best_features.fit(X_encoded, Y)

feature_scores = pd.DataFrame(fit.scores_)
feature_columns = pd.DataFrame(X_encoded.columns)

# Combine Columns and Scores
featureScores = pd.concat(
    [feature_columns, feature_scores],
    axis=1
)

featureScores.columns = ['Features', 'Score']

print(featureScores)

# Top Features
print(featureScores.nlargest(3, 'Score'))
```
<img width="384" height="169" alt="image" src="https://github.com/user-attachments/assets/71998e22-0289-4e1d-b512-287c26117d7c" />

## SAVE CLEANED DATA :

<img width="541" height="160" alt="image" src="https://github.com/user-attachments/assets/93b64a08-b45e-49cb-8ebc-8670d897e3cf" />


# RESULT:
       Thus feature scaling and selection is performed.
