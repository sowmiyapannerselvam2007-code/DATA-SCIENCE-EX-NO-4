# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

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
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
df = pd.read_csv("bmi.csv")  
print("Original Dataset:")
print(df.head())

df = df.dropna()

df_std = df.copy()
scaler_std = StandardScaler()
df_std[['Height', 'Weight']] = scaler_std.fit_transform(df_std[['Height', 'Weight']])
print("\nStandard Scaled Data:")
print(df_std.head())
```

<img width="498" height="339" alt="Screenshot 2026-05-22 205229" src="https://github.com/user-attachments/assets/94d15033-cf2c-44dd-a511-198e93d2ba86" />

```
df_minmax = df.copy()
scaler_minmax = MinMaxScaler()
df_minmax[['Height', 'Weight']] = scaler_minmax.fit_transform(df_minmax[['Height', 'Weight']])

print("\nMin-Max Scaled Data:")
print(df_minmax.head())
```

<img width="466" height="191" alt="Screenshot 2026-05-22 205236" src="https://github.com/user-attachments/assets/dedac0cc-4df8-4bd4-ba3b-33d3331f1043" />

```
df_maxabs = df.copy()
scaler_maxabs = MaxAbsScaler()
df_maxabs[['Height', 'Weight']] = scaler_maxabs.fit_transform(df_maxabs[['Height', 'Weight']])
print("\nMaxAbs Scaled Data:")
print(df_maxabs.head())
```

<img width="427" height="173" alt="Screenshot 2026-05-22 205244" src="https://github.com/user-attachments/assets/b28f8f57-c733-49b8-86cf-57d33a1e6e63" />

```
df_robust = df.copy()
scaler_robust = RobustScaler()
df_robust[['Height', 'Weight']] = scaler_robust.fit_transform(df_robust[['Height', 'Weight']])

print("\nRobust Scaled Data:")
print(df_robust.head())
```

<img width="481" height="174" alt="Screenshot 2026-05-22 205250" src="https://github.com/user-attachments/assets/f5ec4683-33ed-4a71-8e6c-5a9e9f6ad0e7" />

```
df_std.to_csv("BMI_StandardScaled.csv", index=False)
df_minmax.to_csv("BMI_MinMaxScaled.csv", index=False)
df_maxabs.to_csv("BMI_MaxAbsScaled.csv", index=False)
df_robust.to_csv("BMI_RobustScaled.csv", index=False)

print("\nFeature Scaling Completed Successfully.")
```

<img width="439" height="64" alt="Screenshot 2026-05-22 205257" src="https://github.com/user-attachments/assets/b1fb7c71-af87-4ccb-825e-4942cfd6b9a2" />

```
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE, SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
df = pd.read_csv("income(1) (1).csv")
print("Dataset Preview:")
print(df.head())
```

<img width="775" height="471" alt="Screenshot 2026-05-22 205307" src="https://github.com/user-attachments/assets/84a2979d-1e3e-4463-9fd4-82797820f2b9" />

```
#Encode Cateorigal Variables
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation','relationship', 'race', 'gender', 'nativecountry']

df[categorical_columns] = df[categorical_columns].astype('category').apply(lambda x: x.cat.codes)
if df['SalStat'].dtype == 'object':
    df['SalStat'] = df['SalStat'].astype('category').cat.codes
X = df.drop(columns=['SalStat'])
y = df['SalStat']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
selector_chi2 = SelectKBest(score_func=chi2, k=6)
selector_chi2.fit(X_scaled, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("\nChi-Square Selected:", list(selected_features_chi2))
```

<img width="996" height="65" alt="Screenshot 2026-05-22 205319" src="https://github.com/user-attachments/assets/391ea9c7-5ef1-4ebb-a5dd-248691a6fdb3" />

```
selector_anova = SelectKBest(score_func=f_classif, k=5)
selector_anova.fit(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nANOVA Selected:", list(selected_features_anova))
```

<img width="809" height="42" alt="Screenshot 2026-05-22 205334" src="https://github.com/user-attachments/assets/6d5c2efd-8990-4f00-a961-84d3ca7e8352" />

```
logreg = LogisticRegression(max_iter=1000)
rfe = RFE(estimator=logreg, n_features_to_select=6)
rfe.fit(X, y)
selected_features_rfe = X.columns[rfe.support_]
print("\nRFE Selected:", list(selected_features_rfe))
```

<img width="888" height="53" alt="Screenshot 2026-05-22 205342" src="https://github.com/user-attachments/assets/027598e5-5c67-42d0-9160-a2755d8fe513" />

```
rf = RandomForestClassifier(n_estimators=100, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

rf.fit(X_train, y_train)

selector_embedded = SelectFromModel(rf, threshold="mean")
selector_embedded.fit(X_train, y_train)

selected_features_embedded = X.columns[selector_embedded.get_support()]
print("\nEmbedded Method Selected:", list(selected_features_embedded))
```

<img width="1031" height="64" alt="Screenshot 2026-05-22 205352" src="https://github.com/user-attachments/assets/b1e22f93-b3f3-47a0-81bd-fa0ca4844674" />

```
X_train_sel = selector_embedded.transform(X_train)
X_test_sel = selector_embedded.transform(X_test)

rf.fit(X_train_sel, y_train)
y_pred = rf.predict(X_test_sel)

print("\nModel Accuracy (Embedded Method):", accuracy_score(y_test, y_pred))
```

<img width="525" height="67" alt="Screenshot 2026-05-22 205402" src="https://github.com/user-attachments/assets/a05a3102-4938-4b25-b46c-c42b35f96c4e" />




















# RESULT:
       # INCLUDE YOUR RESULT HERE
