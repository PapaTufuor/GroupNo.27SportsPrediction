# -*- coding: utf-8 -*-
"""Mid Sem.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1N7D8BprQ53P-pcdLy2jcn1vD8bFW23PE

**Importing libraries**
"""

import pandas as pd
import joblib
import pickle
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from google.colab import drive
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

drive.mount('/content/drive')



"""## **players_21 dataset**"""

players_21 = pd.read_csv('/content/drive/My Drive/Mid Sem Project/players_21.csv')

players_21 = pd.DataFrame(players_21)

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)

players_21

players_21.head()

players_21.set_index('short_name', inplace=True)

players_21

players_21.info()

players_21.describe()

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)

useless_columns = ['sofifa_id', 'player_url', 'dob', 'long_name', 'club_team_id', 'club_loaned_from',
                   'nationality_id', 'nation_team_id', 'player_tags', 'player_traits', 'real_face',
                   'player_face_url', 'club_logo_url', 'club_flag_url',
                   'nation_logo_url', 'nation_flag_url', 'club_jersey_number', 'nation_jersey_number', 'club_joined',
            'club_contract_valid_until']

players_21 = players_21.drop(columns=useless_columns)

players_21

threshold = len(players_21) * 0.7
players_21 = players_21.dropna(thresh=threshold, axis=1)

players_21

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

players_21.isna().any()

cat_cols = ['club_name', 'league_name', 'club_position']

players_21[cat_cols] = players_21[cat_cols].fillna("Unknown")

num_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
knn_imputer = KNNImputer(n_neighbors=3)

num_cols = ['value_eur', 'wage_eur', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']

players_21[num_cols] = knn_imputer.fit_transform(players_21[num_cols])

players_21['release_clause_eur'].fillna(0, inplace=True)

players_21['league_level'].fillna(0, inplace=True)

players_21.isna().any()



pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)

categorical = ['player_positions', 'club_name', 'league_name', 'club_position', 'nationality_name', 'preferred_foot', 'work_rate', 'body_type']

scaler = StandardScaler()

#Columns probably containing + or -


columns1 = ['ls','st','rs','lw','lf','cf','rf','rw','lam','cam','ram',
            'lm','lcm','cm','rcm','rm','lwb','ldm', 'cdm','rdm','rwb',
            'lb','lcb','cb','rcb','rb', 'gk']

#Dropping values after + or -
for col in columns1:
    players_21[col] = players_21[col].str.split('+', n = 1, expand = True)[0]
    players_21[col] = players_21[col].str.split('-', n = 1, expand = True)[0]
players_21[columns1] = players_21[columns1].apply(pd.to_numeric, errors='coerce')

#Making players' postion only their main position instead of several

players_21['player_positions'] = players_21['player_positions'].apply(lambda x: x.split(',')[0].strip())

unique_positions = players_21['player_positions'].unique()
unique_positions

#https://www.kaggle.com/code/alefernandezarmas/fifa-22-player-overall-predictions?scriptVersionId=129277545&cellId=62

players_21

"""**Exploratory Data Analysis**"""

players_21.describe()

"""Correlation Matrix"""

corr_matrix = players_21.corr()
sns.heatmap(corr_matrix)
plt.title('Correlation Matrix')
plt.show()

"""Overall rating distribution"""

plt.hist(players_21['overall'], bins=30)
plt.title('Distribution of Overall Rating')
plt.show()

"""Ages of players"""

players_21.sort_values(by='overall',ascending=False)[["overall","age"]].head(20)

"""Nationalities of Players"""

players_21 = pd.get_dummies(players_21, columns=categorical)

pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', None)

players_21

y = players_21['overall']

players_21_scaled = pd.DataFrame(scaler.fit_transform(players_21), columns=players_21.columns)

X = players_21_scaled.drop('overall', axis=1)

pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', None)

players_21_scaled

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)

"""## **players_22 dataset**

**Data Loading and Exploration**
"""

players_22 = pd.read_csv('/content/drive/My Drive/Mid Sem Project/players_22.csv')

players_22 = pd.DataFrame(players_22)

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', None)

players_22

players_22.head()

players_22.set_index('short_name', inplace=True)

players_22

players_22.info()

players_22.describe()

"""**Data Cleaning**"""

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', None)

players_22 = players_22.drop(columns=useless_columns)

players_22

players_22 = players_22.dropna(thresh=threshold, axis=1)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

players_22.isna().any()



players_22[cat_cols] = players_22[cat_cols].fillna("Unknown")

players_22[num_cols] = knn_imputer.fit_transform(players_22[num_cols])

players_22['release_clause_eur'].fillna(0, inplace=True)

players_22['league_level'].fillna(0, inplace=True)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

players_22.isna().any()

#Dropping values after + or -
for col in columns1:
    players_22[col] = players_22[col].str.split('+', n = 1, expand = True)[0]
    players_22[col] = players_22[col].str.split('-', n = 1, expand = True)[0]

#Making players' postion only their main position instead of several

players_22['player_positions'] = players_22['player_positions'].apply(lambda x: x.split(',')[0].strip())

unique_positions = players_22['player_positions'].unique()
unique_positions

#https://www.kaggle.com/code/alefernandezarmas/fifa-22-player-overall-predictions?scriptVersionId=129277545&cellId=62

pd.set_option('display.max_rows', 5)
pd.set_option('display.max_columns', None)

players_22

"""**Exploratory Data Analysis**"""

players_22.describe()

"""Correlation Matrix"""

corr_matrix_22 = players_22.corr()
sns.heatmap(corr_matrix_22)
plt.title('Correlation Matrix')
plt.show()

"""Overall rating distribution"""

plt.hist(players_22['overall'], bins=30)
plt.title('Distribution of Overall Rating')
plt.show()

"""Ages of players"""

players_22.sort_values(by='overall',ascending=False)[["overall","age"]].head(20)

players_22 = pd.get_dummies(players_22, columns=categorical)

y_22 = players_22['overall']

y_22

players_22_scaled = pd.DataFrame(scaler.fit_transform(players_22), columns=players_22.columns)

players_22_scaled

X_22 = players_22_scaled.drop('overall', axis=1)

"""Question 2"""

feature_corr = corr_matrix['overall']

top_features=feature_corr.sort_values(ascending=False).head(15)

top_features

feature_model = RandomForestRegressor()

feature_model.fit(X_train,y_train)

feature_importance = feature_model.feature_importances_

feature_importance

feature_importance_df_train = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance})

feature_importance_df_train = feature_importance_df_train.sort_values(by='Importance', ascending=False)

top_features_list = feature_importance_df_train.head(7)["Feature"].tolist()

top_features_list

#top_features_list = top_features.index.tolist()
#top_features_list.remove('overall')

X_train_subset = X_train[top_features_list]
X_test_subset = X_test[top_features_list]

models = [
    ("Linear Regression", LinearRegression()),
    ("Decision Tree", DecisionTreeRegressor()),
    ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42)),
    ("xGB Boost", xgb.XGBRegressor(n_estimators=100, random_state=42)),
    ("Gradient Boost", GradientBoostingRegressor(n_estimators=100, random_state=42))
]

for name, model in models:
    # Train model
    model.fit(X_train_subset, y_train)

    # Cross-validation
    scores = cross_val_score(model, X_train_subset, y_train, cv=5)

    print(f"{name}: Cross-validation scores: {scores}")

for name, model in models:
    # Predict on test set
    y_pred = model.predict(X_test_subset)

    # Calculate MAE and RMSE
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"{name}: MAE = {mae}, RMSE = {rmse}")

random_forest = RandomForestRegressor()
random_forest_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

xgb_model = xgb.XGBRegressor()
xgb_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

gradient_boost = GradientBoostingRegressor()
gradient_boost_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

models_and_params = [
    (random_forest, random_forest_params, "Random Forest"),
    (xgb_model, xgb_params, "XGB Boost"),
    (gradient_boost, gradient_boost_params, "Gradient Boost")
]

for model, param_grid, name in models_and_params:
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
    grid_search.fit(X_train_subset, y_train)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print(f"\n{name} - Best Parameters: {best_params}")

    # Evaluate on the test set
    y_pred = best_model.predict(X_test_subset)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"{name} - MAE = {mae}, RMSE = {rmse}")

best_random_forest = RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_split=2)
best_xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=7)
best_gradient_boost = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=7)

voting_regressor = VotingRegressor(estimators=[
    ('random_forest', best_random_forest),
    ('xgb', best_xgb_model),
    ('gradient_boost', best_gradient_boost)
])

voting_regressor.fit(X_train_subset, y_train)

y_pred = voting_regressor.predict(X_test_subset)

mae_voting = mean_absolute_error(y_test, y_pred)
rmse_voting = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Voting Ensemble Model: MAE = {mae_voting}, RMSE = {rmse_voting}")

X_22 = players_22[X_train_subset.columns]

X_22

X_22 = pd.DataFrame(scaler.fit_transform(X_22), columns=X_22.columns)

X_test_22 = X_22[X_train_subset.columns]

y_pred_22 = voting_regressor.predict(X_test_22)

mae_22 = mean_absolute_error(y_22, y_pred_22)
rmse_22 = np.sqrt(mean_squared_error(y_22, y_pred_22))

print(f"MAE on players_22: {mae_22}")
print(f"RMSE on players_22: {rmse_22}")

data = pd.DataFrame({'Real': y_22, 'Predicted': y_pred_22})

sns.scatterplot(data=data, x='Real', y='Predicted')
plt.plot([y_22.min(), y_22.max()], [y_22.min(), y_22.max()], 'r--', lw=3)
plt.xlabel('Real')
plt.ylabel('Predicted')
plt.show()

best_random_forest.fit(X_train_subset, y_train)

y_pred_22 = best_random_forest.predict(X_test_22)

mae_22 = mean_absolute_error(y_22, y_pred_22)
rmse_22 = np.sqrt(mean_squared_error(y_22, y_pred_22))

print(f"MAE on players_22: {mae_22}")
print(f"RMSE on players_22: {rmse_22}")

data = pd.DataFrame({'Real': y_22, 'Predicted': y_pred_22})

sns.scatterplot(data=data, x='Real', y='Predicted')
plt.plot([y_22.min(), y_22.max()], [y_22.min(), y_22.max()], 'r--', lw=3)
plt.xlabel('Real')
plt.ylabel('Predicted')
plt.show()

best_xgb_model.fit(X_train_subset, y_train)

y_pred_22 = best_xgb_model.predict(X_test_22)

mae_22 = mean_absolute_error(y_22, y_pred_22)
rmse_22 = np.sqrt(mean_squared_error(y_22, y_pred_22))

print(f"MAE on players_22: {mae_22}")
print(f"RMSE on players_22: {rmse_22}")

data = pd.DataFrame({'Real': y_22, 'Predicted': y_pred_22})

sns.scatterplot(data=data, x='Real', y='Predicted')
plt.plot([y_22.min(), y_22.max()], [y_22.min(), y_22.max()], 'r--', lw=3)
plt.xlabel('Real')
plt.ylabel('Predicted')
plt.show()

best_gradient_boost.fit(X_train_subset, y_train)

y_pred_22 = best_gradient_boost.predict(X_test_22)

mae_22 = mean_absolute_error(y_22, y_pred_22)
rmse_22 = np.sqrt(mean_squared_error(y_22, y_pred_22))

print(f"MAE on players_22: {mae_22}")
print(f"RMSE on players_22: {rmse_22}")

data = pd.DataFrame({'Real': y_22, 'Predicted': y_pred_22})

sns.scatterplot(data=data, x='Real', y='Predicted')
plt.plot([y_22.min(), y_22.max()], [y_22.min(), y_22.max()], 'r--', lw=3)
plt.xlabel('Real')
plt.ylabel('Predicted')
plt.show()

with open('top_features_list.pkl', 'wb') as file:
    pickle.dump(top_features_list, file)



joblib.dump(best_random_forest, "model.pkl")

joblib.dump(scaler,'scaler.pkl')

print(type(best_random_forest))

"""# Random Forest provides the best model"""