import pandas as pd


import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from prettytable import PrettyTable
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#================================================================================================================================================
#                                                               Question 1
#================================================================================================================================================

#================================================================================================================================================
#                                                               Question 1a
#================================================================================================================================================

link='https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/Carseats.csv'
df=pd.read_csv(link)
plt.style.use("seaborn-whitegrid")
sns.set_style("whitegrid")
group_data=df.groupby(['ShelveLoc','US'])['Sales'].sum().unstack()
group_data.plot(kind='barh', stacked=False, figsize=(10, 6))
plt.xlabel('Sales')
plt.ylabel('Shelve Location')
plt.title('Shelve Location vs Sales')
plt.show()
plt.grid(True)
max_sales_location = group_data.sum(axis=1).idxmax()
max_sales_value = group_data.loc[max_sales_location]
print(f"Shelve location with the highest sales  '{max_sales_location}'.")
print(max_sales_value)

#================================================================================================================================================
#                                                               Question 1b
#================================================================================================================================================

df_one_hot_encode=pd.get_dummies(df,columns=['ShelveLoc','Urban','US'],drop_first=True)
converted_features = df_one_hot_encode[df_one_hot_encode.columns.difference(df.columns)]
print(converted_features.head())

#================================================================================================================================================
#                                                               Question 1c
#================================================================================================================================================

non_encoded_columns=['Sales','CompPrice','Income','Advertising','Population','Price','Age','Education']
scaler=StandardScaler()
df_standard=df_one_hot_encode;
df_standard[non_encoded_columns]=scaler.fit_transform(df_one_hot_encode[non_encoded_columns])
df_train,df_test=train_test_split(
    df_standard,test_size=0.20,shuffle=True,random_state=5805
)
print(df_train.head())
print(df_test.head())

#================================================================================================================================================
# Question 2
#================================================================================================================================================

#================================================================================================================================================
#                                                               Question 2a
#================================================================================================================================================

depend_variable='Sales'
independ_variables=[col for col in df_train.columns if col != depend_variable]

X_train=df_train[independ_variables]
X_test=df_test[independ_variables]
y_train=df_train[depend_variable]
y_test=df_test[depend_variable]

# creating an array for droped features
dropped_features=[]


X_model_train=sm.add_constant(X_train)
X_test=sm.add_constant(X_test)


model=sm.OLS(y_train,X_model_train).fit()
print(model.summary())
# Creating PrettyTables
feature_names=X_model_train.columns
summary_table=PrettyTable()
summary_table.field_names=['process Update','AIC','BIC','Adj R^2','p-value']
compare_table=PrettyTable()
compare_table.field_names=['Model','R^2','Adj-R^2','AIC','BIC','MSE']



p_value = model.pvalues['Population']
summary_table.add_row(['Population',round(model.aic,3),round(model.bic,3),round(model.rsquared_adj,3),round(p_value, 3)])



X_model_train.drop(columns=['Population'],inplace=True)
dropped_features.append('Population')
model=sm.OLS(y_train,X_model_train).fit()
p_value = model.pvalues['Education']
summary_table.add_row(['Education',round(model.aic,3),round(model.bic,3),round(model.rsquared_adj,3),round(p_value, 3)])
print(model.summary())



X_model_train.drop(columns=['Education'],inplace=True)
dropped_features.append('Education')
model=sm.OLS(y_train,X_model_train).fit()
p_value = model.pvalues['US_Yes']
summary_table.add_row(['US_Yes',round(model.aic,3),round(model.bic,3),round(model.rsquared_adj,3),round(p_value, 3)])
print(model.summary())



X_model_train.drop(columns=['US_Yes'],inplace=True)
dropped_features.append('US_Yes')
model=sm.OLS(y_train,X_model_train).fit()
p_value = model.pvalues['Urban_Yes']
summary_table.add_row(['Urban_Yes',round(model.aic,3),round(model.bic,3),round(model.rsquared_adj,3),round(p_value, 3)])
print(model.summary())



X_model_train.drop(columns=['Urban_Yes'],inplace=True)
dropped_features.append('Urban_Yes')
model=sm.OLS(y_train,X_model_train).fit()
print(model.summary())
print(summary_table)
print(f"The feature dropped{dropped_features}")
selected_features=X_model_train.columns
print(f"Features selected{selected_features}")

#================================================================================================================================================
#                                                               Question 2b
#================================================================================================================================================

model=sm.OLS(y_train,X_model_train).fit()
compare_table.add_row(['Backward Regression',round(model.rsquared,3),round(model.rsquared_adj,3),round(model.aic,3),round(model.bic,3),'0.984'])
print(model.summary())


coefficients = model.params


feature_names = X_model_train.columns



equation = f"Y = {coefficients['const']:.3f} "  # Access the intercept as 'const'
for feature, coefficient in zip(feature_names, coefficients[1:]):
    equation += f"+ {coefficient:.3f} * {feature} "


print("Final Regression Equation:")
print(equation)


#================================================================================================================================================
#                                                               Question 2c
#================================================================================================================================================


final_features=X_model_train.columns
X_model_test=X_test[final_features]
y_pred=model.predict(X_model_test)


y_scaler=StandardScaler()
y_scaler.fit_transform(df[['Sales']])
y_test_destand=y_scaler.inverse_transform(y_test.values.reshape(-1,1)).flatten()
y_pred_2d=y_pred.values.reshape(-1,1)
y_pred_destand=y_scaler.inverse_transform(y_pred_2d)

#================================================================================================================================================
#                                                               Question 2ad
#================================================================================================================================================

mse_1 = mean_squared_error(y_test_destand,y_pred_destand)
print(f"Mean Squared Error (MSE): {mse_1:.3f} ")

#================================================================================================================================================
#                                                   Question 6
#============================================================================================================================================
predictions = model.predict(X_model_test)
pred_int = model.get_prediction(X_model_test).summary_frame(alpha=0.05)
predictions_original=y_scaler.inverse_transform(predictions.values.reshape(-1,1)).flatten()
# Extract the lower and upper prediction interval bounds
lower_pred = pred_int['obs_ci_lower']
upper_pred = pred_int['obs_ci_upper']
lower_pred_orig=y_scaler.inverse_transform(lower_pred.values.reshape(-1,1)).flatten()
upper_pred_orig=y_scaler.inverse_transform(upper_pred.values.reshape(-1,1)).flatten()



plt.figure(figsize=(10, 6))
plt.plot(predictions_original, label="Predicted Sales", color='blue')
plt.fill_between(np.arange(len(predictions)), lower_pred_orig, upper_pred_orig, color='blue', alpha=0.5,
                 label="95% Prediction Interval")
plt.legend()
plt.xlabel("Observation")
plt.ylabel("Sales")
plt.title("Sales Prediction with 95% Prediction Interval")
plt.grid(True)
plt.show()


plt.figure(figsize=(10,8))
plt.plot(y_test_destand,label='Test Sales Values')
plt.plot(y_pred_destand,label='Predicted Sales Values')
plt.xlabel('Samples')
plt.ylabel('Sales')
plt.title('Test Sales VS Predicted Sales Values')
plt.grid(True)
plt.legend()
plt.show()

#================================================================================================================================================
#                                                           Question 3
#================================================================================================================================================


X_scaled = df_standard.drop(columns='Sales')
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
# explained_variance_ratio = pca.explained_variance_ratio_
# cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
# n_components_range = np.arange(1, 11)
# n_components_needed = np.argmax(cumulative_variance_ratio >= 0.90) + 1
# print("Number of components needed to explain 90% of the variance:", n_components_needed)
# plt.figure(figsize=(10,8))
# x=np.arange(1,12,step=1)
# plt.plot(x,cumulative_variance_ratio,marker='o', linestyle='--', color='b')
# plt.xlabel('Number of Components')
# plt.ylabel('Cummulative Variance %')
# plt.xticks(np.arange(1, 12, step=1))
# plt.axhline(y=0.90, color='b', linestyle='-')
# plt.axvline(x=7,color='r',linestyle='-')
# plt.grid(True)
# plt.show()



#================================================================================================================================================
#                                                               Question 3
#================================================================================================================================================

explained_var_ratio = pca.explained_variance_ratio_
cumulative_var_ratio = np.cumsum(explained_var_ratio)
num_features_90_percent = np.where(cumulative_var_ratio > 0.9)[0][0] + 1
print("Number of features needed to explain more than 90% of the dependent variance:", num_features_90_percent)

#================================================================================================================================================
#                                                               Question 3b
#================================================================================================================================================

plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, len(cumulative_var_ratio) + 1, 1), cumulative_var_ratio, label="Cumulative Explained Variance")
plt.xticks(np.arange(1, len(cumulative_var_ratio) + 1, 1))
plt.xlabel("Number of Features")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.legend()
plt.title("Cumulative Explained Variance vs. Number of Features")
plt.show()

# 3c. Draw a vertical line and horizontal line showing the exact 90% threshold and the corresponding number of features.
plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, len(cumulative_var_ratio) + 1, 1), cumulative_var_ratio, label="Cumulative Explained Variance")
plt.axvline(x=num_features_90_percent, color="r", linestyle="--", label="90% Threshold")
plt.axhline(y=0.9, color="r", linestyle="--")
plt.xticks(np.arange(1, len(cumulative_var_ratio) + 1, 1))
plt.xlabel("Number of Features")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.legend()
plt.title("Cumulative Explained Variance vs. Number of Features with 90% Threshold")
plt.show()

#================================================================================================================================================
# Question 4
#================================================================================================================================================

# 4. a
X_rf=df_one_hot_encode.drop(columns=[depend_variable])
y_rf=df[depend_variable]
rf_model = RandomForestRegressor(n_estimators=100, random_state=5805)
rf_model.fit(X_rf, y_rf)
feature_importances = rf_model.feature_importances_
feature_labels = X_rf.columns
### sorting feature Importances ###
sorted_indices = np.argsort(feature_importances)[::-1]
sorted_importances = feature_importances[sorted_indices]
sorted_labels = feature_labels[sorted_indices]
### plot the feature importance ##
plt.figure(figsize=(10,6))
plt.barh(range(len(sorted_labels)),sorted_importances,align='center')
plt.yticks(range(len(sorted_labels)),sorted_labels)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importances (Random Forest)')
plt.gca().invert_yaxis()

plt.show()

#================================================================================================================================================
#                                                               Question 4c
#================================================================================================================================================

# Selecting Threshold as 0.02 and dropping Urban_Yes and US_Yes
X_model_train_rf=sm.add_constant(X_train)
X_test=sm.add_constant(X_test)
dropped_features_rfs=[]


X_model_train_rf.drop(columns=['US_Yes'],inplace=True)
dropped_features_rfs.append('US_Yes')
model=sm.OLS(y_train,X_model_train_rf).fit()
X_model_train_rf.drop(columns=['Urban_Yes'],inplace=True)
dropped_features_rfs.append('Urban_Yes')
model=sm.OLS(y_train,X_model_train_rf).fit()
print(model.summary())

#================================================================================================================================================
#                                                               Question 4a
#================================================================================================================================================

print(f'The dropped features are {dropped_features_rfs}')
selected_features_rfs=X_model_train_rf.columns
print(f'The final Selected features are :{selected_features_rfs}')
compare_table.add_row(['Random Forest',round(model.rsquared,3),round(model.rsquared_adj,3),round(model.aic,3),round(model.bic,3),'0.974'])

#================================================================================================================================================
#                                                               Question 4d
#================================================================================================================================================

final_features_rf=X_model_train_rf.columns
X_model_test_rf=X_test[final_features_rf]
y_pred_rf=model.predict(X_model_test_rf)

y_scaler=StandardScaler()
y_scaler.fit_transform(df[['Sales']])
y_test_dest=y_scaler.inverse_transform(y_test.values.reshape(-1,1)).flatten()
y_pred_2d_rf=y_pred_rf.values.reshape(-1,1)
y_pred_destand_rf=y_scaler.inverse_transform(y_pred_2d_rf)



plt.figure(figsize=(10,8))
plt.plot(y_test_dest,label='Test Sales Values')
plt.plot(y_pred_destand_rf,label='Predicted Sales Values')
plt.xlabel('Samples')
plt.ylabel('Sales')
plt.title('Test Sales VS Predicted Sales Values')
plt.grid(True)
plt.legend()
plt.show()

#================================================================================================================================================
#                                                               Question 4e
#================================================================================================================================================

mse_2 = mean_squared_error(y_test_destand,y_pred_destand_rf)
print(f"Mean Squared Error (MSE): {mse_2:.3f} ")

#================================================================================================================================================
#                                                               Question 5
#================================================================================================================================================

print(compare_table)


#================================================================================================================================================
#                                                                Question 7
#================================================================================================================================================

#================================================================================================================================================
#                                                                Question 7a
#================================================================================================================================================


X_pr_price = df['Price']
y_pr = df['Sales']
X_pr_price = X_pr_price.values.reshape(-1, 1)
y_pr=y_pr.values.reshape(-1,1)
param_grid = {'polynomialfeatures__degree': np.arange(1, 15)}

model = make_pipeline(PolynomialFeatures(), LinearRegression())

grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(np.array(X_pr_price).reshape(-1,1),np.array(y_pr).reshape(-1,1))

#================================================================================================================================================
#                                                                Question 7b
#================================================================================================================================================

best_degree = grid_search.best_params_['polynomialfeatures__degree']
print("Optimum Order (n):", best_degree)

#================================================================================================================================================
#                                                                Question 7c
#================================================================================================================================================

res = grid_search.cv_results_
rmse = np.sqrt(-res['mean_test_score'])
plt.plot(range(1, 15), rmse)
plt.xlabel('Polynomial Degree (n)')
plt.ylabel('RMSE')
plt.title('RMSE vs. Polynomial Degree')
plt.grid(True)
plt.show()

#================================================================================================================================================
#                                                                Question 7d
#================================================================================================================================================


X_train_pre, X_test_pre, y_train_pre, y_test_pre = train_test_split(X_pr_price, y_pr, test_size=0.2, random_state=5805)

# polynomial regression model with the optimum degree
optimal_model = make_pipeline(PolynomialFeatures(degree=best_degree), LinearRegression())
optimal_model.fit(X_train_pre, y_train_pre)

y_pred_pre = optimal_model.predict(X_test_pre)
# Plotting test set versus predicted sales
plt.figure(figsize=(10,8))
plt.plot(y_test_pre,label='Test Set')
plt.plot(y_pred_pre,label='Predicted Set')
plt.xlabel('Samples')
plt.ylabel('Sales')
plt.title('Test Set VS Predicted Set')
plt.grid(True)
plt.legend()
plt.show()

#================================================================================================================================================
#                                                                Question 7e
#================================================================================================================================================

MSE = mean_squared_error(y_test_pre, y_pred_pre)
print(f"Mean Squared Error (MSE): {MSE:.3f}")
