
#=======================================================================================================
#                                       Imports
#=======================================================================================================

import pandas as pd

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.linalg import svd

import warnings

# Ignore DeprecationWarnings related to the mentioned code
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pandas.core.nanops")
# Ignore RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore")

# Your code here

# Reset the warning filter to its default behavior if needed
# warnings.resetwarnings()


# Your code goes here

# Reset the warning filter (optional, use it if you want to re-enable warnings)
warnings.filterwarnings("default")

#=======================================================================================================
#                                     Printing the head and understanding the dataset
#=======================================================================================================
housing = pd.read_csv('immo_data 2.csv')

print(housing.head())

housing.info()

housing.describe()

# removing the rows which have null or 0 values for totalRent as it is the output variable

housing = housing[housing['totalRent'].notna()]

print(housing.columns)
# removing unnecessary columns
housing.drop(columns=['livingSpaceRange','street','description','facilities','geo_krs','geo_plz','scoutId','regio1','telekomUploadSpeed','telekomTvOffer','pricetrend','regio3','noRoomsRange','picturecount','geo_bln','date',\
    'houseNumber','streetPlain','firingTypes','yearConstructedRange'],inplace=True)



#=======================================================================================================
#                                     Dealing with Nans
#=======================================================================================================


housing.dropna(subset=['totalRent'],inplace=True)
#replacing the na values with other
housing['condition'].fillna("Other", inplace=True)
# Specify the conditions to be grouped into 'Other'


others_condition = housing['condition'].value_counts().tail(3).index
# Create a dictionary to map values to 'Other' if they are in the specified conditions
condition_mapping = {condition: 'Other' for condition in others_condition}
# Apply the mapping using the map function
housing['condition'] = housing['condition'].map(condition_mapping).fillna(housing['condition'])
# Display the value counts after grouping
print(housing['condition'].value_counts())


# Specify the number of regions to keep and replace the rest with 'Other'
top_regions = housing['regio2'].value_counts().head(20).index
replace_dict = {region: 'Other' for region in housing['regio2'].unique() if region not in top_regions}
# Replace values using the mapping dictionary
housing['regio2'] = housing['regio2'].replace(replace_dict)
# Display the value counts after grouping
print(housing['regio2'].value_counts())


#=======================================================================================================
#                                     Dealing with missing values
#=======================================================================================================
from datetime import date
# Replace 'housing' with your actual DataFrame
# Calculate the mean by 'condition' for the 'yearConstructed' column
mean_by_condition = housing.groupby('condition')['yearConstructed'].transform('mean')
# Fill missing values in 'yearConstructed' with the calculated mean values
housing['yearConstructed'] = housing['yearConstructed'].fillna(mean_by_condition).round(0)
housing['numberOfYear'] = date.today().year - housing["yearConstructed"]
# Specify the top 25 cities
region_list = housing['regio2'].value_counts().head(25).index
# Filter the DataFrame to keep only the rows where 'regio2' is in the top 25 cities
housing = housing[housing['regio2'].isin(region_list)]
# Display the value counts after filtering
print(housing['regio2'].value_counts())


#=======================================================================================================
#                                     Dealing with Duplicate Values
#=======================================================================================================

import pandas as pd

# Assuming your DataFrame is named 'df'
# df = ...

# Check for duplicate rows
# duplicate_rows = housing[housing.duplicated()]
#
# # Print duplicate rows
# print("******** Duplicate Rows except first occurrence: ********")
# print(duplicate_rows)
#
# # Optionally, you can count the number of duplicate rows
# num_duplicate_rows = len(duplicate_rows)
# print(f"\nNumber of duplicate rows: {num_duplicate_rows}")
#
#
# # Remove duplicate rows, keeping the first occurrence
# housing = housing.drop_duplicates()
#
# # Print the DataFrame without duplicates
# print("DataFrame without duplicates:")
# print(housing)
#
#
# print("******** DataFrame Number of Duplicates *********")
# duplicate_rows = housing[housing.duplicated()]
# num_duplicate_rows = len(duplicate_rows)
# print(f"\nNumber of duplicate rows: {num_duplicate_rows}")

#=======================================================================================================
#                                     Dealing with Outliers
#=======================================================================================================
import matplotlib.pyplot as plt
housing = housing.query(
    '200 < baseRent < 8000 and 200 < totalRent < 9000 and totalRent > baseRent and (totalRent - baseRent) < 500'
)
# Display the filtered DataFrame
print(housing)


plt.scatter(housing['totalRent'], housing['baseRent'])
plt.xlabel('totalRent')
plt.ylabel('Base Rent')
plt.title('Scatter Plot of totalRent vs. baseRent')
plt.show()


# Assuming 'housing' is your DataFrame
# Replace 'housing' with your actual DataFrame
# Use the query method for filtering
housing = housing.query('10 < livingSpace < 400')
# Display the filtered DataFrame
print(housing)
plt.scatter(housing['baseRent'], housing['livingSpace'])
plt.xlabel('Base Rent')
plt.ylabel('Living Space')
plt.title('Scatter Plot of Base Rent vs. Living Space')
plt.show()
# Assuming 'housing' is your DataFrame
# Replace 'housing' with your actual DataFrame
housing = housing.assign(PricePerSQ=housing['baseRent'] / housing['livingSpace'],
                         additioncost=housing['totalRent'] - housing['baseRent'])
# Display the DataFrame with the new columns
print(housing)


plt.scatter(housing['totalRent'], housing['PricePerSQ'])
plt.xlabel('Total Rent')
plt.ylabel('PricePerSQ')
plt.title('Scatter Plot of Total Rent vs. PricePerSQ')
plt.show()
# Assuming 'df' is your DataFrame
# Replace 'df' with your actual DataFrame
# Use the query method for filtering
housing = housing.query('serviceCharge < 1000')
# Display the filtered DataFrame
print(housing)



plt.scatter(housing['totalRent'], housing['serviceCharge'])
plt.xlabel('Total Rent')
plt.ylabel('serviceCharge')
plt.title('Scatter Plot of Total Rent vs. serviceCharge')
plt.show()
import pandas as pd
# Assuming 'housing' is your DataFrame
# Replace 'housing' with your actual DataFrame
# Clip 'floor' values to be within the range -1 to 20
housing['floor'] = housing['floor'].clip(lower=-1, upper=20)
# Fill missing values in 'heatingType' and 'typeOfFlat' with the mode
housing['heatingType'].fillna(housing['heatingType'].mode()[0], inplace=True)
housing['typeOfFlat'].fillna(housing['typeOfFlat'].mode()[0], inplace=True)
# Filter 'heatingType' based on the top 10 values
heating_list = housing['heatingType'].value_counts().head(10).index
housing = housing[housing['heatingType'].isin(heating_list)]
# Display the updated DataFrame
print(housing)

#dropping baseRent as it is very much similar and correlated to the target variable
housing.drop(['baseRent'],axis=1,inplace=True)
# dropping few more columns afer plotting the heatmap for the data
housing.drop(['cellar','floor','garden'],axis=1,inplace=True)
# Assuming 'housing' is your DataFrame
# Replace 'housing' with your actual DataFrame
predict_housing = pd.DataFrame(housing)
predict_housing.head(10)
predict_housing.drop(columns=['yearConstructed'],inplace=True)

dataset_apriori = housing.copy()
#=======================================================================================================
#                                     Standardizing the the numerical features within the data
#=======================================================================================================


from sklearn.preprocessing import StandardScaler

# Assuming 'housing' is your DataFrame
# Replace 'housing' with your actual DataFrame

# Select numeric columns (excluding 'totalRent')
numeric_columns = housing.select_dtypes(include=['int64', 'float64']).columns.difference(['totalRent'])

# Create a copy of the DataFrame to avoid modifying the original
standardized_housing = housing.copy()

# Use StandardScaler to standardize selected numeric columns
scaler = StandardScaler()
housing[numeric_columns] = scaler.fit_transform(housing[numeric_columns])

# Display the updated DataFrame
print(housing)


#=====================================================================================================================================================
#                                     One Hot Encoding categorical Values
#===================================================================================================================================================
# Assuming 'housing' is your DataFrame
# Replace 'housing' with your actual DataFrame

# Select categorical and boolean columns
categorical_columns = housing.select_dtypes(include=['object', 'bool']).columns

# Use get_dummies to create dummy variables
dummies_feature = pd.get_dummies(housing[categorical_columns])

# Concatenate the dummy variables with the original DataFrame
housing_with_dummies = pd.concat([housing, dummies_feature], axis=1)

# Drop the original categorical columns (if needed)
housing_with_dummies = housing_with_dummies.drop(columns=categorical_columns)

# Display the DataFrame with dummy variables
print(housing_with_dummies)
# Assuming 'housing' is your DataFrame
# Replace 'housing' with your actual DataFrame

# Use get_dummies with drop_first=True to create dummy variables and drop original columns
housing = pd.get_dummies(housing, columns=categorical_columns, drop_first=True)

# Display the DataFrame with dummy variables
print(housing.shape)

import pandas as pd

# Assuming 'housing' is your DataFrame
# Replace 'housing' with your actual DataFrame
print(" ****** Null values before dropping ********")
print(housing.isnull().sum())
# Fill NaN values in numerical columns with mean
numerical_columns = housing.select_dtypes(include=['float64', 'int64']).columns
housing[numerical_columns] = housing[numerical_columns].fillna(housing[numerical_columns].mean())

# Fill NaN values in categorical columns with mode
categorical_columns = housing.select_dtypes(include=['object']).columns

for col in categorical_columns:
    mode_value = housing[col].mode().iloc[0]
    housing[col] = housing[col].fillna(mode_value)

# Display the updated DataFrame
print(housing)

copy_df = housing.copy()

# Select features (X) and target variable (y)
housing = housing.dropna()
print("********  Null values after dropping  *********")
print(housing.isnull().sum())
# Assuming 'housing' is your DataFrame
# Replace 'housing' with your actual DataFrame
from sklearn.model_selection import train_test_split



print(housing.shape)
X = housing.drop(columns=['totalRent']).values  # Drop the 'totalRent' column for features
y = housing['totalRent'].values  # Select only the 'totalRent' column as the target variable

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)


#=====================================================================================================================================================
#                                     Dimentionality Reduction
#===================================================================================================================================================



#=====================================================================================================================================================
#                                     1. SVD - (Singular Value Decomposition
#===================================================================================================================================================

import statsmodels.api as sm
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

# Assuming 'housing' is your dataframe
# Replace 'target_column' with the actual name of your target variable column
target_column = 'totalRent'

# Separate features (X) and target variable (y)
X = housing.drop(columns=[target_column]).values
y = housing[target_column].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5805)

# Perform Truncated SVD for dimensionality reduction
svd = TruncatedSVD(n_components=5)  # Choose the desired number of components
X_train_svd = svd.fit_transform(X_train)

# Fit OLS model using the selected features after SVD
X_with_intercept = sm.add_constant(X_train_svd)
ols_model_svd = sm.OLS(y_train, X_with_intercept).fit()

# Print OLS summary
print(ols_model_svd.summary())


#=====================================================================================================================================================
#                                     2. PCA - Principal Component Analysis
#===================================================================================================================================================
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

pca = PCA()
X_pca = pca.fit_transform(X)
explained_var_ratio = pca.explained_variance_ratio_
cumulative_var_ratio = np.cumsum(explained_var_ratio)
num_features_90_percent = np.where(cumulative_var_ratio > 0.90)[0][0] + 1
print("Number of features needed to explain more than 90% of the dependent variance:", num_features_90_percent)

# 3b. Plot the cumulative explained variance versus the number of features.
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
plt.axhline(y=0.90, color="r", linestyle="--")
plt.xticks(np.arange(1, len(cumulative_var_ratio) + 1, 1))
plt.xlabel("Number of Features")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.legend()
plt.title("Cumulative Explained Variance vs. Number of Features with 90% Threshold")
plt.show()


# the ols summary after using PCA for dimentionality reduction

X_pca_selected = X_pca[:, :num_features_90_percent]

# Fit OLS model
X_with_intercept = sm.add_constant(X_pca_selected)
ols_model = sm.OLS(y, X_with_intercept).fit()

# Print OLS summary
print(ols_model.summary())



#=====================================================================================================================================================
#                                     3. Random Forest Analysis
#===================================================================================================================================================
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'housing' is your original DataFrame containing the data

# Subset the data
selected_rows = housing.iloc[:50000, :]
X_rf = selected_rows.drop(columns=['totalRent']).values
y_rf = np.log1p(selected_rows['totalRent'].values)

# Get the original feature names
feature_names = selected_rows.drop(columns=['totalRent']).columns.tolist()

# Split the data into training and testing sets
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X_rf, y_rf, test_size=0.2, random_state=5805, shuffle=True
)

# Perform Random Forest Regression
rfc = RandomForestRegressor(n_estimators=100, random_state=5805, n_jobs=-1)
rfc.fit(X_train_rf, y_train_rf)

# Plot feature importance
def plot_feature_importance(importance, names, model_type, threshold=0.0001):
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    data = {"feature_names": feature_names, "feature_importance": feature_importance}
    fi_df = pd.DataFrame(data)

    fi_df.sort_values(by=["feature_importance"], ascending=False, inplace=True)

    plt.figure(figsize=(10, 8))
    sns.barplot(x=fi_df["feature_importance"], y=fi_df["feature_names"])
    plt.title(f"{model_type} Feature Importance")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Names")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    removed_features = fi_df[fi_df["feature_importance"] < threshold]["feature_names"]
    print("Removed Features:", removed_features.tolist())

    selected_features = fi_df[fi_df["feature_importance"] >= threshold]["feature_names"]
    print("Selected Features:", selected_features.tolist())

    return removed_features, selected_features

# Plot and select features
removed, selected = plot_feature_importance(rfc.feature_importances_, feature_names, "RANDOM FOREST")

print("Removed Features:", removed.tolist())
print("Selected Features:", selected.tolist())

# Drop insignificant features
X_train_rf_df = pd.DataFrame(X_train_rf, columns=feature_names)
X_test_rf_df = pd.DataFrame(X_test_rf, columns=feature_names)

# Drop insignificant features based on the feature selection from the Random Forest
X_train_rf_df.drop(removed, axis=1, inplace=True)
X_test_rf_df.drop(removed, axis=1, inplace=True)

# Perform regression OLS
lin_reg_rf = sm.OLS(y_train_rf, sm.add_constant(X_train_rf_df)).fit()
print(lin_reg_rf.summary())

# Predict on the test set
y_pred_rf = lin_reg_rf.predict(sm.add_constant(X_test_rf_df))

# Do reverse transformation based on formula
y_test_hat_rf = np.expm1(y_pred_rf)

# Calculate MSE
def mse_rf(y_true, y_pred):
    return round(np.mean((y_true - y_pred) ** 2), 3)

# Example of usage:
print("MSE:", mse_rf(np.expm1(y_test_rf), y_test_hat_rf))

# Create a DataFrame for model statistics
table_2 = pd.DataFrame(columns=["AIC", "BIC", "Adj. R2", "MSE"])

# Populate the DataFrame with Random Forest Regression metrics
table_2.loc["Random Forest Regression", "AIC"] = lin_reg_rf.aic.round(3)
table_2.loc["Random Forest Regression", "BIC"] = lin_reg_rf.bic.round(3)
table_2.loc["Random Forest Regression", "Adj. R2"] = lin_reg_rf.rsquared_adj.round(3)
table_2.loc["Random Forest Regression", "MSE"] = mse_rf(np.expm1(y_test_rf), y_test_hat_rf)

# Display the model statistics
print(table_2)

#=====================================================================================================================================================
#                                     4. VIF
#===================================================================================================================================================
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

housing = housing.drop(removed, axis=1)
X = housing.drop(columns=['totalRent']).values  # Drop the 'totalRent' column for features
y = housing['totalRent'].values  # Select only the 'totalRent' column as the target variable

vif_data1 = pd.DataFrame()
vif_data1['feature'] = housing.drop(columns=['totalRent']).columns
vif_data1['VIF'] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]

# Print VIF results for the initial set of features
print("VIF Results (Before Dropping a Feature):")
print(vif_data1)

# Set the threshold for VIF
vif_threshold = 5

# Identify features with high VIF
high_vif_features = vif_data1[vif_data1['VIF'] > vif_threshold]['feature'].tolist()

# Print features with high VIF
print("Features with High VIF:", high_vif_features)

# Drop features with high VIF from the DataFrame 'housing'
housing = housing.drop(columns=high_vif_features)

# Print the DataFrame after removing features with high VIF
print("\nDataFrame After Removing Features with High VIF:")
print(housing)


#=====================================================================================================================================================
#                                     Sample Covariance Matrix Heatmap
#===================================================================================================================================================
cov_matrix = housing.cov()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Sample Covariance Matrix Heatmap')
plt.show()

#=====================================================================================================================================================
#                                     Sample Pearson Correlation Coefficients Heatmap
#===================================================================================================================================================
corr_matrix = housing.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Sample Pearson Correlation Coefficients Heatmap')
plt.show()







#=====================================================================================================================================================
#                               ************   PHASE 2 ************
#===================================================================================================================================================


#=====================================================================================================================================================
#                               Linear Regression
#===================================================================================================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from prettytable import PrettyTable

# housing = housing.drop(removed, axis=1)
from sklearn.linear_model import LinearRegression
# X = housing.drop(columns=['totalRent']).values  # Drop the 'totalRent' column for features
# y = housing['totalRent'].values  # Select only the 'totalRent' column as the target variable

# # Split the data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)
# lr = LinearRegression()
# lr.fit(x_train, y_train)
# y_predict = lr.predict(x_test)
# # df = pd.DataFrame({'Actual':y_test, 'Predicted':y_predict})
# # df
# from sklearn.metrics import r2_score
# r2_score(y_test, y_predict)
# Assuming housing is your DataFrame with the data

# Separate features (X) and target variable (y)
X = housing.drop(columns=['totalRent']).values
y = housing['totalRent'].values
scaler_y = StandardScaler()
y_standardized = scaler_y.fit_transform(y.reshape(-1, 1))

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y_standardized, test_size=0.20, random_state=5805)

# Create and train the model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Make predictions on the test set
y_predict_standardized = lr.predict(x_test)

# Destandardize the predictions
y_predict_dest = scaler_y.inverse_transform(y_predict_standardized)
y_test_dest = scaler_y.inverse_transform(y_test)
y_train_dest = scaler_y.inverse_transform(y_train)

threshold = 8000
replacement_value = 7500

# Apply the threshold and rounding
y_predict_dest = np.where(y_predict_dest > threshold, replacement_value, y_predict_dest)


threshold = 0
replacement_value = 200
y_predict_dest = np.where(y_predict_dest < threshold, replacement_value, y_predict_dest)
# Evaluate the model
mse_1 = mean_squared_error(y_test_dest, y_predict_dest)
print(f"Mean Squared Error (MSE): {mse_1:.3f}")

from sklearn.metrics import r2_score
r2_score(y_test_dest, y_predict_dest)

# Print Predicted and Actual Values
predictions_df = pd.DataFrame({'Actual': y_test_dest.flatten(), 'Predicted': y_predict_dest.flatten()})
print("\nActual vs Predicted Values:")
print(predictions_df.head())

# #=============================================================================
#                 PLOTTING TRAIN TEST PREDICT IN ONE PLOT
# #=============================================================================

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

# Plot Train, Test, and Predicted Values
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_train_dest)), y_train_dest, label='Train', alpha=0.5)
plt.scatter(range(len(y_train), len(y_train) + len(y_test_dest)), y_test_dest, label='Test', alpha=0.5)
plt.scatter(range(len(y_train), len(y_train) + len(y_test_dest)), y_predict_dest, label='Predict', alpha=0.5)
plt.legend()
plt.xlabel('Observations')
plt.ylabel('totalRent')
plt.title('Train, Test, and Predicted Values for Linear Regression')
plt.show()

# #=============================================================================
#                 PRETTY TABLE DISPLAYING THE VALUES
# #=============================================================================
from prettytable import PrettyTable

# Creating PrettyTables
# Calculate AIC and BIC using statsmodels
x_train_sm = sm.add_constant(x_train)
lr_sm = sm.OLS(y_train, x_train_sm).fit()
aic = lr_sm.aic
bic = lr_sm.bic

# Evaluate the model
mse = mean_squared_error(y_test_dest, y_predict_dest)
r2 = r2_score(y_test_dest, y_predict_dest)


compare_table = PrettyTable()
compare_table2 = PrettyTable()
adjusted_r2 = lr_sm.rsquared_adj
compare_table.field_names = ['Model', 'R^2','Adjusted R^2', 'MSE', 'AIC', 'BIC']
compare_table2.field_names = ['Model', 'R^2','Adjusted R^2', 'MSE', 'AIC', 'BIC']
compare_table.add_row(['Linear Regression', round(r2, 4),round(adjusted_r2, 4), round(mse, 4), round(aic, 4), round(bic, 4)])
compare_table2.add_row(['Linear Regression', round(r2, 4),round(adjusted_r2, 4), round(mse, 4), round(aic, 4), round(bic, 4)])



print("\nModel Comparison Table:")
print(compare_table)


# #=============================================================================
#                 T Test
# #=============================================================================


# T-Test for individual coefficients
t_test_results = lr_sm.t_test(np.identity(len(lr_sm.params)))
t_test_p_values = t_test_results.pvalue
print('T-Test p-values:\n', t_test_p_values)


# #=============================================================================
#                 F Test
# #=============================================================================

# F-Test (ANOVA)
f_statistic = lr_sm.fvalue
p_value_anova = lr_sm.f_pvalue
print(f'F-Test (ANOVA): F-statistic = {f_statistic}, p-value = {p_value_anova}')
# #=============================================================================
#                 Confidence interval
# #=============================================================================
# Confidence Intervals
conf_intervals = lr_sm.conf_int()
print(f'Confidence Intervals:\n{conf_intervals}')

# Adjusted R-Square
adjusted_r2 = lr_sm.rsquared_adj
print(f'Adjusted R-Square: {adjusted_r2}')

# Make predictions on the test set
X_with_intercept_test = sm.add_constant(x_test)
y_predict = lr_sm.predict(X_with_intercept_test)












# #====================================================================================================================
#                2. Backward stepwise regression
# #====================================================================================================================

import statsmodels.api as sm

# Assuming you have a DataFrame 'housing' with your data
# X should be the independent variables, and y should be the dependent variable

def backward_stepwise_selection(X, y, threshold_out=0.05, verbose=True):
    included = list(X.columns)
    eliminated = []

    while True:
        changed = False

        model = sm.OLS(y, sm.add_constant(X[included])).fit()
        pvalues = model.pvalues.iloc[1:]  # Exclude the intercept

        worst_pval = pvalues.max()

        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            eliminated.append(worst_feature)

            if verbose:
                print(f'Drop {worst_feature} with p-value {worst_pval}')

        if not changed:
            break

    return included, eliminated

# Example usage

X = copy_df.drop(columns=['totalRent'])  # Drop the 'totalRent' column for features
y = copy_df['totalRent'].values

selected_features_backward, eliminated_features = backward_stepwise_selection(X, y)
print("Selected features (backward stepwise):", selected_features_backward)
print("Eliminated features (backward stepwise):", eliminated_features)

# Fit the final model
final_model_backward = sm.OLS(y, sm.add_constant(X[selected_features_backward])).fit()

# Calculate adjusted R-squared
adjusted_r_squared_backward = final_model_backward.rsquared_adj
print("Adjusted R-squared (backward stepwise):", adjusted_r_squared_backward)


new_df_for_backwardRegg = copy_df[selected_features_backward + ['totalRent']]

# Alternatively, if you want to create a DataFrame without the eliminated features
# new_df = copy_df.drop(columns=eliminated_features)

# Display the new DataFrame
print("New DataFrame with selected features:")
print(new_df_for_backwardRegg.head())


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm


# housing = housing.drop(removed, axis=1)
from sklearn.linear_model import LinearRegression
# X = housing.drop(columns=['totalRent']).values  # Drop the 'totalRent' column for features
# y = housing['totalRent'].values  # Select only the 'totalRent' column as the target variable

# # Split the data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)
# lr = LinearRegression()
# lr.fit(x_train, y_train)
# y_predict = lr.predict(x_test)
# # df = pd.DataFrame({'Actual':y_test, 'Predicted':y_predict})
# # df
# from sklearn.metrics import r2_score
# r2_score(y_test, y_predict)
# Assuming housing is your DataFrame with the data

# Separate features (X) and target variable (y)
X = new_df_for_backwardRegg.drop(columns=['totalRent']).values
y = new_df_for_backwardRegg['totalRent'].values
scaler_y = StandardScaler()
y_standardized = scaler_y.fit_transform(y.reshape(-1, 1))

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y_standardized, test_size=0.20, random_state=5805)

# Create and train the model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Make predictions on the test set
y_predict_standardized = lr.predict(x_test)

# Destandardize the predictions
y_predict_dest = scaler_y.inverse_transform(y_predict_standardized)
y_test_dest = scaler_y.inverse_transform(y_test)
y_train_dest = scaler_y.inverse_transform(y_train)

threshold = 0
replacement_value = 200

# Apply the threshold and rounding
y_predict_dest = np.where(y_predict_dest < threshold, replacement_value, y_predict_dest)


# Evaluate the model
mse_1 = mean_squared_error(y_test_dest, y_predict_dest)
print(f"Mean Squared Error (MSE): {mse_1:.3f}")

from sklearn.metrics import r2_score
r2_score(y_test_dest, y_predict_dest)

# Print Predicted and Actual Values
predictions_df = pd.DataFrame({'Actual': y_test_dest.flatten(), 'Predicted': y_predict_dest.flatten()})
print("\nActual vs Predicted Values:")
print(predictions_df.head())

# #=============================================================================
#                 PLOTTING TRAIN TEST PREDICT IN ONE PLOT
# #=============================================================================

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

# Plot Train, Test, and Predicted Values
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_train_dest)), y_train_dest, label='Train', alpha=0.5)
plt.scatter(range(len(y_train), len(y_train) + len(y_test_dest)), y_test_dest, label='Test', alpha=0.5)
plt.scatter(range(len(y_train), len(y_train) + len(y_test_dest)), y_predict_dest, label='Predict', alpha=0.5)
plt.legend()
plt.xlabel('Observations')
plt.ylabel('totalRent')
plt.title('Train, Test, and Predicted Values for Stepwise Regression')
plt.show()

# #=============================================================================
#                 PRETTY TABLE DISPLAYING THE VALUES
# #=============================================================================
from prettytable import PrettyTable

# Creating PrettyTables
# Calculate AIC and BIC using statsmodels
x_train_sm = sm.add_constant(x_train)
lr_sm = sm.OLS(y_train, x_train_sm).fit()
aic = lr_sm.aic
bic = lr_sm.bic

# Evaluate the model
mse = mean_squared_error(y_test_dest, y_predict_dest)
r2 = r2_score(y_test_dest, y_predict_dest)


compare_table = PrettyTable()
adjusted_r2 = lr_sm.rsquared_adj
compare_table.field_names = ['Model', 'R^2','Adjusted R^2', 'MSE', 'AIC', 'BIC']
compare_table.add_row(['Stepwise Regression', round(r2, 4),round(adjusted_r2, 4), round(mse, 4), round(aic, 4), round(bic, 4)])
compare_table2.add_row(['Stepwise Regression', round(r2, 4), round(adjusted_r2, 4), round(mse, 4), round(aic, 4), round(bic, 4)])




print("\nModel Comparison Table:")
print(compare_table)

print(compare_table2)


# #=============================================================================
#                 T Test
# #=============================================================================


# T-Test for individual coefficients
t_test_results = lr_sm.t_test(np.identity(len(lr_sm.params)))
t_test_p_values = t_test_results.pvalue
print('T-Test p-values:\n', t_test_p_values)


# #=============================================================================
#                 F Test
# #=============================================================================

# F-Test (ANOVA)
f_statistic = lr_sm.fvalue
p_value_anova = lr_sm.f_pvalue
print(f'F-Test (ANOVA): F-statistic = {f_statistic}, p-value = {p_value_anova}')
# #=============================================================================
#                 Confidence interval
# #=============================================================================
# Confidence Intervals
conf_intervals = lr_sm.conf_int()
print(f'Confidence Intervals:\n{conf_intervals}')

# Adjusted R-Square
adjusted_r2 = lr_sm.rsquared_adj
print(f'Adjusted R-Square: {adjusted_r2}')

# Make predictions on the test set
X_with_intercept_test = sm.add_constant(x_test)
y_predict = lr_sm.predict(X_with_intercept_test)

#=====================================================================================================================================================
#                               ************   PHASE 3 ************
#===================================================================================================================================================

#=====================================================================================================================================================
#                               Data Preprocessing
#===================================================================================================================================================

#=====================================================================================================================================================
#                               Binning of output variable and balancing the data
#===================================================================================================================================================
max_value = copy_df['totalRent'].max()
min_value = copy_df['totalRent'].min()

print(max_value)
print(min_value)

bin_edges = [0, 1000, np.inf]  # Adjust these as needed
category_names = [0, 1]

# Use pd.cut to assign categories based on the specified bins
housing['totalRent'] = pd.cut(
    housing['totalRent'], bins=bin_edges, labels=category_names, include_lowest=True, right=False
)

# Print the result
print(housing)
print(housing['totalRent'])

import seaborn as sns
sns.countplot(data=housing, x='totalRent')

 # Show the plot
plt.show()

#========Performing SMOTE to balance data ++++++++++++++++++++

from imblearn.over_sampling import SMOTE
# Separate features and labels
df = housing
X = housing.drop(columns=['totalRent']).values
y = housing['totalRent']

# Instantiate the SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)

# Apply SMOTE only on the features (X)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

resampled_df = pd.DataFrame(X_resampled, columns=housing.drop(columns=['totalRent']).columns)
resampled_df['totalRent'] = y_resampled

# Create a countplot for the balanced data after SMOTE
plt.figure(figsize=(8, 6))
sns.countplot(data=resampled_df, x='totalRent', palette='viridis')
plt.title('Countplot of totalRent After SMOTE')
plt.show()

# Select 20,000 rows from each class
class_1 = resampled_df[resampled_df['totalRent'] == 0].head(25000)
# class_2 = resampled_df[resampled_df['totalRent'] == 'Medium'].head(20000)
class_3 = resampled_df[resampled_df['totalRent'] == 1].head(25000)

# Concatenate the selected subsets to create the final dataset
# final_dataset = pd.concat([class_1, class_2, class_3])
final_dataset = pd.concat([class_1, class_3])
# Shuffle the dataset if needed
final_dataset = final_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
df_shuffled = final_dataset.sample(frac=1, random_state=42).reset_index(drop=True)




X = df_shuffled.drop(columns=['totalRent']).values  # Drop the 'totalRent' column for features
y = df_shuffled['totalRent'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805)
print(y_test.value_counts())


# #====================================================================================================================
#                                   Classifier 1 Naive Bayes Classification
# #====================================================================================================================


# #====================================================================================================================
#                                   1. Implementation of Binary Classifictaion using Naive Bayes classification
# #====================================================================================================================

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from prettytable import PrettyTable
from sklearn.preprocessing import label_binarize

# Assuming X_train, X_test, y_train, y_test are your training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from prettytable import PrettyTable
pt = PrettyTable()
pt2 = PrettyTable()
pt.field_names = ['Classifier', 'Accuracy', 'Confusion Matrix', 'Precision', 'Sensitivity or Recall', 'Specificity', 'F-score', 'AUC']
pt2.field_names = ['Classifier', 'Accuracy', 'Confusion Matrix', 'Precision', 'Sensitivity or Recall', 'Specificity', 'F-score', 'AUC']

# Gaussian Naive Bayes
naive_bayes = GaussianNB()
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=5805)
nb_acc = cross_val_score(naive_bayes, X_train, y_train, cv=stratified_kfold, scoring='accuracy')
print(f'Train accuracy of Gaussian Naive Bayes {round(np.mean(nb_acc), 2)}')
naive_bayes.fit(X_train, y_train)
y_train_pred = naive_bayes.predict(X_train)
y_test_pred = naive_bayes.predict(X_test)
nb_acc_test = accuracy_score(y_test, y_test_pred)
print(f'Test accuracy of Gaussian Naive Bayes {round(nb_acc_test, 2)}')
conf_matrix_nb = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = conf_matrix_nb.ravel()
specificity_nb = tn / (tn + fp)

y_prob_nb = naive_bayes.predict_proba(X_test)[:, 1] # Probability of the positive class
roc_auc_nb = roc_auc_score(y_test, y_prob_nb)
recall_nb= recall_score(y_test, y_test_pred)
precision_nb =  precision_score(y_test, y_test_pred)
f1_nb =  f1_score(y_test, y_test_pred)
pt.add_row(['Gaussian Naive Bayes', round(np.mean(nb_acc),4),
            conf_matrix_nb.round(4), round(np.mean(precision_nb),4),
            round(np.mean(recall_nb),4), round(np.mean(specificity_nb),4),
            round(np.mean(f1_nb),4), round(np.mean(roc_auc_nb),4)])
fpr_nb, tpr_nb, thresholds_svm_poly = roc_curve(y_test, y_prob_nb)
pt2.add_row(['Gaussian Naive Bayes', round(np.mean(nb_acc),4),
            conf_matrix_nb.round(4), round(np.mean(precision_nb),4),
            round(np.mean(recall_nb),4), round(np.mean(specificity_nb),4),
            round(np.mean(f1_nb),4), round(np.mean(roc_auc_nb),4)])
fpr_nb, tpr_nb, thresholds_svm_poly = roc_curve(y_test, y_prob_nb)
plt.figure(figsize=(8, 6))
plt.plot(fpr_nb, tpr_nb, linewidth=2, label="ROC curve of Gaussian Naive Bayes (area = {:.2f})".format(roc_auc_nb))
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve ')
plt.legend(loc="lower right")
plt.show()
print(pt)


# #====================================================================================================================
#                                   Classifier 2
# #====================================================================================================================



# #====================================================================================================================
#                                   2. K Nearest Neighbour
# #====================================================================================================================


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# Assuming X_train, y_train are your training data
error_rate = []

# Try different values of K
for i in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors=i)
    # Using cross-validation to evaluate performance
    score = cross_val_score(knn, X_train, y_train, cv=5)
    error_rate.append(1 - score.mean())

# Plotting the error rate
plt.figure(figsize=(10, 6))
plt.plot(range(1, 30), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()
optimal_k = error_rate.index(min(error_rate)) + 1

# Print optimal K value
print(f'The optimal K value is: {optimal_k}')





stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=5805)

from prettytable import PrettyTable
pt = PrettyTable()
pt.field_names = ['Classifier', 'Accuracy', 'Confusion Matrix', 'Precision', 'Sensitivity or Recall', 'Specificity', 'F-score', 'AUC']

param_grid = {
    'n_neighbors': [optimal_k],
    'weights': ['uniform'],
    'metric': ['manhattan', 'chebyshev'],
}
model_knn = KNeighborsClassifier()
grid_search = GridSearchCV(model_knn, param_grid, cv=stratified_kfold, scoring='accuracy')
grid_search.fit(X_train, y_train)
results = grid_search.cv_results_
k_values = results['param_n_neighbors'].data
mean_test_scores = results['mean_test_score']
error_rate = 1-mean_test_scores
# plt.plot(k_values, error_rate)
# plt.xlabel('K')
# plt.ylabel('Error rate')
# plt.title('Error rate vs K')
# plt.show()

print("Best Hyperparameters:", grid_search.best_params_)
model_knn = grid_search.best_estimator_
model_knn.fit(X_train, y_train)
y_train_pred = model_knn.predict(X_train)
y_test_pred = model_knn.predict(X_test)
knn_acc_train = accuracy_score(y_train, y_train_pred)
knn_acc_test = accuracy_score(y_test, y_test_pred)
print(f'Train accuracy of KNN {round(accuracy_score(y_train, y_train_pred),2)}')
print(f'Test accuracy of KNN {round(knn_acc_test,2)}')
conf_matrix_knn = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp =  conf_matrix_knn.ravel()
specificity_knn = tn/(tn+fp)
y_prob_knn =  model_knn.predict_proba(X_test)[:, 1] # Probability of the positive class
roc_auc_knn =  roc_auc_score(y_test, y_prob_knn)
recall_knn =  recall_score(y_test, y_test_pred)
precision_knn =  precision_score(y_test, y_test_pred)
f1_knn =  f1_score(y_test, y_test_pred)
pt.add_row(['KNN', round(np.mean(knn_acc_test),4), conf_matrix_knn.round(4), round(np.mean(precision_knn),4), round(np.mean(recall_knn),4), round(np.mean(specificity_knn),4), round(np.mean(f1_knn),4), round(np.mean(roc_auc_knn),4)])
pt2.add_row(['KNN', round(np.mean(knn_acc_test),4), conf_matrix_knn.round(4), round(np.mean(precision_knn),4), round(np.mean(recall_knn),4), round(np.mean(specificity_knn),4), round(np.mean(f1_knn),4), round(np.mean(roc_auc_knn),4)])



fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, y_prob_knn)
plt.figure(figsize=(8, 6))
plt.plot(fpr_knn, tpr_knn, linewidth=2, label="ROC curve of KNN (area = {:.2f})".format(roc_auc_knn))
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve ')
plt.legend(loc="lower right")
plt.show()
print(pt)


# #====================================================================================================================
#                                   Classifier 3
# #====================================================================================================================

# #====================================================================================================================
#                                   3. Logistic Regression
# #====================================================================================================================


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

# Assuming X_train, X_test, y_train, y_test are your training and testing data
# Assuming stratified_kfold is defined

# Define the parameter grid for Logistic Regression
param_grid_lr = {
    'C': [0.01, 0.1],  # Inverse of regularization strength
    'penalty': ['l2'],  # Regularization penalty
    'solver': ['liblinear', 'saga'],  # Algorithm to use in the optimization problem

}

# Initialize Logistic Regression model
model_lr = LogisticRegression()

# Create StratifiedKFold
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=5805)

# Create grid search
grid_search_lr = GridSearchCV(model_lr, param_grid_lr, cv=stratified_kfold, scoring='accuracy')
grid_search_lr.fit(X_train, y_train)

# Get the best estimator from grid search
lr_best = grid_search_lr.best_estimator_

# Display the best parameters
print("Best Parameters:")
print(grid_search_lr.best_params_)

# Evaluate the model with best hyperparameters
lr_acc_train = accuracy_score(y_train, lr_best.predict(X_train))
lr_acc_test = accuracy_score(y_test, lr_best.predict(X_test))
print(f'Train accuracy of Logistic Regression: {round(lr_acc_train, 4)}')
print(f'Test accuracy of Logistic Regression: {round(lr_acc_test, 4)}')

# Confusion Matrix
conf_matrix_lr = confusion_matrix(y_test, lr_best.predict(X_test))
tn, fp, fn, tp = conf_matrix_lr.ravel()

# Specificity
specificity_lr = tn / (tn + fp)

# Probability of the positive class
y_prob_lr = lr_best.predict_proba(X_test)[:, 1]

# ROC AUC
roc_auc_lr = roc_auc_score(y_test, y_prob_lr)

# Recall
recall_lr = recall_score(y_test, lr_best.predict(X_test))

# Precision
precision_lr = precision_score(y_test, lr_best.predict(X_test))

# F1 Score
f1_lr = f1_score(y_test, lr_best.predict(X_test))

# Display results using PrettyTable
pt = PrettyTable()
pt.field_names = ['Classifier', 'Accuracy', 'Confusion Matrix', 'Precision', 'Sensitivity or Recall', 'Specificity', 'F-score', 'AUC']
pt.add_row(['Logistic Regression', round(np.mean(lr_acc_test), 4), conf_matrix_lr.round(4),
            round(np.mean(precision_lr), 4), round(np.mean(recall_lr), 4),
            round(np.mean(specificity_lr), 4), round(np.mean(f1_lr), 4),
            round(np.mean(roc_auc_lr), 4)])
pt2.add_row(['Logistic Regression', round(np.mean(lr_acc_test), 4), conf_matrix_lr.round(4),
            round(np.mean(precision_lr), 4), round(np.mean(recall_lr), 4),
            round(np.mean(specificity_lr), 4), round(np.mean(f1_lr), 4),
            round(np.mean(roc_auc_lr), 4)])
print(pt)

# ROC Curve
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_prob_lr)
plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, linewidth=2, label="ROC curve of Logistic Regression (area = {:.2f})".format(roc_auc_lr))
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve ')
plt.legend(loc="lower right")
plt.show()


# #====================================================================================================================
#                                   Classifier 4
# #====================================================================================================================

# #====================================================================================================================
#                                   4. MLP
# #====================================================================================================================

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.model_selection import StratifiedKFold  # Import StratifiedKFold

# Assuming X_train, X_test, y_train, y_test are already defined

# Create the MLPClassifier
model_mlp = MLPClassifier(random_state=42)

# Use StratifiedKFold for better handling of imbalanced classes
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=5805)

# Define the parameter grid for grid search
param_grid = {
    'hidden_layer_sizes': [(5,5)],
    'max_iter': [3],
    'random_state': [5805]
}

# Create the GridSearchCV object
grid_search = GridSearchCV(model_mlp, param_grid, cv=stratified_kfold, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Print the best parameters
print("Best Parameters:")
print(grid_search.best_params_)

# Make predictions
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# Calculate performance metrics
mlp_acc_train = accuracy_score(y_train, y_train_pred)
mlp_acc_test = accuracy_score(y_test, y_test_pred)
conf_matrix_mlp = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = conf_matrix_mlp.ravel()
specificity_mlp = tn / (tn + fp)
precision_mlp = precision_score(y_test, y_test_pred)
recall_mlp = recall_score(y_test, y_test_pred)
f1_mlp = f1_score(y_test, y_test_pred)
roc_auc_mlp = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

# Add results to the PrettyTable
# Add results to the PrettyTable
pt = PrettyTable()
pt.field_names = ['Classifier', 'Accuracy', 'Confusion Matrix', 'Precision', 'Sensitivity or Recall', 'Specificity', 'F-score', 'AUC']
pt.add_row(['MLP (GridSearch)', round(mlp_acc_test, 4), conf_matrix_mlp, round(precision_mlp, 4),
            round(recall_mlp, 4), round(specificity_mlp, 4), round(f1_mlp, 4), round(roc_auc_mlp, 4)])
print(pt)
pt2.add_row(['MLP (GridSearch)', round(mlp_acc_test, 4), conf_matrix_mlp, round(precision_mlp, 4),
            round(recall_mlp, 4), round(specificity_mlp, 4), round(f1_mlp, 4), round(roc_auc_mlp, 4)])


# Plot ROC curve
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr_mlp, tpr_mlp, linewidth=2, label="ROC curve of MLP (area = {:.2f})".format(roc_auc_mlp))
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# #====================================================================================================================
#                                   Classifier 5
# #====================================================================================================================

# #====================================================================================================================
#                                   5. Decision Tree
# #====================================================================================================================



# #====================================================================================================================
#                                   5. Decision Tree Pre pruning
# #====================================================================================================================

# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
from prettytable import PrettyTable
import seaborn as sns
import matplotlib.pyplot as plt

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train the model
dt_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Display classification report
print("Classification Report:")
print(class_report)

# Evaluate the Decision Tree model and display results in a table
dt_table = PrettyTable()
dt_table.field_names = ["Classifier", "Accuracy", "Confusion Matrix", "Precision", "Recall", "F-score", "AUC"]

dt_table.add_row(["Decision Tree", accuracy, conf_matrix,
                  precision_score(y_test, y_pred),
                  recall_score(y_test, y_pred),
                  f1_score(y_test, y_pred),
                  roc_auc_score(y_test, y_pred)])

print("Model Evaluation:")
print(dt_table)

# Plotting feature importances
feature_names = housing.drop(columns=[target_column]).columns
feature_importances = dt_classifier.feature_importances_
feature_imp = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
feature_imp = feature_imp.sort_values(by='Importance', ascending=False)

print("\nImportant Features:")
print(feature_imp)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp)
plt.title('Feature Importances (Decision Tree Classifier)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Set a threshold for feature importance
threshold = 0.05

# Display features with importance greater than or equal to the threshold
more_important = feature_imp[feature_imp['Importance'] >= threshold]
print("\nFeatures with Importance >= 0.05 (considered more important):")
print(more_important)

# Display features with importance less than the threshold
low_importance_features = feature_imp[feature_imp['Importance'] < threshold]
print("\nFeatures with Importance < 0.05 (considered for removal):")
print(low_importance_features)


# #====================================================================================================================
#                                   5. Decision Tree Post Pruning
# #====================================================================================================================
# Specify the features to remove
features_to_remove = ['thermalChar', 'numberOfFloors', 'lastRefurbish']

# Drop the specified features from the 'housing' dataset
housing_dtPre = final_dataset.drop(columns=features_to_remove)






#============================================ code for decision tree






from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=5805)

from prettytable import PrettyTable
pt = PrettyTable()
pt.field_names = ['Classifier', 'Accuracy', 'Confusion Matrix', 'Precision', 'Sensitivity or Recall', 'Specificity', 'F-score', 'AUC']

dt_classifier = DecisionTreeClassifier()

X = housing_dtPre.drop(columns=['totalRent']).values  # Drop the 'totalRent' column for features
y = housing_dtPre['totalRent'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805)

# Define the parameter grid for GridSearchCV
param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [2, 3],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2'],
    'ccp_alpha': [0.0, 0.1, 0.2, 0.3]
}

# Instantiate GridSearchCV with the Decision Tree classifier and parameter grid
grid_search_dt = GridSearchCV(dt_classifier, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the grid search to the data
grid_search_dt.fit(X_train, y_train)

# Get the best hyperparameters from the grid search
best_params_dt = grid_search_dt.best_params_

# Print the best hyperparameters
print("Best Hyperparameters for Decision Tree:", best_params_dt)

# Get the best Decision Tree model
best_dt_model = grid_search_dt.best_estimator_

# Make predictions on the test set
y_pred_dt = best_dt_model.predict(X_test)

# Evaluate the performance of the best Decision Tree model
print("\nClassification Report for Decision Tree:")
print(classification_report(y_test, y_pred_dt))
print("Accuracy:", accuracy_score(y_test, y_pred_dt))



# Evaluate the model with best hyperparameters
dt_acc_train = accuracy_score(y_train, best_dt_model.predict(X_train))
dt_acc_test = accuracy_score(y_test, best_dt_model.predict(X_test))
print(f'Train accuracy of Decision Tree Post Pruning: {round(dt_acc_train, 2)}')
print(f'Test accuracy of Decision Tree Post Pruning: {round(dt_acc_test, 2)}')

# Confusion Matrix
conf_matrix_dt = confusion_matrix(y_test, best_dt_model.predict(X_test))
tn, fp, fn, tp = conf_matrix_dt.ravel()

# Specificity
specificity_dt = tn / (tn + fp)

# Probability of the positive class
y_prob_dt = best_dt_model.predict_proba(X_test)[:, 1]

# ROC AUC
roc_auc_dt = roc_auc_score(y_test, y_prob_dt)

# Recall
recall_dt = recall_score(y_test, best_dt_model.predict(X_test))

# Precision
precision_dt = precision_score(y_test, best_dt_model.predict(X_test))

# F1 Score
f1_dt = f1_score(y_test, best_dt_model.predict(X_test))

# Display results using PrettyTable
pt = PrettyTable()
pt.field_names = ['Classifier', 'Accuracy', 'Confusion Matrix', 'Precision', 'Sensitivity or Recall', 'Specificity', 'F-score', 'AUC']
pt.add_row(['Decision Tree Post Pruning', round(np.mean(dt_acc_test), 4), conf_matrix_dt.round(4),
            round(np.mean(precision_dt), 4), round(np.mean(recall_dt), 4),
            round(np.mean(specificity_dt), 4), round(np.mean(f1_dt), 4), round(np.mean(roc_auc_dt), 4)])
print(pt)
pt2.add_row(['Decision Tree Post Pruning', round(np.mean(dt_acc_test), 4), conf_matrix_dt.round(4),
            round(np.mean(precision_dt), 4), round(np.mean(recall_dt), 4),
            round(np.mean(specificity_dt), 4), round(np.mean(f1_dt), 4), round(np.mean(roc_auc_dt), 4)])


# ROC Curve
fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_test, y_prob_dt)
plt.figure(figsize=(8, 6))
plt.plot(fpr_dt, tpr_dt, linewidth=2, label="ROC curve of Decision Tree Post Pruning (area = {:.2f})".format(roc_auc_dt))
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve ')
plt.legend(loc="lower right")
plt.show()


# #====================================================================================================================
#                                   Classifier 6
# #====================================================================================================================

# #====================================================================================================================
#                                   6. Random Forest
# #====================================================================================================================

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier


stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=5805)

# Assume X and y are your feature matrix and target variable
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [2, 3],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2'],
    'ccp_alpha': [0.0, 0.1, 0.2, 0.3]
}

grid_search_rf = GridSearchCV(rf_classifier, rf_param_grid, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)
best_rf_model = grid_search_rf.best_estimator_

# Bagging Classifier
bagging_classifier = BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=42),
                                       random_state=42)
bagging_param_grid = {
    'n_estimators': [10, 50, 100],
    'max_samples': [0.5, 0.7, 1.0],
}

grid_search_bagging = GridSearchCV(bagging_classifier, bagging_param_grid, cv=5, scoring='accuracy')
grid_search_bagging.fit(X_train, y_train)
best_bagging_model = grid_search_bagging.best_estimator_

# AdaBoost Classifier
adaboost_classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=42),
                                         random_state=42)
adaboost_param_grid = {
    'n_estimators': [10, 50, 100],
    'learning_rate': [0.01, 0.1, 1.0],
}

grid_search_adaboost = GridSearchCV(adaboost_classifier, adaboost_param_grid, cv=5, scoring='accuracy')
grid_search_adaboost.fit(X_train, y_train)
best_adaboost_model = grid_search_adaboost.best_estimator_

# Stacking Classifier
stacking_classifier = StackingClassifier(
    estimators=[('rf', best_rf_model), ('bagging', best_bagging_model), ('adaboost', best_adaboost_model)],
    final_estimator=RandomForestClassifier(random_state=42)
)

stacking_classifier.fit(X_train, y_train)

# Make predictions
stacking_predictions = stacking_classifier.predict(X_test)

# Evaluate accuracy
stacking_accuracy = accuracy_score(y_test, stacking_predictions)
print(f"Stacking Accuracy: {stacking_accuracy}")

# Evaluate performance
stacking_conf_matrix = confusion_matrix(y_test, stacking_predictions)
stacking_precision_micro = precision_score(y_test, stacking_predictions, average='micro')
stacking_precision_macro = precision_score(y_test, stacking_predictions, average='macro')
stacking_recall_micro = recall_score(y_test, stacking_predictions, average='micro')
stacking_recall_macro = recall_score(y_test, stacking_predictions, average='macro')

# Display results
# Get the best estimator from grid search for RandomForest
best_rf_model = grid_search_rf.best_estimator_

# Print the best parameters for RandomForest
print("Best Parameters for RandomForest:")
print(best_rf_model.get_params())

# Evaluate the model with best hyperparameters for RandomForest
rf_acc_train = accuracy_score(y_train, best_rf_model.predict(X_train))
rf_acc_test = accuracy_score(y_test, best_rf_model.predict(X_test))
print(f'Train accuracy of RandomForest: {round(rf_acc_train, 2)}')
print(f'Test accuracy of RandomForest: {round(rf_acc_test, 2)}')

# Confusion Matrix
conf_matrix_rf = confusion_matrix(y_test, best_rf_model.predict(X_test))
tn, fp, fn, tp = conf_matrix_rf.ravel()

# Specificity
specificity_rf = tn / (tn + fp)

# Probability of the positive class
y_prob_rf = best_rf_model.predict_proba(X_test)[:, 1]

# ROC AUC
roc_auc_rf = roc_auc_score(y_test, y_prob_rf)

# Recall
recall_rf = recall_score(y_test, best_rf_model.predict(X_test))

# Precision
precision_rf = precision_score(y_test, best_rf_model.predict(X_test))

# F1 Score
f1_rf = f1_score(y_test, best_rf_model.predict(X_test))

# Display results using PrettyTable
pt_rf = PrettyTable()
pt_rf.field_names = ['Classifier', 'Accuracy', 'Confusion Matrix', 'Precision', 'Sensitivity or Recall', 'Specificity', 'F-score', 'AUC']
pt_rf.add_row(['RandomForest', round(np.mean(rf_acc_test), 3), conf_matrix_rf.round(3),
               round(np.mean(precision_rf), 3), round(np.mean(recall_rf), 3),
               round(np.mean(specificity_rf), 3), round(np.mean(f1_rf), 3), round(np.mean(roc_auc_rf), 3)])
print(pt_rf)
pt2.add_row(['RandomForest', round(np.mean(rf_acc_test), 3), conf_matrix_rf.round(3),
               round(np.mean(precision_rf), 3), round(np.mean(recall_rf), 3),
               round(np.mean(specificity_rf), 3), round(np.mean(f1_rf), 3), round(np.mean(roc_auc_rf), 3)])


# ROC Curve
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_prob_rf)
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, linewidth=2, label="ROC curve of RandomForest (area = {:.2f})".format(roc_auc_rf))
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve ')
plt.legend(loc="lower right")
plt.show()


# #====================================================================================================================
#                                   6. SVM - Support Vector Machine
# #====================================================================================================================
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

# Use StratifiedKFold for better handling of imbalanced classes
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=5805)

parameters = {
    'kernel': ['linear', 'rbf', 'poly'],  # Kernel types to try
}
# Perform grid search with cross-validation
svm_classifier = SVC(probability=True)
grid_search = GridSearchCV(svm_classifier, parameters, cv=stratified_kfold, n_jobs=-1)
grid_search.fit(X, y)
print("Best Hyperparameters:", grid_search.best_params_)
svm_lr = grid_search.best_estimator_
svm_lr.fit(X_train, y_train)
y_train_pred = svm_lr.predict(X_train)
y_test_pred = svm_lr.predict(X_test)
svm_lr_acc = accuracy_score(y_test, y_test_pred)
print(f'Train accuracy of SVM with best hyperparameters {round(np.mean(svm_lr_acc),2)}')
print(f'Test accuracy of SVM with best hyperparameters {round(accuracy_score(y_test, y_test_pred),2)}')
conf_matrix_svm_lr = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp =    conf_matrix_svm_lr.ravel()
specificity_svm_lr= tn/(tn+fp)
y_prob_svm_lr =     svm_lr.predict_proba(X_test)[:, 1] # Probability of the positive class
roc_auc_svm_lr =    roc_auc_score(y_test, y_prob_svm_lr)
recall_svm_lr =     recall_score(y_test, y_test_pred)
precision_svm_lr =  precision_score(y_test, y_test_pred)
f1_svm_lr =         f1_score(y_test, y_test_pred)



pt = PrettyTable()
pt.field_names = ['Classifier', 'Accuracy', 'Confusion Matrix', 'Precision', 'Sensitivity or Recall', 'Specificity', 'F-score', 'AUC']

pt.add_row(['SVM', round(np.mean(svm_lr_acc),4), conf_matrix_svm_lr.round(3), round(np.mean(precision_svm_lr),3), round(np.mean(recall_svm_lr),3), round(np.mean(specificity_svm_lr),3), round(np.mean(f1_svm_lr),3), round(np.mean(roc_auc_svm_lr),3)])

pt2.add_row(['SVM', round(np.mean(svm_lr_acc),4), conf_matrix_svm_lr.round(3), round(np.mean(precision_svm_lr),3), round(np.mean(recall_svm_lr),3), round(np.mean(specificity_svm_lr),3), round(np.mean(f1_svm_lr),3), round(np.mean(roc_auc_svm_lr),3)])

fpr_svm_lr, tpr_svm_lr, thresholds_svm_lr = roc_curve(y_test, y_prob_svm_lr)
plt.figure(figsize=(8, 6))
plt.plot(fpr_svm_lr, tpr_svm_lr, linewidth=2, label="ROC curve of SVM with Linear Kernel(area = {:.2f})".format(roc_auc_svm_lr))
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve ')
plt.legend(loc="lower right")
plt.show()
print(pt)

print(pt2)

#=====================================================================================================================================================
#                               ************   PHASE 4 ************
#===================================================================================================================================================

# #====================================================================================================================
#                                           K means Clustering
# #====================================================================================================================


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules

# Assuming 'df' is your input dataframe
ais_data_encoded1 = copy_df.sample(n=50000, random_state=5805)
# K-means clustering
def kmeans_clustering(dataframe, k_values):
    # Silhouette analysis for k selection
    silhouette_scores = []
    within_cluster_var = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=5808)
        kmeans.fit(ais_data_encoded1)
        labels = kmeans.labels_

        # Silhouette score
        silhouette_scores.append(silhouette_score(ais_data_encoded1, labels))

        # Within-cluster variation
        within_cluster_var.append(kmeans.inertia_)

    # Plotting Silhouette score for different k values
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_values, silhouette_scores, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis for KMeans')

    # Plotting within-cluster variation
    plt.subplot(1, 2, 2)
    plt.plot(k_values, within_cluster_var, 'ro-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within-cluster Variation')
    plt.title('Within-cluster Variation Plot')

    plt.tight_layout()
    plt.show()

    return silhouette_scores, within_cluster_var


#  usage
k_values = range(2, 11)  # Range of k values to test
silhouette_scores, within_cluster_var = kmeans_clustering(copy_df, k_values)


# # #====================================================================================================================
# #                                           Apriori Algorithm
# # #====================================================================================================================
# Identify and display all categorical columns
categorical_columns = dataset_apriori.select_dtypes(include='object').columns

# Display the DataFrame with only categorical columns
categorical_df = dataset_apriori[categorical_columns]

print("\nDataFrame with All Categorical Columns:")
print(categorical_df)

# Assuming df_apriori is your DataFrame after classifying columns

# Drop unwanted columns
columns_to_keep = ['heatingType', 'typeOfFlat', 'interiorQual', 'condition','petsAllowed','regio2']
df_apriori_filtered = dataset_apriori[columns_to_keep]

# Displaying the filtered DataFrame
print(df_apriori_filtered)

# Assuming df_filtered_apriori is your DataFrame
df_apriori_filtered = df_apriori_filtered.dropna()

# Displaying the filtered DataFrame
print(df_apriori_filtered)
print(df_apriori_filtered.isna().sum())

from mlxtend.frequent_patterns import apriori, association_rules

# Apply aprori algorithm to the above dataframe
df_apriori_filtered = pd.get_dummies(df_apriori_filtered,drop_first=True)
frequent_itemsets = apriori(df_apriori_filtered, min_support=0.1, use_colnames=True, verbose=1)
# Generate association rules
# metric can be set to 'confidence', 'lift', 'leverage', and 'conviction'
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.2)
# Sort rules by confidence in descending order
rules = rules.sort_values(['confidence'], ascending=False)
# Print top 10 rules
print(rules.head(10))