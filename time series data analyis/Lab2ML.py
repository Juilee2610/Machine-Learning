import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

#-------------------------------------------------------------------------------------------------------------------------------------------
#                                                   Question 1
#------------------------------------------------------------------------------------------------------------------------------------------
df = pd.read_csv('https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/stock%20prices.csv')

print(df.isna().sum())
df['open'].fillna(df['open'].mean(), inplace=True)
df['high'].fillna(df['high'].mean(), inplace=True)
df['low'].fillna(df['low'].mean(), inplace=True)

print("After filling the null values the total number of null values are:")
print(df.isna().sum())
print(df.isna().sum().sum())
if df.isna().sum().sum()==0:
    print("Missing value are fixed")
else:
    print("Missing values are not fixed")

#-------------------------------------------------------------------------------------------------------------------------------------------
#                                                   Question 2
#------------------------------------------------------------------------------------------------------------------------------------------
print(df.head())
unique_companies = df['symbol'].unique()

num_unique_companies = len(unique_companies)
print(df['symbol'].unique())
print("Number of Unique companies ")
print(num_unique_companies)


newdf = ['AAPL','GOOGL']
filtered_df = df[df['symbol'].isin(newdf)]

plt.figure(figsize=(12, 8))

for company in newdf:
    company_data = filtered_df[filtered_df['symbol'] == company]
    plt.plot(company_data['date'], company_data['close'], label=company)

plt.xlabel('Date')
plt.ylabel('USD ($)')
plt.title('APPLE AND GOOGLE closing value comparison')
 # Rotate x-axis labels for better visibility

plt.gca().xaxis.set_major_locator(plt.MaxNLocator(6))
plt.legend()
plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------
#                                                   Question 3
#------------------------------------------------------------------------------------------------------------------------------------------


aggregated_df = df.groupby('symbol').sum()


print("First 5 rows of the aggregated dataset:")
print(aggregated_df.head())

original = len(df)
aggregated = len(aggregated_df)

print(f"Number of objects in the original dataset: {original}")
print(f"Number of objects in the aggregated dataset: {aggregated}")

#-------------------------------------------------------------------------------------------------------------------------------------------
#                                                   Question 4
#------------------------------------------------------------------------------------------------------------------------------------------
sliced_df = df[['symbol', 'close', 'volume']]
aggregated_df = sliced_df.groupby('symbol').agg({'close': ['mean', 'var'], 'volume': ['mean', 'var']}).reset_index()

# Find the company with the maximum variance in the closing cost
max_variance_company = aggregated_df.loc[np.argmax(aggregated_df[('close', 'var')])]['symbol']
max_variance_value = np.max(aggregated_df[('close', 'var')])

# Display the company with the maximum variance on the console
print(f"The company with the maximum variance in closing cost is '{max_variance_company}' with a variance of {max_variance_value:.3f}.")
#-------------------------------------------------------------------------------------------------------------------------------------------
#                                                   Question 5
#------------------------------------------------------------------------------------------------------------------------------------------
df['date'] = pd.to_datetime(df['date'])
google_df = df[(df['symbol']=='GOOGL') & (df['date']>'2015-01-01')]
print("First 5 rows of the Google stock closing cost after January 1, 2015:")
print(google_df[['symbol','close']].head())


#-------------------------------------------------------------------------------------------------------------------------------------------
#                                                   Question 6
#------------------------------------------------------------------------------------------------------------------------------------------

rolling_mean = google_df['close'].rolling(window=30).mean()

# Create a figure and axis for the plot
plt.figure(figsize=(12, 6))
# Plot the closing cost
plt.plot(google_df['date'], google_df['close'], label='Closing Cost', color='blue')

# Plot the rolling mean
plt.plot(google_df['date'], rolling_mean, label='30-Day Rolling Mean', color='orange')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
# Customize the plot
plt.xlabel('Date')
plt.ylabel('USD($)')
plt.title('Google Stock Closing stock price after Jan 2015 versus rolling window')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

# Calculate the number of observations missed due to rolling window
missed_observations = len(google_df) - len(rolling_mean.dropna())
print(f"Number of observations missed due to rolling window: {missed_observations}")


#-------------------------------------------------------------------------------------------------------------------------------------------
#                                                   Question 7
#------------------------------------------------------------------------------------------------------------------------------------------
# Define bin labels
bin_labels = ['very low', 'low', 'normal', 'high', 'very high']

# Discretize the 'ClosingPrice' into 5 equal-width bins and create the 'price_category' column
google_df['price_category'] = pd.cut(google_df['close'], bins=5, labels=bin_labels)

# Plot a count plot of the 'price_category' feature
plt.figure(figsize=(8, 6))
sns.countplot(data=google_df, x='price_category', order=bin_labels)
plt.xlabel('Price Category')
plt.ylabel('Count')
plt.title('Count Plot of Price Category')
plt.show()

print(google_df.to_string(index=False))

#-------------------------------------------------------------------------------------------------------------------------------------------
#                                                   Question 8
#------------------------------------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.hist(google_df['close'], bins=5)
plt.xlabel('Closing Price')
plt.ylabel('Frequency')
plt.title('Histogram of Closing Price')
plt.grid(True)
plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------
#                                                   Question 9
#------------------------------------------------------------------------------------------------------------------------------------------
google_df['price_category'] = pd.qcut(google_df['close'], q=5, labels=['very low', 'low', 'normal', 'high', 'very high'])
plt.figure(figsize=(8,6))
sns.countplot(data = google_df, x='price_category', order=['very low', 'low', 'normal', 'high', 'very high'])
plt.xlabel('Price Category')
plt.ylabel('Count')
plt.title('Count Plot of Price Category')
plt.show()

print(google_df.to_string(index=False))

#-------------------------------------------------------------------------------------------------------------------------------------------
#                                                   Question 10
#------------------------------------------------------------------------------------------------------------------------------------------

numerical_columns = ['open', 'high', 'low', 'close', 'volume']
google_num = google_df[numerical_columns]
mean_vector = np.mean(google_num, axis=0)
centered_data = google_num - mean_vector
covariance_matrix = np.dot(centered_data.T, centered_data) / (len(google_num) - 1)
covariance_matrix_df = pd.DataFrame(covariance_matrix, columns=numerical_columns, index=numerical_columns)
print("Covariance Matrix for Numerical Variables:")
print(covariance_matrix_df.round(3))

#-------------------------------------------------------------------------------------------------------------------------------------------
#                                                   Question 11
#------------------------------------------------------------------------------------------------------------------------------------------
cov_matrix = google_num.cov()
print("Covariance Matrix (Using built-in cov()):")
print(cov_matrix)