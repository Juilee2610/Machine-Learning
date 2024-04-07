**Report Summary: Exploratory Data Analysis and Manipulation on Stock Prices Dataset**

This report outlines the procedures and analyses conducted on the 'stock prices.csv' dataset, focusing on data cleaning, aggregation, visualization, and feature engineering. The dataset is sourced from the course GitHub repository and contains information on stock prices for various companies.

**1. Data Cleaning:**
- Loaded the dataset using Pandas and identified missing entries.
- Imputed missing values with the mean and verified the cleanliness of the dataset.
- Displayed confirmation of missing values being fixed.

**2. Exploratory Analysis:**
- Identified the number of unique companies and categorized predictors into quantitative and qualitative.
- Created a new DataFrame comprising only 'GOOGLE' and 'APPLE' stocks and plotted their closing stock values.

**3. Aggregation:**
- Aggregated the dataset by 'symbol' using summation operation and compared the number of objects between the cleaned and aggregated datasets.
- Displayed the first 5 rows of the aggregated dataset.

**4. Feature Engineering:**
- Sliced the dataset to include 'symbol', 'close', and 'volume', then aggregated by 'symbol' using mean and variance operations.
- Identified the company with the maximum variance in closing costs and displayed a message.

**5. Temporal Analysis:**
- Filtered the dataset to include only Google stock closing costs after 2015-01-01.
- Displayed the first 5 rows of the filtered dataset and plotted closing costs against time with a rolling window of 30 days.

**6. Discretization:**
- Discretized the 'close' feature into 5 equal-width bins and visualized the count plot of the new categorical feature.
- Displayed the created dataset and plotted a histogram of the 'close' feature with 5 bins.

**7. Covariance Analysis:**
- Estimated the covariance matrix without using built-in Python functions, followed by estimation using the cov() function.
- Compared the results and discussed observations about the covariance matrix, emphasizing the linear relationships between features.

**Conclusion:**
This report demonstrates a comprehensive exploration of the 'stock prices.csv' dataset, covering data cleaning, aggregation, visualization, feature engineering, and covariance analysis. The analyses provide valuable insights into stock market dynamics and serve as a foundation for further quantitative analysis and modeling.


