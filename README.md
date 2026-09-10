# 📈 AAPL Stock Data Analysis

## 📌 Project Overview

This project performs a **stock market data analysis of Apple Inc. (AAPL)** using Python.

The project uses historical AAPL stock data stored in a CSV file and applies **Pandas, NumPy, Matplotlib, and Seaborn** for data processing, analysis, and visualization.

The main goal of this project is to understand stock price trends, daily returns, yearly performance, moving averages, and investment profit or loss.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**

---

## 📂 Dataset

The project uses an `AAPL.csv` file containing historical Apple stock data.

The dataset includes columns such as:

| Column      | Description                  |
| ----------- | ---------------------------- |
| `Date`      | Trading date                 |
| `Open`      | Opening price                |
| `High`      | Highest price during the day |
| `Low`       | Lowest price during the day  |
| `Close`     | Closing price                |
| `Adj Close` | Adjusted closing price       |
| `Volume`    | Number of shares traded      |

---

# 🔄 Project Workflow

The project follows these main steps:

```text
Load Dataset
     ↓
Explore Dataset
     ↓
Check Missing Values
     ↓
Convert Date
     ↓
Sort Data
     ↓
Calculate Daily Return
     ↓
Create Next-Day Target
     ↓
Remove Missing Values
     ↓
Extract Year
     ↓
Exploratory Data Analysis
     ↓
Year-wise Analysis
     ↓
Calculate Yearly Return
     ↓
Calculate Moving Averages
     ↓
Investment Profit/Loss Analysis
```

---

# 1. Import Libraries

The required libraries are imported first.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

### Pandas

Pandas is used to load, clean, and analyze the stock data.

### NumPy

NumPy is used for numerical calculations.

### Matplotlib

Matplotlib is used to create charts and graphs.

### Seaborn

Seaborn is imported for data visualization and can be useful for creating statistical graphs.

---

# 2. Load the Dataset

The AAPL CSV file is loaded using Pandas.

```python
df = pd.read_csv("AAPL.csv")
```

The dataset is stored in a variable called `df`.

`df` stands for **DataFrame**.

A DataFrame is a table containing rows and columns of data.

---

# 3. View the First Rows

```python
df.head()
```

The `head()` function displays the first five rows of the dataset.

This helps us understand what the data looks like.

---

# 4. Check Dataset Shape

```python
df.shape
```

The `shape` function tells us the number of:

* Rows
* Columns

The result is shown as:

```text
(rows, columns)
```

---

# 5. Check Dataset Information

```python
df.info()
```

This provides information about:

* Column names
* Number of values
* Data types
* Missing values

This is useful during the preprocessing stage.

---

# 6. Statistical Summary

```python
df.describe()
```

The `describe()` function provides statistical information about numerical columns.

It includes:

* Count
* Mean
* Standard deviation
* Minimum
* 25th percentile
* Median
* 75th percentile
* Maximum

This gives us an initial understanding of the stock data.

---

# 7. Check Missing Values

```python
df.isnull().sum()
```

This checks how many missing values exist in each column.

Missing values are important because they can cause problems during calculations and analysis.

---

# 8. Convert Date Column

First, the date column is examined:

```python
df['Date']
```

Then it is converted into Pandas datetime format:

```python
df['Date'] = pd.to_datetime(df['Date'])
```

This allows us to perform date-based operations.

For example, we can later extract:

```text
Year
Month
Day
```

from the date.

---

# 9. Sort the Data by Date

```python
df = df.sort_values('Date')
```

This sorts the dataset according to the `Date` column.

Sorting is important because stock data should be in chronological order when calculating things such as:

* Daily returns
* Moving averages
* Next-day prices

---

# 10. Calculate Daily Return

The daily return is calculated using:

```python
df['Daily_Return'] = df['Adj Close'].pct_change()
```

`pct_change()` calculates the percentage change between the current value and the previous value.

The basic idea is:

$$
Daily\ Return =
\frac{Today's\ Price - Yesterday's\ Price}
{Yesterday's\ Price}
$$

For example, if yesterday's adjusted closing price was `$100` and today's was `$105`:

$$
\frac{105-100}{100}=0.05
$$

So the daily return is:

```text
0.05 = 5%
```

---

# 11. Create Next-Day Target

```python
df['Target_Next_Day_Close'] = df['Adj Close'].shift(-1)
```

The `shift(-1)` function moves the next day's adjusted closing price into the current row.

For example:

| Date  | Adj Close | Target Next Day Close |
| ----- | --------: | --------------------: |
| Day 1 |       100 |                   105 |
| Day 2 |       105 |                   103 |
| Day 3 |       103 |                   110 |

This creates a target that could later be used for a machine learning prediction project.

---

# 12. Remove Missing Values

After calculating `pct_change()` and `shift(-1)`, some rows contain missing values.

They are removed using:

```python
df = df.dropna()
```

This removes rows containing missing values.

---

# 13. Extract the Year

A new `Year` column is created:

```python
df['Year'] = df['Date'].dt.year
```

The `.dt.year` operation extracts the year from the date.

For example:

```text
2020-05-15 → 2020
2021-08-20 → 2021
2022-11-10 → 2022
```

This allows us to perform year-wise analysis.

---

# 📊 Exploratory Data Analysis

## 14. Adjusted Closing Price

The first major chart shows the historical adjusted closing price.

```python
plt.plot(df['Date'], df['Adj Close'])
```

### Purpose

This chart helps us understand how Apple's adjusted stock price changed over time.

It can show:

* Long-term growth
* Price decreases
* Major trends
* Periods of high volatility

---

## 15. Trading Volume

The next chart displays trading volume.

```python
plt.plot(df['Date'], df['Volume'])
```

Trading volume represents the number of shares traded during a particular day.

High volume can indicate increased trading activity.

---

## 16. Distribution of Daily Returns

A histogram is used to visualize daily returns.

```python
plt.hist(df['Daily_Return'].dropna(), bins=50)
```

The histogram shows how frequently different daily return values occurred.

`bins=50` divides the range of daily returns into 50 groups.

This helps us understand the distribution of daily stock returns.

---

## 17. Daily High and Low Prices

The project also compares the daily high and low prices.

```python
plt.plot(df['Date'], df['High'], label='High')
plt.plot(df['Date'], df['Low'], label='Low')
```

This allows us to see the range in which the stock traded each day.

The difference between the high and low price can give an idea of daily price movement.

---

# 📅 Year-wise Analysis

## 18. Calculate Yearly Statistics

The data is grouped by year:

```python
yearly = df.groupby('Year').agg(
    Start_Price=('Adj Close', 'first'),
    End_Price=('Adj Close', 'last'),
    Highest_Price=('Adj Close', 'max'),
    Lowest_Price=('Adj Close', 'min'),
    Average_Price=('Adj Close', 'mean'),
    Total_Volume=('Volume', 'sum')
).reset_index()
```

This creates a new table containing yearly information.

### Calculated values

**Start_Price**

The first adjusted closing price recorded for the year.

**End_Price**

The last adjusted closing price recorded for the year.

**Highest_Price**

The highest adjusted closing price during the year.

**Lowest_Price**

The lowest adjusted closing price during the year.

**Average_Price**

The average adjusted closing price for the year.

**Total_Volume**

The total trading volume for the year.

---

# 19. Yearly Return

Yearly return is calculated using:

```python
yearly['Yearly_Return'] = (
    (yearly['End_Price'] - yearly['Start_Price'])
    / yearly['Start_Price']
) * 100
```

The formula is:

$$
Yearly\ Return =
\frac{End\ Price-Start\ Price}
{Start\ Price}
\times100
$$

For example, if:

```text
Start Price = $100
End Price = $120
```

then:

$$
\frac{120-100}{100}\times100=20\%
$$

The yearly return is therefore **20%**.

---

# 📈 Year-wise Closing Price

The project creates a line chart showing the year-end adjusted closing price.

```python
plt.plot(
    yearly['Year'],
    yearly['End_Price'],
    marker='o'
)
```

This helps compare Apple's year-end stock prices across different years.

---

# 📊 Year-wise Return

A bar chart is used to display yearly returns.

```python
plt.bar(
    yearly['Year'],
    yearly['Yearly_Return']
)
```

A horizontal zero line is added:

```python
plt.axhline(0, linewidth=1)
```

This makes it easier to identify:

* Positive returns → profit/growth
* Negative returns → loss/decline

---

# 📉 Moving Averages

Two moving averages are calculated:

```python
df['MA_20'] = df['Adj Close'].rolling(window=20).mean()

df['MA_50'] = df['Adj Close'].rolling(window=50).mean()
```

## 20-Day Moving Average

The 20-day moving average calculates the average adjusted closing price over the most recent 20 trading days.

## 50-Day Moving Average

The 50-day moving average calculates the average adjusted closing price over the most recent 50 trading days.

Moving averages help smooth out daily price fluctuations and make the overall trend easier to see.

---

# 📈 Price with Moving Averages

The project plots:

* Adjusted Close
* 20-Day Moving Average
* 50-Day Moving Average

```python
plt.plot(df['Date'], df['Adj Close'])
plt.plot(df['Date'], df['MA_20'])
plt.plot(df['Date'], df['MA_50'])
```

This allows us to compare the actual stock price with its moving averages.

---

# 💰 Investment Profit/Loss Analysis

The project allows the user to enter:

```text
Investment amount
Starting year
Ending year
```

For example:

```text
Enter your investment amount: 1000
Enter starting year: 2010
Enter ending year: 2015
```

The selected years are filtered:

```python
year_data = df[
    (df['Year'] >= start_year) &
    (df['Year'] <= end_year)
]
```

---

# 💵 Calculate Number of Shares

The first adjusted closing price is obtained:

```python
first_price = year_data['Adj Close'].iloc[0]
```

The last adjusted closing price is obtained:

```python
last_price = year_data['Adj Close'].iloc[-1]
```

The number of shares that could be purchased is:

```python
shares = initial_investment / first_price
```

For example, if:

```text
Investment = $1,000
First Price = $100
```

then:

$$
Shares = \frac{1000}{100}=10
$$

So the investment would buy **10 shares**.

---

# 💰 Calculate Final Investment Value

The final investment value is calculated using:

```python
final_value = shares * last_price
```

For example, if you own:

```text
10 shares
```

and the ending price is:

```text
$150
```

then:

$$
10\times150=\$1500
$$

The final value would be **$1,500**.

---

# 📈 Calculate Profit or Loss

Profit or loss is calculated using:

```python
profit_loss = final_value - initial_investment
```

If the final value is higher than the initial investment, there is a **profit**.

If the final value is lower, there is a **loss**.

---

# 📊 Calculate Profit/Loss Percentage

The percentage return is calculated using:

```python
profit_loss_percent = (
    profit_loss / initial_investment
) * 100
```

The formula is:

$$
Profit/Loss\% =
\frac{Profit/Loss}{Initial\ Investment}
\times100
$$

---

# 🎯 Project Objectives

The main objectives of this project are to:

* Understand stock market data
* Perform data preprocessing
* Analyze historical AAPL prices
* Calculate daily returns
* Analyze yearly performance
* Calculate moving averages
* Visualize stock trends
* Calculate investment growth
* Calculate profit or loss
* Understand how historical stock data can be analyzed using Python

---

# 📚 Key Concepts Learned

Through this project, I learned:

* How to load CSV data using Pandas
* How to inspect a dataset
* How to check missing values
* How to convert dates using `pd.to_datetime()`
* How to sort data by date
* How `pct_change()` works
* How `shift(-1)` works
* How to extract the year from a date
* How to use `groupby()` and `agg()`
* How to calculate yearly returns
* How to create line charts
* How to create bar charts
* How to create histograms
* How moving averages work
* How to calculate investment profit/loss

---

# ⚠️ Important Note

This project is for **educational and data-analysis purposes**.

The profit/loss calculation assumes that:

* Shares are purchased at the first available adjusted closing price in the selected period.
* Shares are sold at the last available adjusted closing price.
* Fractional shares are allowed.
* No brokerage fees, taxes, or other transaction costs are included.
* Dividends are not separately modeled because the analysis uses `Adj Close`.

Historical stock performance does **not** guarantee future results.

---

# 🚀 Conclusion

This project demonstrates a complete beginner-level stock data analysis workflow using AAPL historical data.

The analysis starts with loading and cleaning the data and continues through exploratory data analysis, daily returns, yearly statistics, moving averages, and investment profit/loss calculations.

The project provides practical experience with **Pandas, NumPy, and Matplotlib** while building a foundation for more advanced financial data analysis and machine learning projects.
