import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import yfinance as yf
from tqdm import tqdm

class NiftyCandlestickChart:
    def __init__(self, df):
        self.df = df.sort_values('Datetime')
        self.time_dict = {i: timestamp for i, timestamp in enumerate(self.df['Datetime'].sort_values())}
        self.reverse_time_dict = {v: k for k, v in self.time_dict.items()}


    def plot_even_candlestick(self):
        fig, ax = plt.subplots(figsize=(15, 10))
        for index, row in self.df.iterrows():
            color = 'g' if row['Close'] >= row['Open'] else 'r'
            x_value = self.reverse_time_dict[row['Datetime']]
            ax.plot([x_value, x_value], [row['Low'], row['High']], color=color, linewidth=2)
            ax.plot([x_value, x_value], [row['Open'], row['Close']], color=color, linewidth=8, solid_capstyle='butt')

        ax.set_xticks(list(self.reverse_time_dict.values())[::len(self.reverse_time_dict)//10])
        ax.set_xticklabels([str(self.time_dict[k]) for k in ax.get_xticks()], rotation=45)

        ax.set_xlabel('Datetime')
        ax.set_ylabel('Price')
        ax.set_title('Evenly Spaced Nifty Candlestick Chart')
        plt.grid(True)
        plt.show()

    def plot_channel_for_range(self, start, end):
        fig, ax = plt.subplots(figsize=(10, 7))
        for index, row in tqdm(self.df.iterrows()):
            color = 'g' if row['Close'] >= row['Open'] else 'r'
            x_value = self.reverse_time_dict[row['Datetime']]
            ax.plot([x_value, x_value], [row['Low'], row['High']], color=color, linewidth=2)
            ax.plot([x_value, x_value], [row['Open'], row['Close']], color=color, linewidth=6, solid_capstyle='butt')

        subset_df = self.df.iloc[-start:-end]
        X = np.array(list(self.reverse_time_dict.values())[-start:-end]).reshape(-1, 1)
        y_high = subset_df['High'].values
        y_low = subset_df['Low'].values

        reg_high = LinearRegression().fit(X, y_high)
        upper_line = reg_high.predict(X)
        reg_low = LinearRegression().fit(X, y_low)
        lower_line = reg_low.predict(X)
        reg_high = LinearRegression().fit(X, y_high)
        upper_line = reg_high.predict(X)
        r2_high = reg_high.score(X, y_high)

        # Linear regression for lower channel line (Lows)
        reg_low = LinearRegression().fit(X, y_low)
        lower_line = reg_low.predict(X)
        r2_low = reg_low.score(X, y_low)

        # Print R-squared values
        st.write(f"R-squared value for the upper channel line: {r2_high:.4f}")
        st.write(f"R-squared value for the lower channel line: {r2_low:.4f}")
        ax.plot(X, upper_line, color='black', linewidth=2)
        ax.plot(X, lower_line, color='black', linewidth=2)

        ax.set_xticks(list(self.reverse_time_dict.values())[::len(self.reverse_time_dict)//10])
        ax.set_xticklabels([str(self.time_dict[k]) for k in ax.get_xticks()], rotation=45)

        ax.set_xlabel('Datetime')
        ax.set_ylabel('Price')
        ax.set_title(f'Nifty Candlestick Chart with Single Channel (Range: {start} to {end})')
        plt.grid(True)
        st.pyplot(plt)


st.title('Nifty Candlestick Chart with Channel Lines')

# User input for the ticker symbol, start date, and end date
ticker = st.text_input("Enter the ticker symbol (e.g., ^NSEI):", "^NSEI")
start_date = st.date_input("Start date:", min_value=pd.to_datetime("2022-01-01"))
end_date = st.date_input("End date:", min_value=start_date)

# Download the data
st.write(f"Downloading data for {ticker} from {start_date} to {end_date}...")
interval = st.text_input('Interval:',value='5min')
df = pd.DataFrame(yf.download(ticker, interval=interval, start=start_date, end=end_date))
df = df.reset_index()

if 'Date' in df.columns:
    df['Datetime'] = df['Date']

# Display the downloaded data
st.write("Here is the downloaded data:")
st.dataframe(df)

# User input for the channel durations (start and end)
with st.sidebar:
  start_duration = st.slider("Start duration for channel:", min_value=5, max_value=len(df), value=50)
  end_duration = st.slider("End duration for channel:", min_value=1, max_value=start_duration, value=30)

# Generate the candlestick chart with channel lines
st.write(f"Generating the candlestick chart with channel lines for {ticker}...")
chart = NiftyCandlestickChart(df)
#st.write('Created class')
chart.plot_channel_for_range(start=start_duration, end=end_duration)
