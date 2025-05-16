# coca_cola_stock_project.py — Full Detailed Version

# ========== STEP 1: Import Libraries ==========
import yfinance as yf
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import openpyxl
import logging

# ========== STEP 2: Set Up Logging ==========
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ========== STEP 3: Fetch Live and Historical Data ==========
def fetch_data(ticker='KO', period='5y', interval='1d'):
    logging.info(f"Fetching data for {ticker} from Yahoo Finance...")
    data = yf.download(ticker, period=period, interval=interval)
    data.reset_index(inplace=True)
    logging.info("Data fetched successfully.")
    return data

# ========== STEP 4: Store Data in SQL (SQLite) ==========
def store_to_sql(data, db_name='coca_cola_stock.db'):
    logging.info(f"Storing data to SQLite database: {db_name}...")
    conn = sqlite3.connect(db_name)
    data.to_sql('stock_data', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()
    logging.info("Data stored successfully.")

# ========== STEP 5: Load Data from SQL ==========
def load_from_sql(db_name='coca_cola_stock.db'):
    logging.info(f"Loading data from SQLite database: {db_name}...")
    conn = sqlite3.connect(db_name)
    df = pd.read_sql('SELECT * FROM stock_data', conn)
    conn.close()
    logging.info("Data loaded successfully.")
    return df

# ========== STEP 6: Exploratory Data Analysis ==========
def plot_eda(df):
    logging.info("Generating EDA plot...")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Date', y='Close')
    plt.title('Coca-Cola Stock Closing Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    logging.info("EDA plot displayed.")

# ========== STEP 7: Feature Engineering ==========
def create_features(df):
    logging.info("Performing feature engineering...")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Weekday'] = df['Date'].dt.weekday
    df['Close_shifted'] = df['Close'].shift(-1)
    df = df.dropna()
    logging.info("Feature engineering completed.")
    return df

# ========== STEP 8: Machine Learning Model ==========
def train_model(df):
    logging.info("Training machine learning model...")
    features = ['Open', 'High', 'Low', 'Volume', 'Day', 'Month', 'Year', 'Weekday']
    target = 'Close_shifted'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)

    logging.info(f'RMSE: {rmse:.2f}')
    logging.info(f'MAE: {mae:.2f}')

    return model

# ========== STEP 9: Export to Excel ==========
def export_to_excel(df, filename='CocaCola_Stock_Report.xlsx'):
    logging.info(f"Exporting data to Excel file: {filename}...")
    df.to_excel(filename, index=False)
    logging.info("Data exported successfully.")

# ========== STEP 10: Run Full Pipeline ==========
def run_pipeline():
    logging.info("Starting Coca-Cola Stock Analysis Pipeline...")

    # Step 1: Fetch
    data = fetch_data()

    # Step 2: Store
    store_to_sql(data)

    # Step 3: Load
    df = load_from_sql()

    # Step 4: EDA
    plot_eda(df)

    # Step 5: Features
    df = create_features(df)

    # Step 6: ML Model
    train_model(df)

    # Step 7: Excel Export
    export_to_excel(df)

    logging.info("Pipeline execution completed successfully.")

# ========== Main Execution ==========
if __name__ == "__main__":
    run_pipeline()
