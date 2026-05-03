# 📈 AI-Driven Stock Forecaster & Visualization Dashboard

A high-performance financial analytics tool built with **Python** and **Streamlit**. This application utilizes Meta's **Prophet** algorithm to provide multi-year price predictions while offering an interactive interface for historical trend analysis.

## 🌟 Key Features
- **Predictive AI Modeling**: Leverages Meta's Prophet to decompose stock data into trend, weekly, and yearly seasonality.
- **Dynamic Multi-Ticker Support**: Supports 100+ tickers including major Indian (NSE/BSE) and Global (NASDAQ) stocks.
- **Smart Currency Detection**: Automatically toggles between **₹ (INR)** and **$ (USD)** based on the exchange suffix (.NS/.BO).
- **Interactive Visuals**: Powered by **Altair** and **Plotly** for responsive, zoomable price action charts.
- **Forecast Export**: Integrated CSV downloader for AI-generated prediction data (Predicted Price, Upper/Lower bounds).
- **Optimized Performance**: Uses `@st.cache_data` and `@st.cache_resource` for lightning-fast data fetching and model re-runs.

## 🛠️ Technical Stack
- **Framework**: Streamlit
- **Machine Learning**: Meta Prophet
- **Data Source**: Yahoo Finance API (yfinance)
- **Visualization**: Plotly, Altair, Matplotlib
- **Data Science**: Pandas, NumPy

## ⚙️ Installation & Setup
1. **Clone the repo**:
   ```bash
 git clone https://github.com/Nagendra-071/ai-stock-forecaster-visualization-dashboard.git


 🧠 How the AI Works
The application treats stock price as a time-series problem. By applying the Prophet model, the app identifies:

Trend: The overall long-term direction of the stock.

Seasonality: Recurring patterns (e.g., "Does this stock perform better on certain days?").

Forecasting: Predicts future prices based on historical "changepoints."

📝 Disclaimer
This tool is for educational purposes only. Financial markets are volatile. AI predictions should not be used as the sole basis for investment decisions.
