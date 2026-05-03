import streamlit as st
from datetime import date
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import altair as alt
import numpy as np

START = "2015-01-01"

st.set_page_config(page_title="AI Stock Predictor", layout="wide")
st.title("📈 AI-Powered Stock Forecasting & Visualization Dashboard")

with st.sidebar:
    st.header("Control Panel")
    stocks = (
        "AAPL", "RELIANCE.NS", "HDB", "IBN", "SBIN.NS",
        "HINDUNILVR.NS", "INFY", "BAJFINANCE.NS", "LICI.NS", "ITC.NS", "LT.NS",
        "MARUTI.NS", "M&M.NS", "HCLTECH.NS", "KOTAKBANK.NS", "SUNPHARMA.NS",
        "ULTRACEMCO.NS", "AXISBANK.BO", "TITAN.NS", "NTPC.NS", "BAJAJFINSV.NS",
        "DMART.NS", "ONGC.NS", "HAL.NS", "ETERNAL.NS", "ADANIPORTS.NS",
        "BEL.NS", "POWERGRID.NS", "WIT", "ADANIENT.NS", "JSWSTEEL.NS",
        "TATAMOTORS.NS", "BAJAJ-AUTO.NS", "ASIANPAINT.NS", "COALINDIA.NS",
        "ADANIPOWER.NS", "NESTLEIND.NS", "INDIGO.NS", "TATASTEEL.NS",
        "HYUNDAI.NS", "JIOFIN.NS", "IOC.NS", "TRENT.NS", "GRASIM.NS", "DLF.NS",
        "HINDZINC.NS", "SBILIFE.NS", "EICHERMOT.NS", "VEDL.NS", "VBL.NS",
        "HDFCLIFE.NS", "DIVISLAB.NS", "HINDALCO.NS", "TVSMOTOR.NS", "IRFC.NS",
        "PIDILITIND.NS", "ADANIGREEN.NS", "LTIM.NS", "BAJAJHLDNG.NS",
        "AMBUJACEM.NS", "BRITANNIA.NS", "BPCL.NS", "TECHM.NS", "GODREJCP.NS",
        "PFC.NS", "SOLARINDS.NS", "CIPLA.NS", "TATAPOWER.NS", "BANKBARODA.NS",
        "BOSCHLTD.NS", "TORNTPHARM.NS", "CHOLAFIN.NS", "LODHA.NS", "HDFCAMC.NS",
        "PNB.NS", "GAIL.NS", "CGPOWER.NS", "SIEMENS.NS", "MAXHEALTH.NS",
        "MUTHOOTFIN.NS", "APOLLOHOSP.NS", "INDHOTEL.NS", "ABB.NS", "MAZDOCK.NS",
        "SHRIRAMFIN.NS", "SHREECEM.NS", "TATACONSUM.NS", "POLYCAB.NS",
        "DIXON.NS", "HEROMOTOCO.NS", "CUMMINSIND.NS", "RDY", "MANKIND.NS",
        "JINDALSTEL.NS", "ZYDUSLIFE.NS", "MOTHERSON.NS", "HAVELLS.NS",
        "SWIGGY.NS", "UNIONBANK.NS","GMBREW.NS"
    ) 
    selected_stock = st.selectbox("Select Stock Ticker", stocks)
    n_years = st.slider("Forecast Period (Years):", 1, 5)
    period = n_years * 365
    predict_button = st.button("Generate AI Prediction")
    st.info("The model uses Meta's Prophet algorithm to decompose trends and seasonality.")

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start=START)
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

@st.cache_resource
def get_prediction(df, days):
    df_train = df[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=days)
    forecast = m.predict(future)
    return m, forecast

data = load_data(selected_stock)

if data.empty:
    st.error(f"No data found for {selected_stock}. The ticker might be delisted or API is throttled.")
else:
    if selected_stock.endswith(".NS") or selected_stock.endswith(".BO"):
        currency_symbol = "₹"
    else:
        currency_symbol = "$"

    col_left, col_right = st.columns([3, 1])

    with col_left:
        st.subheader(f"Historical Price Action: {selected_stock}")
        base = alt.Chart(data).encode(x='Date:T')
        line = base.mark_line(color='#1f77b4').encode(y=alt.Y('Close:Q', title=f'Price ({currency_symbol})'))
        st.altair_chart(line.properties(height=400).interactive(), use_container_width=True)

    with col_right:
        st.subheader("Latest Market Data")
        last_price = data['Close'].iloc[-1]
        if len(data) > 1:
            prev_price = data['Close'].iloc[-2]
            change = last_price - prev_price
        else:
            change = 0
       
        st.metric("Current Price", f"{currency_symbol}{last_price:.2f}", f"{change:.2f}")
        st.write("Recent Logs", data.tail(5))

    if predict_button:
        st.divider()
        with st.spinner("AI is analyzing historical cycles and trends..."):
            model, forecast = get_prediction(data, period)
            st.subheader('🚀 Future Forecast Analysis')

            fig1 = plot_plotly(model, forecast)
            fig1.update_layout(title=f"Forecast for {selected_stock} ({n_years} Years Out)")
            st.plotly_chart(fig1, use_container_width=True)

            st.subheader("📥 Export Forecast Data")
            csv_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(period)
            csv_data.columns = ['Date', 'Predicted_Price', 'Minimum_Expected', 'Maximum_Expected']
            csv_data['Date'] = csv_data['Date'].dt.strftime('%Y-%m-%d')
            csv = csv_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Prediction as CSV",
                data=csv,
                file_name=f'{selected_stock}_forecast.csv',
                mime='text/csv',
            )
            
            with st.expander("Show AI Logic (Trend & Seasonality)"):
                st.write("The model breaks the stock into these components:")
                fig2 = model.plot_components(forecast)
                st.pyplot(fig2)
    else:
        st.info("Adjust the settings in the sidebar and click 'Generate AI Prediction' to start.")