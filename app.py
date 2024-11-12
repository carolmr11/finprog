#===============================================================================
#Call Libreries
#===============================================================================
import streamlit as st
import numpy as np                        # Array, Calculation
import pandas as pd                       # DataFrame
import matplotlib.pyplot as plt           # Visualization
import plotly.express as px               # Visualization
import plotly.graph_objects as go         # Visualization
from datetime import datetime, timedelta  # Date-time
import yfinance as yf
#==============================================================================
# HOT FIX FOR YFINANCE .INFO METHOD Taken from class 5 document
# Ref: https://github.com/ranaroussi/yfinance/issues/1729
#==============================================================================

import requests
import urllib

class YFinance:
    user_agent_key = "User-Agent"
    user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/58.0.3029.110 Safari/537.36")
    
    def __init__(self, ticker):
        self.yahoo_ticker = ticker

    def __str__(self):
        return self.yahoo_ticker

    def _get_yahoo_cookie(self):
        cookie = None

        headers = {self.user_agent_key: self.user_agent_value}
        response = requests.get("https://fc.yahoo.com",
                                headers=headers,
                                allow_redirects=True)

        if not response.cookies:
            raise Exception("Failed to obtain Yahoo auth cookie.")

        cookie = list(response.cookies)[0]

        return cookie

    def _get_yahoo_crumb(self, cookie):
        crumb = None

        headers = {self.user_agent_key: self.user_agent_value}

        crumb_response = requests.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers=headers,
            cookies={cookie.name: cookie.value},
            allow_redirects=True,
        )
        crumb = crumb_response.text

        if crumb is None:
            raise Exception("Failed to retrieve Yahoo crumb.")

        return crumb

    @property
    def info(self):
        # Yahoo modules doc informations :
        # https://cryptocointracker.com/yahoo-finance/yahoo-finance-api
        cookie = self._get_yahoo_cookie()
        crumb = self._get_yahoo_crumb(cookie)
        info = {}
        ret = {}

        headers = {self.user_agent_key: self.user_agent_value}

        yahoo_modules = ("assetProfile,"  # longBusinessSummary
                         "summaryDetail,"
                         "financialData,"
                         "majorHoldersBreakdown,"
                         "indexTrend,"
                         "defaultKeyStatistics,"
                         "majorHoldersBreakdown,"
                         "insiderHolders")

        url = ("https://query1.finance.yahoo.com/v10/finance/"
               f"quoteSummary/{self.yahoo_ticker}"
               f"?modules={urllib.parse.quote_plus(yahoo_modules)}"
               f"&ssl=true&crumb={urllib.parse.quote_plus(crumb)}")

        info_response = requests.get(url,
                                     headers=headers,
                                     cookies={cookie.name: cookie.value},
                                     allow_redirects=True)

        info = info_response.json()
        info = info['quoteSummary']['result'][0]

        for mainKeys in info.keys():
            for key in info[mainKeys].keys():
                if isinstance(info[mainKeys][key], dict):
                    try:
                        ret[key] = info[mainKeys][key]['raw']
                    except (KeyError, TypeError):
                        pass
                else:
                    ret[key] = info[mainKeys][key]

        return ret
#===============================================================================
#programing for the header
#===============================================================================
#Here you can find the home page where people can find the title and choose the features needed
st.title("S&P500 PERFORMANCE")
col1, col2 = st.columns((1,5))
col1.write("Data Source:")
col2.image('Yfinancelogo.jpeg', width=100)
tlist = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
col1, col2, col3 = st.columns(3)
ticker = col1.selectbox("Ticker", tlist)
start_date = col2.date_input("Start date", datetime.today().date() - timedelta(days=30))
end_date = col3.date_input("End date", datetime.today().date())
#===============================================================================
#programing for the summary page
#===============================================================================
def sumup(ticker):
    # Get the company information
    @st.cache_data
    def GetCompanyInfo(ticker):
        """
        This function get the company information from Yahoo Finance.
        """
        return yf.Ticker(ticker).info
    
    # If the ticker is already selected
    if ticker != '':
        # Get the company information in list format
        info = GetCompanyInfo(ticker)
        
        # Show the company description using markdown + HTML #Taken from class exercises
        st.write('**1. Business Summary:**')
        st.markdown('<div style="text-align: justify;">' + \
                    info['longBusinessSummary'] + \
                    '</div><br>',
                    unsafe_allow_html=True)
        
        # Show some statistics as a DataFrame
        st.write('**2. Key Statistics:**')
        info_keys = {'previousClose':'Previous Close',
                     'open'         :'Open',
                     'bid'          :'Bid',
                     'ask'          :'Ask',
                     'marketCap'    :'Market Cap',
                     'volume'       :'Volume'}
        company_stats = {info_keys[key]: info.get(key, "N/A") for key in info_keys}  # Dictionary
        company_stats = pd.DataFrame({'Value':pd.Series(company_stats)})  # Convert to DataFrame
        st.dataframe(company_stats)

@st.cache_data
def getdata(ticker, start_date, end_date):
    stock = yf.Ticker(ticker).history(start=start_date, end=end_date)
    stock.reset_index(inplace=True)
    stock['Date'] = stock['Date'].dt.date
    return stock
#creating chart for summary
def createsct(ticker, start_date, end_date):
    from plotly.subplots import make_subplots
    buttoms_f = [{'count': 1, 'label': "1M", 'step': 'month', 'stepmode': 'backward'},
            {'count': 6, 'label': "6M", 'step': 'month', 'stepmode': 'backward'},
            {'count': 1, 'label': "YTD", 'step': 'year', 'stepmode': 'todate'},
            {'count': 3, 'label': "3Y", 'step': 'year', 'stepmode': 'backward'},
            {'count': 5, 'label': "5Y", 'step': 'year', 'stepmode': 'backward'},
            {'step': 'all'}]
    df = getdata(ticker, start_date, end_date)
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], fill='tozeroy', fillcolor='rgba(133,133,241,0.2)', showlegend=False), secondary_y=True)
    fig.update_layout(title=f'{ticker} Historical Stock Price', xaxis_title='Time', yaxis_title='Stock Price (USD)', showlegend=False, xaxis=dict(rangeselector=dict(buttons=buttoms_f)))
    fig.update_yaxes(range=[0,1000000000], secondary_y=False, showticklabels=False)
    return fig
#================================================================
#organize programing in the summary page
#================================================================
def summary():
    st.title("Summary and Historical Prices")
    st.write(sumup(ticker), createsct(ticker, start_date, end_date))
#================================================================
#programing the chart page
#================================================================
def sctbarchart():
        from plotly.subplots import make_subplots
        buttoms_f = [{'count': 1, 'label': "1MTD", 'step': 'month', 'stepmode': 'backward'},
            {'count': 6, 'label': "6M", 'step': 'month', 'stepmode': 'backward'},
            {'count': 1, 'label': "YTD", 'step': 'year', 'stepmode': 'todate'},
            {'count': 3, 'label': "3Y", 'step': 'year', 'stepmode': 'backward'},
            {'count': 5, 'label': "5Y", 'step': 'year', 'stepmode': 'backward'},
            {'step': 'all'}]
        df = getdata(ticker, start_date, end_date)
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Scatter(x=df["Date"], y=df['Close'], fill='tozeroy', fillcolor='rgba(133,133,241,0.2)', showlegend=False), secondary_y=True)
        fig.add_trace(go.Bar(x=df["Date"], y=df['Volume'],marker_color=np.where(df['Close'].pct_change() > 0, 'green', 'red')))
        fig.update_layout(title=f'{ticker} Stock Price', xaxis_title='Time', yaxis_title='Stock Price (USD)', showlegend=False,xaxis=dict(rangeselector=dict(buttons=buttoms_f)))
        fig.update_yaxes(range=[0,100000000], secondary_y=False, showticklabels=False)
        return fig
def candlechart():
        from plotly.subplots import make_subplots
        buttoms_f = [{'count': 1, 'label': "1MTD", 'step': 'month', 'stepmode': 'backward'},
            {'count': 6, 'label': "6M", 'step': 'month', 'stepmode': 'backward'},
            {'count': 1, 'label': "YTD", 'step': 'year', 'stepmode': 'todate'},
            {'count': 3, 'label': "3Y", 'step': 'year', 'stepmode': 'backward'},
            {'count': 5, 'label': "5Y", 'step': 'year', 'stepmode': 'backward'},
            {'step': 'all'}]
        df = getdata(ticker, start_date, end_date)
        fig = make_subplots(specs=[[{"secondary_y":False}]])  
        fig.add_trace(go.Candlestick(x=df["Date"], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], showlegend=False))
        fig.update_layout(title=f'{ticker} Stock Price', xaxis_title='Time', yaxis_title='Stock Price', showlegend=False, xaxis=dict(rangeselector=dict(buttons=buttoms_f)))
        return fig
#================================================================
#organizing information for the chart page
#================================================================
def chart():
    candle = st.checkbox("Show Candle Stick Chart")
    if candle:
        st.write(candlechart())
    else: st.write(sctbarchart())
#================================================================
#programing financial statements
#================================================================
@st.cache_data
def finstate(ticker, financial, period):
    stock2= yf.Ticker(ticker)
    balancesheet = stock2.balance_sheet
    if financial == 'balance_sheet' and period == "anual":
        return balancesheet
    qbalancesheet = stock2.quarterly_balance_sheet
    if financial == 'balance_sheet' and period == 'quarter':
        return qbalancesheet
    incomestatement = stock2.income_stmt
    if financial == 'income_statement' and period == 'anual':
        return incomestatement
    qincomestatement = stock2.quarterly_income_stmt
    if financial == 'income_statement' and period == 'quarter':
        return qincomestatement
    cashflow = stock2.cashflow
    if financial == 'cashflow' and period == 'anual':
        return cashflow
    qcashflow = stock2.quarterly_cashflow
    if financial == 'cashflow' and period == 'quarter':
        return qcashflow
#=====================================================================
#organizing information of financial statement
#=====================================================================
def Financials():
    st.write('You can find the different Financial Statements of each one of the S&P500 companies')
    def balance_sheet_anual():
        st.subheader("Anual Balance Sheet")
        st.write(finstate(ticker, 'balance_sheet', "anual"))
    def balance_sheet_quarter():
        st.subheader("Quarter Balance Sheet")
        st.write(finstate(ticker, 'balance_sheet', "quarter"))
    def income_statement_anual():
        st.subheader("Anual Income Statement")
        st.write(finstate(ticker, 'income_statement', "anual"))
    def income_statetment_quarter():
        st.subheader("Quarter Income Statement")
        st.write(finstate(ticker, 'income_statement', "quarter"))
    def cashflow_anual():
        st.subheader("Anual Cash Flow")
        st.write(finstate(ticker, 'cashflow', "anual"))
    def cashflow_quarter():
        st.subheader("Quarter Cash Flow")
        st.write(finstate(ticker, 'cashflow', "quarter"))
    report = st.radio(label="",options=["Balance Sheet", "Income Statement", "Cash Flow"])
    period = st.radio(label="",options=["anual", "quarter"])
    if report == "Balance Sheet" and period == "anual":
        balance_sheet_anual()
    elif report == "Balance Sheet" and period == "quarter":
        balance_sheet_quarter()
    elif report == "Income Statement" and period == "anual":
        income_statement_anual()
    elif report == "Income Statement" and period == "quarter":
        income_statetment_quarter()
    elif report == "Cash Flow" and period == "anual":
        cashflow_anual()
    elif report == "Cash Flow" and period == "quarter":
        cashflow_quarter()
#=====================================================================
#Programming montecarlo statement #taken from class execises
#=====================================================================
stock_price = getdata(ticker, start_date, end_date)
seed = 10
def run_simulation(stock_price, time_horizon, n_simulation, seed):
    # Daily return (of close price)
    daily_return = stock_price['Close'].pct_change()
    # Daily volatility (of close price)
    daily_volatility = np.std(daily_return)

    # Run the simulation
    np.random.seed(seed)
    simulation_df = pd.DataFrame()  # Initiate the data frame

    for i in range(n_simulation):

        # The list to store the next stock price
        next_price = []

        # Create the next stock price
        last_price = stock_price['Close'].iloc[-1]

        for j in range(time_horizon):

            # Generate the random percentage change around the mean (0) and std (daily_volatility)
            future_return = np.random.normal(0, daily_volatility)

            # Generate the random future price
            future_price = last_price * (1 + future_return)

            # Save the price and go next
            next_price.append(future_price)
            last_price = future_price

        # Store the result of the simulation
        next_price_df = pd.Series(next_price).rename('sim' + str(i))
        simulation_df = pd.concat([simulation_df, next_price_df], axis=1)

    return simulation_df
def plot_simulation_price(stock_price, simulation_df):
    """
    This function plot the simulated stock prices using line plot.
    
    Input:
        - stock_price : A DataFrame store the stock price (from Yahoo Finance)
        - simulation_df : A DataFrame stores the simulated prices
        
    Output:
        - Plot the stock prices
    """
    # Plot the simulation stock price in the future
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(simulation_df)
    ax.set_title('Monte Carlo simulation for the stock price in next ' + str(simulation_df.shape[0]) + ' days')
    ax.set_xlabel('Day')
    ax.set_ylabel('Price')
    ax.axhline(y=stock_price['Close'].iloc[-1], color='red')
    ax.legend(['Current stock price is: ' + str(np.round(stock_price['Close'].iloc[-1], 2))])
    ax.get_legend().legend_handles[0].set_color('red')
    return fig
def value_at_risk(stock_price, simulation_df):
    """
    This function calculate the Value at Risk (VaR) of the stock based on the Monte Carlo simulation.
    
    Input:
        - stock_price : A DataFrame store the stock price (from Yahoo Finance)
        - simulation_df : A DataFrame stores the simulated prices
        
    Output:
        - VaR value
    """
        
    # Price at 95% confidence interval
    future_price_95ci = np.percentile(simulation_df.iloc[-1:, :].values[0, ], 5)

    # Value at Risk
    VaR = stock_price['Close'].iloc[-1] - future_price_95ci
    return('VaR at 95% confidence interval is: ' + str(np.round(VaR, 2)) + ' USD')
#=====================================================================
#organizing information of montecarlo statement
#=====================================================================
def Montecarlo():
    col1, col2 = st.columns(2)
    time_horizon= col1.selectbox("Time Horizon",[30, 60, 90])
    n_simulation = col2.selectbox("Number of Simulations", [200, 500, 1000])
    st.write(plot_simulation_price(stock_price, run_simulation(stock_price, time_horizon, n_simulation, seed)), value_at_risk(stock_price,  run_simulation(stock_price, time_horizon, n_simulation, seed)))
#=====================================================================
#Programming plus page
#=====================================================================
def insights(ticker1, ticker2):
    ticker_str1 = str(ticker1)
    stock1 = yf.Ticker(ticker_str1)
    stock_p1 = getdata(ticker1, start_date, end_date)
    daily_return1 = stock_p1['Close'].pct_change()
    ticker_str2 = str(ticker2)
    stock2 = yf.Ticker(ticker_str2)
    stock_p2 = getdata(ticker2, start_date, end_date)
    daily_return2 = stock_p2['Close'].pct_change()
    tickerinfo = {"company_one" :{"Ticker1": ticker1,
                "Daily Volatility": np.std(daily_return1),
                "Sector": stock1.info.get('sector', None),
                "Industry": stock1.info.get('industry', None),
                "Enterprise Value": stock1.info.get('enterpriseValue', None),
                "Gross Profits": stock1.info.get('grossProfits', None),
                "Total Revenue": stock1.info.get('totalRevenue', None),
                "Gross Margins": stock1.info.get('grossMargins', None),
                "ebitdaMargins": stock1.info.get('ebitdaMargins', None)},
                "company_two" :{"Ticker1": ticker2,
                "Daily Volatility": np.std(daily_return2),
                "Sector": stock2.info.get('sector', None),
                "Industry": stock2.info.get('industry', None),
                "Enterprise Value": stock2.info.get('enterpriseValue', None),
                "Gross Profits": stock2.info.get('grossProfits', None),
                "Total Revenue": stock2.info.get('totalRevenue', None),
                "Gross Margins": stock2.info.get('grossMargins', None),
                "ebitdaMargins": stock2.info.get('ebitdaMargins', None)}}
    pd.DataFrame(tickerinfo)
    return tickerinfo
#=====================================================================
#organizing information of plus statement
#=====================================================================
def Plus():
    ticker2 = st.selectbox("Select a company to compare", tlist)
    st.dataframe(insights(ticker, ticker2))
#=====================================================================
#organizing the whole page
#=====================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary", "Chart", "Financials", "Montecarlo", "Plus"])
with tab1:
    summary()
with tab2:
    chart()
with tab3:
    Financials()
with tab4:
    Montecarlo()
with tab5:
    Plus()