from neuralintents import GenericAssistant
import pandas as pd
import pandas_datareader as web
import mplfinance as mpf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import linear_model
import pickle
import sys
import datetime as dt
from tabulate import tabulate

# Computation of the hotness score ###############################################################

# Load CSV data
data = pd.read_csv('stocks.csv')

# Drop rows with null values
data.dropna(inplace=True)

# Get asset universe
assetUniverse = data.asset.unique().tolist()

# Get Hotness score

hotScores = {}

for asset in assetUniverse:

    try:

        # Get hot score
        df = data[data.asset == asset]
        score = 0.33 * df.twitter_volume.mean() + 0.33 * df.trendvalue.mean() + 0.33 * df.news_volume.mean() + 0.33 * df.reddit_volume.mean()
        hotScores[asset] = [score]

        # Get asset training and test data
        df_train = data[data.asset == asset].iloc[:-5]
        df_test = data[data.asset == asset].iloc[-5:]

        # Define factors and target variable
        factors = ['trendvalue', 'twitter_volume', 'twitter_sentiment', 'news_volume', 'news_sentiment',
                   'reddit_volume', 'reddit_sentiment']
        X = df_train[factors]
        Y = df_train['price']

        # OLS Multiple Linear Regression
        regr = linear_model.LinearRegression()
        regr.fit(X, Y)

        # Get price target
        price_target = regr.predict([df_test[factors].iloc[-1].values.tolist()])[0]

        # Print regression summary
        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()
        predictions = model.predict(X)
        summary = model.summary()
        


    except Exception as e:
        print(e)
        continue

# Get Top 5 Hottest
hotdf = pd.DataFrame(hotScores).T
hotdf.reset_index(inplace=True)
hotdf.columns = ['asset', 'hotness']
hotdf.sort_values(by='hotness', inplace=True, ascending=False)
top5_hot = hotdf.iloc[:5, :]
bottom5_hot = hotdf.iloc[-5:, :]
print(tabulate(hotdf.values.tolist()[:5], hotdf.columns.tolist(), tablefmt="psql"))
# Get Bottom 5
print(tabulate(hotdf.values.tolist()[-5:], hotdf.columns.tolist(), tablefmt="psql"))

# AI CHATBOT ###########################################################################################################

# Training of the AI chatbot model

assistant_AI = GenericAssistant('intents.json', intent_methods=intents_mapping,
                                model_name="AI_Financial_Assistant_model")
# 1st time initialization of the model
assistant_AI.train_model()
assistant_AI.save_model()
# After first initialization (load model instead)
assistant_AI.load_model(model_name="AI_Financial_Assistant_model")

while True:
    message = input("")
    assistant_AI.request(message)  # Ask something to the assistant


portfolio = {}
with open('portfolio.pkl', 'wb') as f:
    pickle.dump(portfolio, f)

with open('portfolio.pkl', 'rb') as f:
    portfolio = pickle.load(f)


def hotScores():
    ticker = input("Enter the ticker of the stock: ")
    asset_hot = hotdf.loc[hotdf['asset'] == ticker]
    print(f"For {ticker}, the score of Hotness is: ", asset_hot.iloc[0, 1])



# Saves any modification made to the portfolio
def save_portfolio():
    with open('portfolio.pkl', 'wb') as f:
        pickle.dump(portfolio, f)

def user_help():
    print("I am able to the do the current functions for you: \n"
          "1 - Show you your portfolio and all the shares it contains\n"
          "2 - Add the desired amount of shares of a firm to your portfolio\n"
          "3 - Remove a certain amount of shares of a firm from your portfolio\n"
          "4 - Calculate the current worth of your portfolio\n"
          "5 - Compute the portfolio gains you made in stock value when compared to the date of your choice\n"
          "6 - Chart the price evolution of a stock from a requested date up to today")

# Add a firm's share into the user's portfolio
def add_portfolio():
    ticker = input("Which stock would you like to add to your portfolio ? : ")
    nb_shares = input("How many shares would you like to buy ? : ")

    if ticker in portfolio.keys():
        portfolio[ticker] += int(nb_shares)
    else:
        portfolio[ticker] = int(nb_shares)
    save_portfolio()


# Removes a share from the user's porfolio
def remove_portfolio():
    ticker = input("Which stock would you like to sell from your portfolio ? : ")
    nb_shares = input("How many shares would you like to sell ? : ")

    if ticker in portfolio.keys():
        if int(nb_shares) <= portfolio[ticker]:
            portfolio[ticker] -= int(nb_shares)
            if portfolio[ticker] == 0:
                portfolio.pop(ticker)
            save_portfolio()
        else:
            print("You don't have enough shares to do that!")
    else:
        print(f"You don't own any shares of {ticker}")


# Show the number of shares in the user's porfolio
def show_porfolio():
    print("Your portfolio: ")
    for ticker in portfolio.keys():
        print(f"You own {portfolio[ticker]} shares of {ticker}")


# Returns the value of the actual porfolio in USD
def portfolio_worth():
    sum = 0
    for ticker in portfolio.keys():
        stock_data = web.DataReader(ticker, 'yahoo')
        stock_price = stock_data['Close'].iloc[-1]
        sum += stock_price
    print(f"Your portfolio is worth {sum} USD")


# Showcase the gain of value of the porfolio stocks compared to the user inputted date
def portfolio_gains():
    start_date = input("Enter the date you want to use for comparison (YYYY-MM-DD): ")
    sum_today = 0
    sum_start = 0
    try:
        for ticker in portfolio.keys():
            stock_data = web.DataReader(ticker, 'yahoo')
            stock_price_today = stock_data['Close'].iloc[-1]
            stock_price_start = stock_data.loc[stock_data.index == start_date]['Close'].values[0]
            sum_today += stock_price_today
            sum_start += stock_price_start
        print(f"Relative Gains: {((sum_today - sum_start) / sum_start) * 100}%")
        print(f"Absolute Gains: {sum_today - sum_start} USD")
    except IndexError:
        print("There wasn't any trading happening on this day !")


# Allows to plot a specific stock in a candle chart
def plot_chart():
    ticker = input("Choose a ticker symbol: ")
    start_string = input("Choose a starting date (DD/MM/YYYY): ")

    start_date = dt.datetime.strptime(start_string, "%d%m%Y")
    end_date = dt.datetime.now()
    stock_data = web.DataReader(ticker, 'yahoo', start_date, end_date)

    # Visual set up for the candlestick chart (from mplfinance)
    plt.style.use('dark_background')
    colors = mpf.make_marketcolors(up='#00ff00', down='#ff0000', volume='in', wick='inherit', edge='inherit')
    # Up = Green, Down = Red
    mpf_style = mpf.make_mpf_style(base_mpf_style='mike', marketcolors=colors)
    mpf.plot(stock_data, type='candle', style=mpf_style, volume=True)


def bye():
    sys.exit(0)


# Mapping of the intents from the .json file to train the AI chatbot to recognize patterns of speech and requests
intents_mapping = {
    'plot_chart': plot_chart,
    'add_portfolio': add_portfolio,
    'remove_portfolio': remove_portfolio,
    'show_portfolio': show_porfolio,
    'portfolio_worth': portfolio_worth,
    'portfolio_gains': portfolio_gains,
    'bye': bye,
    'user_help': user_help,

}
