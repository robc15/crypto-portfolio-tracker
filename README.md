# Crypto Portfolio Tracker

This project is a cryptocurrency portfolio tracker built using Streamlit. It allows users to input transactions, track their portfolio, set sell targets, and visualize performance. The app uses a local SQLite database for persistent storage and fetches live prices from CoinGecko.

## Features

- Add, view, and delete cryptocurrency transactions (purchases and sales)
- Track portfolio value, realized and unrealized P/L, and set sell targets
- Visualize portfolio allocation, performance, and value trends
- Fetch live prices for supported coins using the CoinGecko API
- Store and update daily portfolio value history in a local SQLite database
- Modern, interactive UI built with Streamlit

## Requirements

- Python 3.8+
- The following Python packages (see `requirements.txt`):
  - streamlit
  - requests
  - python-dotenv
  - sqlalchemy
  - pandas
  - plotly
  - numpy

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/robc15/crypto-portfolio-tracker.git
   cd crypto-portfolio-tracker
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Running the Application

To run the Streamlit application, use the following command:

```sh
streamlit run app.py
```

This will start the application, and you can access it in your web browser at `http://localhost:8501`.

## Usage

1. **Add Transactions:** Use the "Add New Transaction" section to record purchases or sales of cryptocurrencies. For sales, you can specify if you are selling to USD or receiving Bitcoin (BTC Satoshis).
2. **Portfolio Overview:** View your total portfolio value, realized and unrealized profit/loss, and top movers.
3. **Set Sell Targets:** Set target prices for each coin. The app will show your progress toward these targets.
4. **Visualizations:** Explore allocation pie charts, performance bar charts, and historical value trends.
5. **Delete Transactions:** Remove any transaction using the delete button in the transactions table.

## Notes

- The app uses a local SQLite database (`portfolio.db`) for persistent storage of transactions and daily portfolio values.
- CoinGecko API is used for live price data. If you encounter rate limits, try again later.
- All calculations and visualizations are performed locally.

## Contributing

Contributions are welcome! Please submit issues or pull requests for improvements or new features.

## License

This project is licensed under the MIT License.
