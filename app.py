import streamlit as st
import requests
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import plotly.express as px

# --- Database setup ---
engine = create_engine('sqlite:///portfolio.db')
Base = declarative_base()


class PortfolioValueHistory(Base):
    __tablename__ = 'portfolio_value_history'
    id = Column(Integer, primary_key=True)
    date = Column(String, unique=True, index=True)  # Store date as YYYY-MM-DD string
    total_value_usd = Column(Float)


class PortfolioEntry(Base):
    __tablename__ = 'portfolio'
    id = Column(Integer, primary_key=True)
    coin = Column(String)
    coins_purchased = Column(Float)
    purchase_price = Column(Float)
    transaction_type = Column(String, default="Purchase")
    sell_to = Column(String, default=None)
    btc_price_at_sale = Column(Float, default=None)
    timestamp = Column(String, default=lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


class SellTarget(Base):
    __tablename__ = 'sell_targets'
    id = Column(Integer, primary_key=True)
    coin = Column(String, unique=True)
    target_price = Column(Float)


Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
db_session = Session()


# --- Utility functions (Database) ---
def add_entry(coin, coins_purchased, purchase_price, transaction_type, sell_to=None, btc_price_at_sale=None, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    entry = PortfolioEntry(
        coin=coin,
        coins_purchased=coins_purchased,
        purchase_price=purchase_price,
        transaction_type=transaction_type,
        sell_to=sell_to,
        btc_price_at_sale=btc_price_at_sale,
        timestamp=timestamp
    )
    db_session.add(entry)
    db_session.commit()


def get_portfolio():
    return db_session.query(PortfolioEntry).all()


def set_sell_target(coin, target_price):
    target = db_session.query(SellTarget).filter(SellTarget.coin == coin).first()
    if target:
        target.target_price = target_price
    else:
        target = SellTarget(coin=coin, target_price=target_price)
        db_session.add(target)
    db_session.commit()


def get_sell_targets():
    targets = db_session.query(SellTarget).all()
    return {t.coin: t.target_price for t in targets}


def record_daily_portfolio_value(date_obj: datetime.date, total_value_usd: float):  # Type hint corrected to datetime.date
    """
    Records or updates the total portfolio value for a given date.
    If no record exists for the date, a new one is created.
    If a record exists, it's updated only if the new total_value_usd is higher.

    date_obj: A datetime.date object.
    total_value_usd: The total portfolio value in USD.
    """
    if total_value_usd is None:  # Do not record if the value is None
        return

    date_str = date_obj.strftime('%Y-%m-%d')
    session = Session()  # Create a new session
    try:
        existing_entry = session.query(PortfolioValueHistory).filter_by(date=date_str).first()

        if existing_entry is None:
            # No entry for this date, create a new one
            new_entry = PortfolioValueHistory(date=date_str, total_value_usd=total_value_usd)
            session.add(new_entry)
            session.commit()
            # st.sidebar.info(f"Recorded initial portfolio value for {date_str}: ${total_value_usd:,.2f}") # Optional
        elif total_value_usd > existing_entry.total_value_usd:
            # New value is higher than the existing recorded value for the day, so update
            existing_entry.total_value_usd = total_value_usd
            session.commit()
            # st.sidebar.info(f"Updated portfolio value for {date_str} to a new high: ${total_value_usd:,.2f}") # Optional
        # else:
            # Current value is not higher than already recorded for today, do nothing
            # st.sidebar.info(f"Portfolio value for {date_str} (${total_value_usd:,.2f}) is not higher than recorded (${existing_entry.total_value_usd:,.2f}). No update.") # Optional

    except Exception as e:
        session.rollback()
        st.sidebar.error(f"Error updating/recording daily portfolio value for {date_str}: {e}")
    finally:
        session.close()


def get_trend_chart_data_from_db(start_date_obj: datetime, end_date_obj: datetime):
    """
    Fetches recorded portfolio values between two dates from the database.
    start_date_obj: A datetime.date object for the start of the range.
    end_date_obj: A datetime.date object for the end of the range.
    Returns: A Pandas DataFrame with 'date' and 'Total Portfolio Value' columns.
    """
    start_date_str = start_date_obj.strftime('%Y-%m-%d')
    end_date_str = end_date_obj.strftime('%Y-%m-%d')

    session = Session()  # Create a new session
    try:
        query = session.query(PortfolioValueHistory.date, PortfolioValueHistory.total_value_usd).\
            filter(PortfolioValueHistory.date >= start_date_str).\
            filter(PortfolioValueHistory.date <= end_date_str).\
            order_by(PortfolioValueHistory.date.asc())
        results = query.all()
    except Exception as e:
        st.error(f"Error fetching trend chart data from DB: {e}")
        results = []
    finally:
        session.close()

    df = pd.DataFrame(results, columns=['date', 'Total Portfolio Value'])
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])  # Convert string date from DB to datetime for Plotly
    return df


# --- CoinGecko API Utilities ---
COINGECKO_ID_OVERRIDES = {
    "Spectral": "spectral",
    "Bitcoin": "bitcoin",
    "Dogecoin": "dogecoin",
    "Ethereum": "ethereum",
    "Avalanche": "avalanche-2",
    "US Dollars": "us-dollars",  # Placeholder, not a real CoinGecko ID for price
    "ChainGPT": "chaingpt",
    "Polkadot": "polkadot",
    "ResearchCoin": "researchcoin",
    "Cellframe": "cellframe",
    "The Graph": "the-graph"
}


@st.cache_data(show_spinner=False)
def get_coingecko_coins_list():
    url = "https://api.coingecko.com/api/v3/coins/list"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching CoinGecko coins list: {e}. App may not function fully.")
        return []


def build_coingecko_id_maps(coins_list_data):
    name_to_id_map = {}
    symbol_to_id_map = {}
    if isinstance(coins_list_data, list):
        for coin_data in coins_list_data:
            if isinstance(coin_data, dict) and 'name' in coin_data and 'id' in coin_data:
                name_to_id_map[coin_data['name'].lower()] = coin_data['id']
            if isinstance(coin_data, dict) and 'symbol' in coin_data and 'id' in coin_data:
                symbol_to_id_map[coin_data['symbol'].lower()] = coin_data['id']
    return name_to_id_map, symbol_to_id_map


def lookup_coingecko_id(coin_name, symbol, name_to_id_map, symbol_to_id_map, coingecko_id_overrides):  # Added coingecko_id_overrides
    # Prioritize manual overrides
    for override_name, override_id in coingecko_id_overrides.items():  # Use the passed map
        if coin_name.strip().lower() == override_name.lower():
            return override_id

    # Try direct name match
    if coin_name.lower().strip() in name_to_id_map:
        return name_to_id_map[coin_name.lower().strip()]

    # Try direct symbol match if name fails
    if symbol and symbol.lower().strip() in symbol_to_id_map:
        return symbol_to_id_map[symbol.lower().strip()]

    # Try base name if name contains parentheses (e.g., "Coin Name (Symbol)")
    if '(' in coin_name and ')' in coin_name:
        base_name = coin_name.split('(')[0].strip()
        if base_name.lower() in name_to_id_map:
            return name_to_id_map[base_name.lower()]
    return None  # No ID found


@st.cache_data(ttl=600, show_spinner=False)
def fetch_latest_prices(user_coins, user_symbols, name_to_id_map, symbol_to_id_map):
    coingecko_ids = []
    id_map_debug = {}
    for coin, symbol in zip(user_coins, user_symbols):
        cg_id = lookup_coingecko_id(coin, symbol, name_to_id_map, symbol_to_id_map, COINGECKO_ID_OVERRIDES)  # New
        if not cg_id:
            cg_id = coin.lower().replace(" ", "-").replace("_", "-")  # Fallback ID
        coingecko_ids.append(cg_id)
        id_map_debug[coin] = cg_id

    latest_prices_from_api = {coin_name: None for coin_name in id_map_debug.keys()}

    valid_ids_for_api = [i for i in coingecko_ids if i and i != "us-dollars"]  # Exclude placeholder
    if not valid_ids_for_api:
        return latest_prices_from_api, id_map_debug, False

    ids_param = ','.join(valid_ids_for_api)
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids_param}&vs_currencies=usd"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        prices_api_response = response.json()

        for coin_name_key, mapped_cg_id in id_map_debug.items():
            if coin_name_key == "US Dollars":  # Handle USD explicitly if needed elsewhere
                latest_prices_from_api[coin_name_key] = 1.0
                continue
            if (
                mapped_cg_id in prices_api_response and
                'usd' in prices_api_response[mapped_cg_id]
            ):
                latest_prices_from_api[coin_name_key] = prices_api_response[mapped_cg_id]['usd']
            elif mapped_cg_id and mapped_cg_id not in prices_api_response:
                if coin_name_key != "US Dollars":
                    st.sidebar.warning(
                        f"Price data not in API response for '{coin_name_key}' (ID: {mapped_cg_id})."
                    )
        return latest_prices_from_api, id_map_debug, False

    except requests.exceptions.HTTPError as http_err:
        is_rate_limited = False
        error_message = f"HTTP error fetching prices: {http_err} for URL: {url}"
        if http_err.response is not None:
            if http_err.response.status_code == 404:
                error_message = f"CoinGecko API endpoint not found (404): {url}."
            elif http_err.response.status_code == 429:
                is_rate_limited = True
                error_message = (
                    f"CoinGecko API rate limit hit: {http_err} for URL: {url}. "
                    "Latest prices may not be shown."
                )
        st.warning(error_message)
        return latest_prices_from_api, id_map_debug, is_rate_limited
    except requests.exceptions.RequestException as req_err:
        st.warning(
            f"Network error fetching prices: {req_err} for URL: {url}. Latest prices may not be shown."
        )
        return latest_prices_from_api, id_map_debug, False
    except Exception as ex:
        st.warning(f"Unexpected error processing prices: {ex} for URL: {url}")
        return latest_prices_from_api, id_map_debug, False


# --- Initialize CoinGecko data globally ---
coins_list_data_global = get_coingecko_coins_list()
name_to_id_global, symbol_to_id_global = build_coingecko_id_maps(coins_list_data_global)
name_to_cg_symbol_global = {}
if isinstance(coins_list_data_global, list):
    name_to_cg_symbol_global = {
        str(c.get('name', '')).lower(): str(c.get('symbol', '')).upper()
        for c in coins_list_data_global if c.get('name') and c.get('symbol')
    }


# --- DataFrame Utilities ---
def build_transactions_df(portfolio_entries_list):
    df = pd.DataFrame(
        [
            (
                e.id,
                e.coin,
                e.coins_purchased,
                e.purchase_price,
                getattr(e, 'transaction_type', 'Purchase'),
                getattr(e, 'timestamp', '')
            )
            for e in portfolio_entries_list
        ],
        columns=[
            "ID", "Coin", "Coins Purchased", "Purchase Price (USD)", "Type", "Timestamp"
        ]
    )
    manual_symbol_overrides = {
        "Ethereum": "ETH", "Bitcoin": "BTC", "Avalanche": "AVAX", "ChainGPT": "CGPT",
        "Polkadot": "DOT", "ResearchCoin": "RSC", "Cellframe": "CELL", "The Graph": "GRT",
        "US Dollars": "USD"
    }

    def get_display_symbol(coin_name_str):
        if not isinstance(coin_name_str, str):
            return "N/A"
        if coin_name_str in manual_symbol_overrides:
            return manual_symbol_overrides[coin_name_str]
        cg_symbol = name_to_cg_symbol_global.get(coin_name_str.lower())
        if cg_symbol:
            return cg_symbol
        if len(coin_name_str) >= 4:
            return coin_name_str[:4].upper()
        return coin_name_str.upper()

    df["Symbol"] = df["Coin"].apply(get_display_symbol)
    desired_cols_order = [
        "ID", "Coin", "Symbol", "Coins Purchased", "Purchase Price (USD)", "Type", "Timestamp"
    ]
    current_cols = [col for col in desired_cols_order if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in current_cols]
    df = df[current_cols + remaining_cols]
    df["Coins Purchased"] = pd.to_numeric(df["Coins Purchased"], errors="coerce").round(8)
    df["Purchase Price (USD)"] = pd.to_numeric(df["Purchase Price (USD)"], errors="coerce")
    df["Total Cost"] = (df["Coins Purchased"] * df["Purchase Price (USD)"]).fillna(0.0)
    return df


def build_summary_df(transactions_df):
    df = transactions_df.copy()
    df["Coins Purchased Num"] = pd.to_numeric(df["Coins Purchased"], errors="coerce").fillna(0.0)
    df["Purchase Price (USD) Numeric"] = pd.to_numeric(df["Purchase Price (USD)"], errors='coerce').fillna(0.0)
    # Use pre-calculated Total Cost
    df["Total Cost Numeric"] = pd.to_numeric(df["Total Cost"], errors="coerce").fillna(0.0)
    df["Signed Coins"] = df.apply(
        lambda row: row["Coins Purchased Num"] if row["Type"] == "Purchase" else -row["Coins Purchased Num"],
        axis=1
    )

    summary_rows = []
    for coin_name in df["Coin"].unique():
        coin_df = df[df["Coin"] == coin_name].copy()
        if coin_name == "US Dollars":
            net_usd_balance = coin_df["Signed Coins"].sum()
            summary_rows.append({
                "Coin": "US Dollars",
                "Coins Purchased": net_usd_balance,
                "Total Cost": net_usd_balance,
                "Average Price (USD)": 1.0,
                "Realized P/L": 0.0
            })
            continue

        buys = coin_df[coin_df["Type"] == "Purchase"]
        sells = coin_df[coin_df["Type"] == "Sell"]
        # Sum cost only from actual buys
        total_coins_bought = buys["Coins Purchased Num"].sum()
        total_cost_of_buys = buys[buys["Purchase Price (USD) Numeric"] > 0]["Total Cost Numeric"].sum()

        avg_buy_price = (total_cost_of_buys / total_coins_bought) if total_coins_bought > 0 else 0.0

        realized_pl_for_coin = 0.0
        for _, sell_row in sells.iterrows():
            sell_qty = sell_row["Coins Purchased Num"]
            sale_price_per_coin = sell_row["Purchase Price (USD) Numeric"]
            cost_of_sold_coins = sell_qty * avg_buy_price
            proceeds_from_sale = sell_qty * sale_price_per_coin
            realized_pl_for_coin += (proceeds_from_sale - cost_of_sold_coins)

        net_coins_held = coin_df["Signed Coins"].sum()
        current_total_cost_of_held_coins = 0.0
        if net_coins_held > 1e-8:
            current_total_cost_of_held_coins = net_coins_held * avg_buy_price
        elif abs(net_coins_held) < 1e-8:
            net_coins_held = 0.0
            current_total_cost_of_held_coins = 0.0
            avg_buy_price = 0.0
        if abs(current_total_cost_of_held_coins) < 1e-8:
            current_total_cost_of_held_coins = 0.0

        summary_rows.append({
            "Coin": coin_name,
            "Coins Purchased": net_coins_held,
            "Total Cost": current_total_cost_of_held_coins,
            "Average Price (USD)": avg_buy_price if net_coins_held > 0 else 0.0,
            "Realized P/L": realized_pl_for_coin
        })

    summary = pd.DataFrame(summary_rows)
    for col in ["Coins Purchased", "Total Cost", "Average Price (USD)", "Realized P/L"]:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors='coerce').fillna(0.0)

    internal_summary_columns = [
        "Coin", "Coins Purchased", "Average Price (USD)", "Total Cost", "Realized P/L"
    ]
    summary = summary[[col for col in internal_summary_columns if col in summary.columns]]
    return summary


def format_summary_display(summary_df_to_format):
    summary = summary_df_to_format.copy()

    def format_currency(value, decimals=2):
        if pd.isnull(value) or not isinstance(value, (int, float)):
            return "-"
        return f"${float(value):,.{decimals}f}"

    def format_sats(value):
        if pd.isnull(value) or not isinstance(value, (int, float)):
            return "-"
        return f"{int(value):,}"

    def format_coins(value, decimals=8):
        if pd.isnull(value) or not isinstance(value, (int, float)):
            return "-"
        return f"{float(value):,.{decimals}f}"

    if "Latest Price (USD) (raw)" in summary.columns:
        summary["Latest Price (USD)"] = summary["Latest Price (USD) (raw)"].apply(
            lambda x: format_currency(x, decimals=4 if (x and abs(x) < 1) else 2)
        )
    if "Current Value (USD) (raw)" in summary.columns:
        summary["Current Value (USD)"] = summary["Current Value (USD) (raw)"].apply(format_currency)
    if "Unrealized P/L (raw)" in summary.columns:
        summary["Unrealized P/L"] = summary["Unrealized P/L (raw)"].apply(format_currency)

    # --- New Formatting for Unrealized P/L % ---
    if "Unrealized P/L % (raw)" in summary.columns:
        summary["Unrealized P/L %"] = summary["Unrealized P/L % (raw)"].apply(
            lambda x: (
                f"{float(x):,.2f}%" if pd.notnull(x) and np.isfinite(x)
                else ("+∞%" if x == np.inf else ("-∞%" if x == -np.inf else "N/A"))
            )
        )
    # --- End New Formatting ---

    if "Current Value (Sats) (raw)" in summary.columns:
        summary["Current Value (Sats)"] = summary["Current Value (Sats) (raw)"].apply(format_sats)
    for col in ["Average Price (USD)", "Total Cost", "Realized P/L"]:
        if col in summary.columns:
            summary[col] = summary[col].apply(format_currency)
    if "Coins Purchased" in summary.columns:
        summary["Coins Purchased"] = summary["Coins Purchased"].apply(
            lambda x: format_coins(x, decimals=8)
        )
    if "Sell Target (USD) (raw)" in summary.columns:
        summary["Sell Target (USD)"] = summary["Sell Target (USD) (raw)"].apply(format_currency)
    if "Sell Target (Sats) (raw)" in summary.columns:
        summary["Sell Target (Sats)"] = summary["Sell Target (Sats) (raw)"].apply(format_sats)
    if "Profit at Sell Target (USD) (raw)" in summary.columns:
        summary["Profit at Sell Target (USD)"] = summary[
            "Profit at Sell Target (USD) (raw)"
        ].apply(format_currency)

    raw_cols_to_drop = [c for c in summary.columns if c.endswith("(raw)")]
    if raw_cols_to_drop:
        summary = summary.drop(columns=raw_cols_to_drop, errors='ignore')
    return summary


def bitcoin_first_then_alpha(df_to_sort):
    if df_to_sort.empty or "Coin" not in df_to_sort.columns:
        return df_to_sort
    order = []
    processed_coins = set()
    dfs_to_concat = []
    if "US Dollars" in df_to_sort["Coin"].values:
        order.append("US Dollars")
    if "Bitcoin" in df_to_sort["Coin"].values:
        order.append("Bitcoin")
    unique_coins = df_to_sort["Coin"].unique()
    rest = [c for c in sorted(unique_coins) if c not in order]
    for coin_name in order + rest:
        if coin_name in unique_coins and coin_name not in processed_coins:
            dfs_to_concat.append(df_to_sort[df_to_sort["Coin"] == coin_name])
            processed_coins.add(coin_name)
    if dfs_to_concat:
        return pd.concat(dfs_to_concat, ignore_index=True)
    return pd.DataFrame(columns=df_to_sort.columns)


# --- Calculation Helper Functions ---
def calc_sell_target_sats(row, btc_price_usd_val):
    sell_target_usd_raw = row.get("Sell Target (USD) (raw)")
    if (
        pd.notnull(sell_target_usd_raw)
        and pd.notnull(btc_price_usd_val)
        and isinstance(btc_price_usd_val, (int, float))
        and btc_price_usd_val > 0
    ):
        try:
            return int((float(sell_target_usd_raw) / float(btc_price_usd_val)) * 100_000_000)
        except (ValueError, TypeError):
            return None
    return None


def calc_current_value(row):
    price = row.get("Latest Price (USD) (raw)")
    coins_held = row.get("Coins Purchased")
    if (
        pd.notnull(price)
        and pd.notnull(coins_held)
        and isinstance(price, (int, float))
        and isinstance(coins_held, (int, float))
    ):
        try:
            return float(coins_held) * float(price)
        except (ValueError, TypeError):
            return None
    return None


def calc_current_value_sats(row, btc_price_usd_val):
    usd_value = calc_current_value(row)
    if (
        pd.notnull(usd_value)
        and pd.notnull(btc_price_usd_val)
        and isinstance(btc_price_usd_val, (int, float))
        and btc_price_usd_val > 0
    ):
        try:
            return int((float(usd_value) / float(btc_price_usd_val)) * 100_000_000)
        except (ValueError, TypeError, OverflowError):
            return None
    return None


def calc_profit(row):
    coins_held = row.get("Coins Purchased")
    sell_target = row.get("Sell Target (USD) (raw)")
    total_cost_of_held = row.get("Total Cost")
    if (
        pd.notnull(sell_target)
        and pd.notnull(coins_held)
        and pd.notnull(total_cost_of_held)
        and isinstance(sell_target, (int, float))
        and isinstance(coins_held, (int, float))
        and isinstance(total_cost_of_held, (int, float))
    ):
        try:
            if abs(float(coins_held)) < 1e-8:
                return 0.0
            return float(sell_target) * float(coins_held) - float(total_cost_of_held)
        except (ValueError, TypeError):
            return None
    return None


def calc_unrealized_pl(row):
    if row.get("Coin") == "US Dollars":
        return 0.0
    current_value_raw = row.get("Current Value (USD) (raw)")
    total_cost_numeric = row.get("Total Cost")
    coins_held_numeric = row.get("Coins Purchased")
    if (
        pd.isnull(current_value_raw)
        or pd.isnull(total_cost_numeric)
        or pd.isnull(coins_held_numeric)
        or not isinstance(current_value_raw, (int, float))
        or not isinstance(total_cost_numeric, (int, float))
        or not isinstance(coins_held_numeric, (int, float))
    ):
        return None
    try:
        if abs(float(coins_held_numeric)) < 1e-8:
            return 0.0
        return float(current_value_raw) - float(total_cost_numeric)
    except (ValueError, TypeError):
        return None


# --- Main Display Functions ---
def display_portfolio_summary(portfolio_entries_list):
    transactions_df = build_transactions_df(portfolio_entries_list)
    summary_df = build_summary_df(transactions_df)  # This df has raw 'Total Cost'

    user_coins = list(summary_df["Coin"].unique())
    user_symbols = [
        transactions_df[transactions_df["Coin"] == cn]["Symbol"].iloc[0]
        if not transactions_df[transactions_df["Coin"] == cn].empty
        else cn[:4].upper()
        for cn in user_coins
    ]

    cg_coins_fetch = [c for c in user_coins if c != "US Dollars"]
    cg_symbols_fetch = [s for c, s in zip(user_coins, user_symbols) if c != "US Dollars"]
    if "Bitcoin" not in cg_coins_fetch and "Bitcoin" in COINGECKO_ID_OVERRIDES:
        cg_coins_fetch.append("Bitcoin")
        cg_symbols_fetch.append("BTC")

    latest_prices, id_map_debug, was_rate_limited_main = fetch_latest_prices(
        cg_coins_fetch, cg_symbols_fetch, name_to_id_global, symbol_to_id_global
    )  # Ensure fetch_latest_prices passes COINGECKO_ID_OVERRIDES to lookup_coingecko_id

    summary_df["Latest Price (USD) (raw)"] = summary_df["Coin"].map(
        lambda c: latest_prices.get(c) if c != "US Dollars" else 1.0
    )

    btc_price_usd = latest_prices.get("Bitcoin")
    if btc_price_usd is None and not was_rate_limited_main and "Bitcoin" in COINGECKO_ID_OVERRIDES:
        st.sidebar.info("Attempting to fetch Bitcoin price separately...")
        # Ensure this call to fetch_latest_prices also correctly uses the modified lookup_coingecko_id
        temp_btc_price_data, _, was_rate_limited_btc = fetch_latest_prices(
            ["Bitcoin"], ["BTC"], name_to_id_global, symbol_to_id_global
        )
        btc_price_usd = temp_btc_price_data.get("Bitcoin")
        if was_rate_limited_btc:
            st.sidebar.warning("Could not fetch Bitcoin price separately due to API rate limits.")
    elif btc_price_usd is None and was_rate_limited_main:
        st.sidebar.warning("Bitcoin price not determined due to earlier API rate limit.")
    if btc_price_usd is None:
        btc_price_usd = 0.0
        if not was_rate_limited_main:
            st.sidebar.warning("Bitcoin price could not be determined. Calculations involving BTC value may be incorrect.")

    summary_df["CoinGecko ID"] = summary_df["Coin"].map(
        lambda c: id_map_debug.get(c, "-") if c != "US Dollars" else "us-dollars"
    )
    sell_targets = get_sell_targets()
    summary_df["Sell Target (USD) (raw)"] = summary_df["Coin"].map(lambda c: sell_targets.get(c))
    summary_df["Sell Target (Sats) (raw)"] = summary_df.apply(lambda row: calc_sell_target_sats(row, btc_price_usd), axis=1)
    summary_df["Current Value (USD) (raw)"] = summary_df.apply(calc_current_value, axis=1)
    summary_df["Current Value (Sats) (raw)"] = summary_df.apply(lambda row: calc_current_value_sats(row, btc_price_usd), axis=1)
    summary_df["Profit at Sell Target (USD) (raw)"] = summary_df.apply(calc_profit, axis=1)
    summary_df["Unrealized P/L (raw)"] = summary_df.apply(calc_unrealized_pl, axis=1)

    summary_df['Total Cost'] = pd.to_numeric(summary_df['Total Cost'], errors='coerce')
    summary_df['Unrealized P/L (raw)'] = pd.to_numeric(summary_df['Unrealized P/L (raw)'], errors='coerce')

    def calculate_unrealized_pl_percentage(row):
        cost = row['Total Cost']
        pl = row['Unrealized P/L (raw)']
        if pd.isna(cost) or pd.isna(pl) or row['Coin'] == 'US Dollars':
            return None
        if abs(cost) < 0.01:
            if pl > 0.01:
                return np.inf
            elif pl < -0.01:
                return -np.inf
            else:
                return 0.0
        return (pl / cost) * 100.0

    summary_df["Unrealized P/L % (raw)"] = summary_df.apply(calculate_unrealized_pl_percentage, axis=1)

    raw_data_for_styling_lookup = pd.DataFrame()
    if "Coin" in summary_df.columns:
        if summary_df["Coin"].nunique() == len(summary_df):
            raw_data_for_styling_lookup = summary_df.set_index("Coin")
        else:
            raw_data_for_styling_lookup = summary_df.copy()
    else:
        raw_data_for_styling_lookup = summary_df.copy()

    summary_for_plotting = summary_df.copy()
    summary_for_table_display = format_summary_display(summary_df.copy())
    summary_for_table_display = bitcoin_first_then_alpha(summary_for_table_display)

    # Calculate current total portfolio value (this will be recorded)
    total_current_value_all_coins = 0.0
    if "Current Value (USD) (raw)" in summary_for_plotting.columns:
        total_current_value_all_coins = pd.to_numeric(
            summary_for_plotting["Current Value (USD) (raw)"], errors='coerce'
        ).fillna(0.0).sum()

    # --- Record today's portfolio value to the database ---
    if total_current_value_all_coins is not None:
        today_date_obj = datetime.now().date()
        record_daily_portfolio_value(today_date_obj, total_current_value_all_coins)
    # --- End recording logic ---

    # Other aggregate metrics for display
    total_realized_pl_numeric = 0.0
    if "Realized P/L" in summary_for_plotting.columns:  # Assuming "Realized P/L" is already numeric from build_summary_df
        total_realized_pl_numeric = pd.to_numeric(summary_for_plotting["Realized P/L"], errors='coerce').fillna(0.0).sum()

    total_unrealized_pl = 0.0
    if "Unrealized P/L (raw)" in summary_for_plotting.columns:
        total_unrealized_pl = pd.to_numeric(summary_for_plotting["Unrealized P/L (raw)"], errors='coerce').fillna(0.0).sum()

    st.subheader("📈 Portfolio Overview")
    metric_cols = st.columns(3)
    tile_style = ("padding: 15px; border: 1px solid #ddd; border-radius: 8px; text-align: center; height: 100px; "
                  "display: flex; flex-direction: column; justify-content: center;")
    label_style = "font-size: 0.9em; color: #555; margin-bottom: 5px;"
    value_style_base = "font-size: 1.6em; font-weight: bold;"

    with metric_cols[0]:
        st.markdown(
            f"<div style='{tile_style}'><span style='{label_style}'>Total Portfolio Value</span>"
            f"<span style='{value_style_base}'>${total_current_value_all_coins:,.2f}</span></div>",
            unsafe_allow_html=True
        )
    with metric_cols[1]:
        unrealized_pl_color = "#555"
        if total_unrealized_pl > 0.01:
            unrealized_pl_color = "green"
        elif total_unrealized_pl < -0.01:
            unrealized_pl_color = "red"
        st.markdown(
            f"<div style='{tile_style}'><span style='{label_style}'>Total Unrealized P/L</span>"
            f"<span style='{value_style_base} color: {unrealized_pl_color};'>${total_unrealized_pl:,.2f}</span></div>",
            unsafe_allow_html=True
        )
    with metric_cols[2]:
        realized_pl_color = "#555"
        if total_realized_pl_numeric > 0.01:
            realized_pl_color = "green"
        elif total_realized_pl_numeric < -0.01:
            realized_pl_color = "red"
        st.markdown(
            f"<div style='{tile_style}'><span style='{label_style}'>Total Realized P/L</span>"
            f"<span style='{value_style_base} color: {realized_pl_color};'>${total_realized_pl_numeric:,.2f}</span></div>",
            unsafe_allow_html=True
        )

    st.markdown("#### <span style='font-size: 1.1em;'>🚀 Top Movers by Unrealized P/L %</span>", unsafe_allow_html=True)
    mover_cols = st.columns(2)
    movers_df = summary_df[
        (summary_df['Coin'] != 'US Dollars') &
        summary_df['Unrealized P/L % (raw)'].notna() &
        np.isfinite(summary_df['Unrealized P/L % (raw)']) &
        (summary_df['Total Cost'].notna() & (abs(summary_df['Total Cost']) > 0.01))
    ].copy()
    top_gainer_info, top_loser_info = None, None
    if not movers_df.empty:
        movers_df_sorted = movers_df.sort_values(by="Unrealized P/L % (raw)", ascending=False)
        if not movers_df_sorted.empty and movers_df_sorted.iloc[0]["Unrealized P/L % (raw)"] > 0:
            gainer_row = movers_df_sorted.iloc[0]
            top_gainer_info = {"coin": gainer_row["Coin"], "change": gainer_row["Unrealized P/L % (raw)"]}
        losers_only_df = movers_df_sorted[movers_df_sorted["Unrealized P/L % (raw)"] < 0]
        if not losers_only_df.empty:
            loser_row = losers_only_df.iloc[-1]
            top_loser_info = {"coin": loser_row["Coin"], "change": loser_row["Unrealized P/L % (raw)"]}

    tile_style_small = ("padding: 10px; border: 1px solid #ddd; border-radius: 8px; text-align: center; height: 110px; "
                        "display: flex; flex-direction: column; justify-content: center;")
    label_style_small = "font-size: 0.85em; color: #555; margin-bottom: 4px;"
    coin_name_style_small = (
        "font-size: 1.0em; font-weight: normal; color: #333; margin-bottom: 4px; "
        "white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"
    )
    value_style_small_base_mover = "font-size: 1.4em; font-weight: bold;"  # Renamed to avoid conflict

    with mover_cols[0]:
        if top_gainer_info:
            st.markdown(
                f"""<div style=\"{tile_style_small}\">
                <span style='{label_style_small}'>🌟 Top Gainer</span>
                <span style='{coin_name_style_small}' title='{top_gainer_info['coin']}'>{top_gainer_info['coin']}</span>
                <span style='{value_style_small_base_mover} color: green;'>+{top_gainer_info['change']:.2f}%</span>
                </div>""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='{tile_style_small}'><span style='{label_style_small}'>🌟 Top Gainer</span>"
                f"<span style='{value_style_small_base_mover}'>N/A</span></div>",
                unsafe_allow_html=True
            )
    with mover_cols[1]:
        if top_loser_info:
            st.markdown(
                f"""<div style=\"{tile_style_small}\">
                <span style='{label_style_small}'>🔻 Top Loser</span>
                <span style='{coin_name_style_small}' title='{top_loser_info['coin']}'>{top_loser_info['coin']}</span>
                <span style='{value_style_small_base_mover} color: red;'>{top_loser_info['change']:.2f}%</span>
                </div>""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='{tile_style_small}'><span style='{label_style_small}'>🔻 Top Loser</span>"
                f"<span style='{value_style_small_base_mover}'>N/A</span></div>",
                unsafe_allow_html=True
            )
    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("💰 Portfolio Summary Details")
    if "CoinGecko ID" in summary_for_table_display.columns:
        summary_for_table_display = summary_for_table_display.drop(columns=["CoinGecko ID"])
    desired_final_order = [
        "Coin", "Coins Purchased", "Latest Price (USD)", "Average Price (USD)", "Total Cost",
        "Current Value (USD)", "Current Value (Sats)", "Unrealized P/L", "Unrealized P/L %",
        "Realized P/L", "Sell Target (USD)", "Sell Target (Sats)", "Profit at Sell Target (USD)"
    ]
    ordered_summary_cols = [col for col in desired_final_order if col in summary_for_table_display.columns]
    remaining_display_cols = [col for col in summary_for_table_display.columns if col not in ordered_summary_cols]
    summary_for_table_display = summary_for_table_display[ordered_summary_cols + remaining_display_cols]
    summary_for_table_display.index = pd.RangeIndex(start=1, stop=len(summary_for_table_display) + 1, step=1)
    summary_for_table_display.index.name = "#"

    def p_l_color_styler(val_str_cell, column_name=""):
        if isinstance(val_str_cell, str):
            if val_str_cell.startswith("$"):
                try:
                    num_val = float(val_str_cell.replace("$", "").replace(",", ""))
                    if num_val > 0.01:
                        return "color: green;"
                    elif num_val < -0.01:
                        return "color: red;"
                except ValueError:
                    pass
            elif val_str_cell.endswith("%") and column_name == "Unrealized P/L %":
                try:
                    if val_str_cell == "+∞%":
                        return "color: green;"
                    if val_str_cell == "-∞%":
                        return "color: red;"
                    if val_str_cell == "N/A":
                        return ""
                    num_val = float(val_str_cell.replace("%", "").replace(",", ""))
                    if num_val > 0.01:
                        return "color: green;"
                    elif num_val < -0.01:
                        return "color: red;"
                except ValueError:
                    pass
        return ""

    def apply_sell_target_styles_lookup(row_from_display_df, btc_price_usd_val_func):
        styles = pd.Series('', index=row_from_display_df.index)
        coin_name = row_from_display_df.get("Coin")
        if not coin_name or coin_name == "US Dollars":
            return styles
        raw_coin_data = None
        if not raw_data_for_styling_lookup.empty:
            if raw_data_for_styling_lookup.index.name == "Coin" and coin_name in raw_data_for_styling_lookup.index:
                raw_coin_data = raw_data_for_styling_lookup.loc[coin_name]
            else:
                temp_lookup = raw_data_for_styling_lookup[raw_data_for_styling_lookup["Coin"] == coin_name]
                if not temp_lookup.empty:
                    raw_coin_data = temp_lookup.iloc[0]
        if raw_coin_data is None:
            return styles
        sell_target_usd_raw = raw_coin_data.get("Sell Target (USD) (raw)")
        latest_price_usd_raw = raw_coin_data.get("Latest Price (USD) (raw)")
        if pd.notnull(sell_target_usd_raw) and pd.notnull(latest_price_usd_raw):
            try:
                sell_target_usd_raw_float = float(sell_target_usd_raw)
                latest_price_usd_raw_float = float(latest_price_usd_raw)
                if sell_target_usd_raw_float > 0:
                    if latest_price_usd_raw_float >= sell_target_usd_raw_float:
                        styles['Sell Target (USD)'] = 'background-color: darkgreen; color: white;'
                    else:
                        proximity_usd = latest_price_usd_raw_float / sell_target_usd_raw_float
                        if proximity_usd >= 0.90:
                            styles['Sell Target (USD)'] = 'background-color: #A6FFA6; color: black;'
                        elif proximity_usd >= 0.75:
                            styles['Sell Target (USD)'] = 'background-color: #FFFFCC; color: black;'
            except ValueError:
                pass
        sell_target_sats_raw = raw_coin_data.get("Sell Target (Sats) (raw)")
        latest_price_sats_raw = None
        if (
            pd.notnull(latest_price_usd_raw)
            and pd.notnull(btc_price_usd_val_func)
            and isinstance(btc_price_usd_val_func, (float, int))
            and btc_price_usd_val_func > 0
        ):
            try:
                latest_price_sats_raw = int(
                    (float(latest_price_usd_raw) / float(btc_price_usd_val_func)) * 100_000_000
                )
            except (ValueError, TypeError, OverflowError):
                pass
        if pd.notnull(sell_target_sats_raw) and pd.notnull(latest_price_sats_raw):
            try:
                sell_target_sats_raw_int = int(sell_target_sats_raw)
                latest_price_sats_raw_int = int(latest_price_sats_raw)
                if sell_target_sats_raw_int > 0:
                    if latest_price_sats_raw_int >= sell_target_sats_raw_int:
                        styles['Sell Target (Sats)'] = 'background-color: darkgreen; color: white;'
                    else:
                        proximity_sats = latest_price_sats_raw_int / sell_target_sats_raw_int
                        if proximity_sats >= 0.90:
                            styles['Sell Target (Sats)'] = 'background-color: #A6FFA6; color: black;'
                        elif proximity_sats >= 0.75:
                            styles['Sell Target (Sats)'] = 'background-color: #FFFFCC; color: black;'
            except ValueError:
                pass
        return styles

    styler_obj = summary_for_table_display.style
    pl_dollar_columns_to_style = [col for col in ["Unrealized P/L", "Realized P/L"] if col in summary_for_table_display.columns]
    if pl_dollar_columns_to_style:
        styler_obj = styler_obj.map(lambda x: p_l_color_styler(x), subset=pl_dollar_columns_to_style)
    if "Unrealized P/L %" in summary_for_table_display.columns:
        styler_obj = styler_obj.map(
            lambda x: p_l_color_styler(x, column_name="Unrealized P/L %"),
            subset=["Unrealized P/L %"]
        )
    current_btc_price_for_styling = btc_price_usd if isinstance(btc_price_usd, (float, int)) else 0.0
    if not raw_data_for_styling_lookup.empty and not summary_for_table_display.empty:
        styler_obj = styler_obj.apply(lambda row: apply_sell_target_styles_lookup(row, current_btc_price_for_styling), axis=1)
    st.dataframe(styler_obj, hide_index=False, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Portfolio Visualizations")
    
    # --- Allocation Pie Chart ---
    plot_df_coins_only = summary_for_plotting[summary_for_plotting['Coin'] != 'US Dollars'].copy()
    plot_df_allocation = plot_df_coins_only[
        plot_df_coins_only['Current Value (USD) (raw)'].notna() &
        (pd.to_numeric(plot_df_coins_only['Current Value (USD) (raw)'], errors='coerce').fillna(0) > 0.01)
    ].copy()
    if not plot_df_allocation.empty:
        st.markdown("#### Portfolio Allocation (by Current Value)")
        fig_alloc = px.pie(plot_df_allocation, names='Coin', values='Current Value (USD) (raw)',
                           title='Portfolio Allocation by Current Value', hole=0.3)
        fig_alloc.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_alloc, use_container_width=True)
    else:
        st.markdown("_(Not enough data for Portfolio Allocation chart.)_")

    # --- Investment Performance Bar Chart ---
    # ... (Existing code for Investment Performance chart with log_y_perf toggle) ...
    plot_df_perf_source = plot_df_coins_only[
        plot_df_coins_only['Total Cost'].notna() &
        plot_df_coins_only['Current Value (USD) (raw)'].notna() &
        (pd.to_numeric(plot_df_coins_only['Total Cost'], errors='coerce').fillna(0) > 0)
    ].copy()
    if not plot_df_perf_source.empty:
        st.markdown("#### Investment Performance (Cost vs. Value)")
        log_y_perf = st.checkbox("Logarithmic Y-axis", key="log_y_perf_toggle_main", value=True)
        try:
            df_to_melt = plot_df_perf_source[[
                'Coin', 'Total Cost', 'Current Value (USD) (raw)']].copy()
            df_to_melt['Current Value (USD) (raw)'] = pd.to_numeric(
                df_to_melt['Current Value (USD) (raw)'], errors='coerce')
            df_to_melt.dropna(subset=['Current Value (USD) (raw)'], inplace=True)
            sorted_coin_order = df_to_melt.sort_values(
                by='Current Value (USD) (raw)', ascending=False)['Coin'].tolist()
            plot_df_perf_melted = pd.melt(
                df_to_melt, id_vars=['Coin'],
                value_vars=['Total Cost', 'Current Value (USD) (raw)'],
                var_name='Metric', value_name='Amount (USD)'
            )
            if not plot_df_perf_melted.empty:
                plot_df_perf_melted['Coin'] = pd.Categorical(
                    plot_df_perf_melted['Coin'], categories=sorted_coin_order, ordered=True)
                plot_df_perf_melted = plot_df_perf_melted.sort_values(by=['Coin', 'Metric'])
            plot_df_perf_melted['Metric'] = plot_df_perf_melted['Metric'].replace(
                {'Current Value (USD) (raw)': 'Current Value (USD)'}
            )
            perf_chart_title = 'Cost Basis vs. Current Market Value'
            perf_yaxis_title = 'Amount (USD)'
            if log_y_perf:
                perf_yaxis_title += ' (Log Scale)'
            fig_perf = px.bar(
                plot_df_perf_melted, x='Coin', y='Amount (USD)', color='Metric', barmode='group',
                title=perf_chart_title,
                labels={'Amount (USD)': 'Amount (USD)', 'Coin': 'Cryptocurrency', 'Metric': 'Performance Metric'},
                log_y=log_y_perf
            )
            fig_perf.update_layout(yaxis_title=perf_yaxis_title)
            st.plotly_chart(fig_perf, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating Investment Performance chart: {e}")
    else:
        st.markdown(
            "_(Not enough data for Investment Performance chart. Ensure assets have a cost basis > 0 and current value.)_"
        )

    # --- Unrealized P/L Bar Chart ---
    plot_df_unrealized = plot_df_coins_only[plot_df_coins_only['Unrealized P/L (raw)'].notna()].copy()
    if not plot_df_unrealized.empty:
        st.markdown("#### Unrealized P/L by Coin")
        log_y_unrealized = st.checkbox(
            "Logarithmic Y-axis",
            key="log_y_unrealized_toggle_main",
            value=False,
            help=(
                "Note: Log scale typically applies to positive values. "
                "Negative or zero P/L might not display as expected on a log scale."
            )
        )
        plot_df_unrealized['Unrealized P/L (raw)'] = pd.to_numeric(
            plot_df_unrealized['Unrealized P/L (raw)'], errors='coerce'
        )
        plot_df_unrealized.dropna(subset=['Unrealized P/L (raw)'], inplace=True)
        plot_df_unrealized = plot_df_unrealized.sort_values(by='Unrealized P/L (raw)', ascending=False)
        unrealized_chart_title = 'Unrealized Profit/Loss by Coin'
        unrealized_yaxis_title = 'Unrealized P/L (USD)'
        if log_y_unrealized:
            unrealized_yaxis_title += ' (Log Scale - Positive Values)'
        fig_unrealized = px.bar(
            plot_df_unrealized, x='Coin', y='Unrealized P/L (raw)', title=unrealized_chart_title,
            labels={'Unrealized P/L (raw)': 'Unrealized P/L (USD)'},
            color='Unrealized P/L (raw)', color_continuous_scale=px.colors.diverging.RdYlGn,
            color_continuous_midpoint=0, log_y=log_y_unrealized
        )
        fig_unrealized.update_layout(xaxis_title="Coin", yaxis_title=unrealized_yaxis_title)
        st.plotly_chart(fig_unrealized, use_container_width=True)
    else:
        st.markdown("_(Not enough data for Unrealized P/L chart.)_")
    # --- Sell Targets vs. Current Price Chart ---
    # ... (Existing code for Sell Targets chart with log_y_targets toggle) ...
    st.markdown("---")
    st.markdown("#### Sell Targets vs. Current Price")
    plot_df_targets_source = plot_df_coins_only[
        plot_df_coins_only['Sell Target (USD) (raw)'].notna() &
        plot_df_coins_only['Latest Price (USD) (raw)'].notna() &
        (pd.to_numeric(plot_df_coins_only['Sell Target (USD) (raw)'], errors='coerce').fillna(0) > 0) &
        (pd.to_numeric(plot_df_coins_only['Latest Price (USD) (raw)'], errors='coerce').fillna(0) > 0)
    ].copy()
    if not plot_df_targets_source.empty:
        log_y_targets = st.checkbox("Logarithmic Y-axis", key="log_y_targets_toggle_main", value=True)
        try:
            plot_df_targets_source['Latest Price (USD) (raw)'] = pd.to_numeric(
                plot_df_targets_source['Latest Price (USD) (raw)'], errors='coerce'
            )
            plot_df_targets_source['Sell Target (USD) (raw)'] = pd.to_numeric(
                plot_df_targets_source['Sell Target (USD) (raw)'], errors='coerce'
            )
            plot_df_targets_source.dropna(
                subset=['Latest Price (USD) (raw)', 'Sell Target (USD) (raw)'], inplace=True
            )
            plot_df_targets_source = plot_df_targets_source[
                plot_df_targets_source['Sell Target (USD) (raw)'] != 0
            ]
            plot_df_targets_source['Target Proximity Ratio'] = (
                plot_df_targets_source['Latest Price (USD) (raw)'] /
                plot_df_targets_source['Sell Target (USD) (raw)']
            )
            sorted_target_coin_order = plot_df_targets_source.sort_values(
                by='Target Proximity Ratio', ascending=False
            )['Coin'].tolist()
            df_to_melt_targets = plot_df_targets_source[
                ['Coin', 'Sell Target (USD) (raw)', 'Latest Price (USD) (raw)']
            ].copy()
            df_to_melt_targets.rename(
                columns={
                    'Sell Target (USD) (raw)': 'Sell Target',
                    'Latest Price (USD) (raw)': 'Current Price'
                }, inplace=True
            )
            plot_df_targets_melted = pd.melt(
                df_to_melt_targets, id_vars=['Coin'],
                value_vars=['Sell Target', 'Current Price'],
                var_name='Price Type', value_name='Price (USD)'
            )
            if not plot_df_targets_melted.empty:
                plot_df_targets_melted['Coin'] = pd.Categorical(
                    plot_df_targets_melted['Coin'],
                    categories=sorted_target_coin_order, ordered=True
                )
                plot_df_targets_melted = plot_df_targets_melted.sort_values(by=['Coin', 'Price Type'])
            targets_chart_title = 'Sell Targets vs. Current Market Price (Sorted)'
            targets_yaxis_title = 'Price (USD)'
            if log_y_targets:
                targets_yaxis_title += ' (Log Scale)'
            else:
                targets_yaxis_title += ' (Linear Scale)'
            fig_targets = px.bar(
                plot_df_targets_melted, x='Coin', y='Price (USD)', color='Price Type',
                barmode='group', title=targets_chart_title,
                labels={
                    'Price (USD)': 'Price (USD)',
                    'Price Type': 'Metric',
                    'Coin': 'Cryptocurrency'
                },
                color_discrete_map={'Sell Target': 'orangered', 'Current Price': 'dodgerblue'},
                log_y=log_y_targets
            )
            fig_targets.update_layout(yaxis_title=targets_yaxis_title)
            st.plotly_chart(fig_targets, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating Sell Targets vs. Current Price chart: {e}")
    else:
        st.markdown(
            "_(Not enough data for Sell Targets vs. Current Price chart. Ensure assets have positive sell targets and current prices.)_"
        )

    # --- MODIFIED: Total Portfolio Value Trend (from Database) ---
    st.markdown("---")
    st.markdown("#### 📈 Total Portfolio Value Trend (Last Year - Recorded Daily)")

    current_date_for_trend = datetime.now().date()
    start_date_for_trend = (pd.Timestamp(current_date_for_trend) - pd.DateOffset(years=1)).date()

    historical_db_df = get_trend_chart_data_from_db(start_date_for_trend, current_date_for_trend)

    if not historical_db_df.empty and len(historical_db_df) > 1:
        if historical_db_df["Total Portfolio Value"].sum() > 0 or historical_db_df["Total Portfolio Value"].nunique() > 1:
            fig_trend = px.line(
                historical_db_df, x='date', y='Total Portfolio Value',
                title='Portfolio Value Over Last Year (Recorded)', markers=True,
                labels={"date": "Date", "Total Portfolio Value": "Portfolio Value (USD)"}
            )
            fig_trend.update_layout(xaxis_title="Date", yaxis_title="Total Portfolio Value (USD)")
            st.plotly_chart(fig_trend, use_container_width=True)
        elif not historical_db_df.empty:
            st.markdown("_(Trend chart data from DB available, but values are zero or constant. Plotting anyway.)_")
            fig_trend = px.line(
                historical_db_df, x='date', y='Total Portfolio Value',
                title='Portfolio Value Over Last Year (Recorded)', markers=True,
                labels={"date": "Date", "Total Portfolio Value": "Portfolio Value (USD)"}
            )
            fig_trend.update_layout(xaxis_title="Date", yaxis_title="Total Portfolio Value (USD)")
            st.plotly_chart(fig_trend, use_container_width=True)
    elif not historical_db_df.empty and len(historical_db_df) == 1:
        st.markdown(
            f"_(Only one data point recorded in DB for portfolio value trend on "
            f"{historical_db_df['date'].iloc[0].strftime('%Y-%m-%d')}: "
            f"${historical_db_df['Total Portfolio Value'].iloc[0]:,.2f}. More data needed for a trend line.)_"
        )
    else:
        st.markdown(
            "_(Not enough data recorded in the database yet to generate portfolio value trend chart. "
            "Data will be recorded each day you view the portfolio.)_"
        )
    # --- End Trend Chart from DB ---

    return transactions_df, summary_for_table_display, btc_price_usd, id_map_debug, latest_prices


def display_transactions_table(transactions_df_to_display):
    st.subheader("🧾 All Transactions")
    if transactions_df_to_display.empty:
        st.write("No transactions yet.")
        return
    df_display = transactions_df_to_display.copy()
    if "Purchase Price (USD)" in df_display.columns:
        df_display["Purchase Price (USD)"] = df_display["Purchase Price (USD)"].apply(
            lambda x: f"${x:,.2f}" if pd.notnull(x) else "-"
        )
    if "Total Cost" in df_display.columns:
        df_display["Total Cost"] = df_display["Total Cost"].apply(
            lambda x: f"${x:,.2f}" if pd.notnull(x) else "-"
        )
    if "Coins Purchased" in df_display.columns:
        df_display["Coins Purchased"] = df_display["Coins Purchased"].apply(
            lambda x: f"{x:,.8f}" if pd.notnull(x) else "-"
        )

    display_cols = [col for col in df_display.columns if col != "ID"]
    header_cols = st.columns(list(map(lambda x: 1.5, display_cols)) + [0.5])
    for i, col_name in enumerate(display_cols):
        header_cols[i].markdown(f"**{col_name}**")
    header_cols[-1].markdown("**Del**")

    for idx, row_data in transactions_df_to_display.iterrows():
        form_key = f"delete_form_{row_data['ID']}"
        confirm_key = f"confirm_delete_{row_data['ID']}"
        with st.form(key=form_key, clear_on_submit=True):
            form_cols = st.columns(list(map(lambda x: 1.5, display_cols)) + [0.5])
            for i, col_name in enumerate(display_cols):
                form_cols[i].write(str(df_display.loc[idx, col_name]))
            delete_clicked = form_cols[-1].form_submit_button(
                "🗑️", help="Delete this transaction", use_container_width=True
            )
            if delete_clicked:
                st.session_state[confirm_key] = True

        if st.session_state.get(confirm_key, False):
            confirm_cols = st.columns([2, 1, 1])
            confirm_cols[0].warning(
                f"Are you sure you want to delete transaction ID {row_data['ID']} ({row_data['Coin']})?"
            )
            yes = confirm_cols[1].button(
                "Yes, Delete", key=f"yes_delete_{row_data['ID']}"
            )
            cancel = confirm_cols[2].button(
                "Cancel", key=f"cancel_delete_{row_data['ID']}"
            )
            if yes:
                entry_to_delete = db_session.query(PortfolioEntry).filter(
                    PortfolioEntry.id == row_data['ID']
                ).first()
                if entry_to_delete:
                    db_session.delete(entry_to_delete)
                    db_session.commit()
                    st.success(
                        f"Transaction ID {row_data['ID']} ({entry_to_delete.coin}) deleted."
                    )
                    st.session_state.pop(confirm_key, None)
                    st.rerun()
                else:
                    st.error(
                        f"Could not find transaction ID {row_data['ID']} to delete."
                    )
                    st.session_state.pop(confirm_key, None)
            elif cancel:
                st.session_state.pop(confirm_key, None)


def display_sell_targets_ui(portfolio_summary_df, btc_price_usd_val):
    st.header("🎯 Set Your Sell Targets")
    if portfolio_summary_df.empty:
        st.write("Portfolio summary is empty.")
        return
    coins_in_portfolio = sorted([
        coin for coin in portfolio_summary_df["Coin"].unique() if coin != "US Dollars"
    ])
    if not coins_in_portfolio:
        st.write("No non-USD coins in portfolio to set targets for.")
        return

    current_targets = get_sell_targets()
    cols_st = st.columns([2, 2, 2, 1])
    sel_coin = cols_st[0].selectbox("Select Coin", coins_in_portfolio, key="sell_target_coin_sel")

    avg_p_str = "-"
    if sel_coin and "Average Price (USD)" in portfolio_summary_df.columns:
        avg_p_series = portfolio_summary_df[portfolio_summary_df["Coin"] == sel_coin]["Average Price (USD)"]
        if not avg_p_series.empty:
            avg_p_str = avg_p_series.iloc[0]
    help_avg = f"Avg. Cost: {avg_p_str}"

    curr_target_usd = float(current_targets.get(sel_coin, 0.0))
    target_usd_in = cols_st[1].number_input(
        "Target Price (USD)",
        min_value=0.0,
        value=curr_target_usd,
        step=0.01,
        format="%.2f",
        key=f"sell_target_usd_in_{sel_coin}",
        help=help_avg
    )

    sats_disp = "-"
    if (
        pd.notnull(btc_price_usd_val)
        and isinstance(btc_price_usd_val, (int, float))
        and btc_price_usd_val > 0
        and target_usd_in > 0
    ):
        sats_disp = f"{int((target_usd_in / btc_price_usd_val) * 100_000_000):,}"
    cols_st[2].metric(label="Equivalent (Sats)", value=sats_disp)

    if cols_st[3].button("Set", key=f"set_target_btn_{sel_coin}", help="Set/update sell target"):
        if sel_coin:
            set_sell_target(sel_coin, target_usd_in)
            msg = f"Sell target for {sel_coin} {'cleared' if target_usd_in <= 0 else f'set to ${target_usd_in:,.2f}'}."
            if target_usd_in > 0:
                st.success(msg)
            else:
                st.info(msg)
            st.rerun()
        else:
            st.error("No coin selected.")


def handle_add_purchase():
    with st.expander("➕ Add New Transaction", expanded=False):
        with st.form(key="add_transaction_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                st.text_input("Cryptocurrency Name*", key="add_form_coin_name")
                st.number_input("Number of Coins*", min_value=1e-8, step=1e-8, format="%.8f", key="add_form_num_coins")
                st.selectbox(
                    "Type*",
                    ["Purchase", "Sell"],
                    key="add_form_tx_type",
                    index=0
                )  # Default to Purchase
            with c2:
                st.number_input("Price per Coin (USD)*", min_value=0.0, step=1e-8, format="%.8f", key="add_form_purchase_price")
                sell_to_options = ["USD", "BTC Satoshis", "N/A"]
                default_sell_to_idx = 2  # N/A
                # Use session state for reliable current type for index logic
                current_tx_type = st.session_state.get("add_form_tx_type", "Purchase")
                if current_tx_type == "Purchase":
                    default_sell_to_idx = 2  # N/A
                elif current_tx_type == "Sell":
                    prev_sell_to = st.session_state.get("add_form_sell_to")
                    if prev_sell_to == "USD":
                        default_sell_to_idx = 0
                    elif prev_sell_to == "BTC Satoshis":
                        default_sell_to_idx = 1
                    else:
                        default_sell_to_idx = 0  # Default to USD for new Sell
                st.selectbox("Sell To", options=sell_to_options, index=default_sell_to_idx, key="add_form_sell_to")
                st.number_input(
                    "BTC Price at Sale (USD)",
                    min_value=0.0,
                    step=0.01,
                    format="%.2f",
                    value=st.session_state.get("add_form_btc_price_sale", 0.0),
                    key="add_form_btc_price_sale"
                )

            st.markdown("Transaction Timestamp:")
            ts_c1a, ts_c2a = st.columns(2)
            # Default to current date/time
            with ts_c1a:
                st.date_input(
                    "Date",
                    value=st.session_state.get("add_form_date", datetime.now().date()),
                    key="add_form_date"
                )
            with ts_c2a:
                st.time_input(
                    "Time",
                    value=st.session_state.get(
                        "add_form_time",
                        datetime.now().time().replace(second=0, microsecond=0)
                    ),
                    key="add_form_time",
                    step=60
                )

            submitted = st.form_submit_button("Add Transaction")
            if submitted:
                # Retrieve values from session state for robustness
                form_coin_name = st.session_state.add_form_coin_name
                form_num_coins = st.session_state.add_form_num_coins
                form_purchase_price = st.session_state.add_form_purchase_price
                form_transaction_type = st.session_state.add_form_tx_type
                form_sell_to = st.session_state.add_form_sell_to
                form_btc_price_sale = st.session_state.add_form_btc_price_sale
                form_date_val = st.session_state.add_form_date
                form_time_val = st.session_state.add_form_time

                if not form_coin_name:
                    st.error("Cryptocurrency name is required.")
                    return
                if form_num_coins <= 0:
                    st.error("Number of coins must be positive.")
                    return
                if form_transaction_type == "Purchase" and form_purchase_price < 0:
                    st.error("Purchase price cannot be negative.")
                    return
                if form_transaction_type == "Sell" and form_purchase_price <= 0:
                    st.error("Sale price must be positive.")
                    return

                timestamp_str = datetime.combine(form_date_val, form_time_val).strftime('%Y-%m-%d %H:%M:%S')
                if form_transaction_type == "Sell":
                    portfolio = get_portfolio()  # Fetch current portfolio to check balance
                    balance = sum(e.coins_purchased if e.transaction_type == "Purchase" else -e.coins_purchased for e in portfolio if e.coin == form_coin_name)
                    if form_num_coins > balance + 1e-8:  # Add epsilon for float comparison
                        st.error(
                            f"Cannot sell {form_num_coins} {form_coin_name}. You only own approx. {balance:,.8f}."
                        )
                        return

                    if form_sell_to == "BTC Satoshis":
                        if not form_btc_price_sale or form_btc_price_sale <= 0:
                            st.error("Valid BTC price at sale is required for selling to BTC Satoshis.")
                            return
                        add_entry(
                            form_coin_name, form_num_coins, form_purchase_price, "Sell",
                            sell_to="BTC Satoshis", btc_price_at_sale=form_btc_price_sale, timestamp=timestamp_str
                        )
                        btc_rcvd = (form_purchase_price / form_btc_price_sale) * form_num_coins
                        add_entry(
                            "Bitcoin", btc_rcvd, form_btc_price_sale, "Purchase", timestamp=timestamp_str
                        )  # Add received BTC as a purchase
                        st.success(
                            f"Sold {form_num_coins} {form_coin_name}. Added {btc_rcvd:,.8f} Bitcoin."
                        )
                    elif form_sell_to == "USD":
                        add_entry(
                            form_coin_name, form_num_coins, form_purchase_price, "Sell",
                            sell_to="USD", timestamp=timestamp_str
                        )
                        usd_rcvd = form_num_coins * form_purchase_price
                        add_entry(
                            "US Dollars", usd_rcvd, 1.0, "Purchase", timestamp=timestamp_str
                        )  # Add received USD as a purchase
                        st.success(
                            f"Sold {form_num_coins} {form_coin_name}. Added ${usd_rcvd:,.2f} USD."
                        )
                    elif form_sell_to == "N/A":
                        st.error(
                            "For 'Sell' transactions, 'Sell To' must be 'USD' or 'BTC Satoshis'. 'N/A' is not a valid option here."
                        )
                        return
                    else:
                        # Should not be reached with current options
                        st.error(
                            f"Invalid 'Sell To' option: {form_sell_to} for a Sell transaction."
                        )
                        return
                elif form_transaction_type == "Purchase":
                    if form_sell_to != "N/A":
                        st.info(
                            f"For 'Purchase' transactions, 'Sell To' is usually 'N/A'. "
                            f"Your selection ('{form_sell_to}') and any BTC sale price will be ignored for this purchase logic."
                        )
                    add_entry(
                        form_coin_name, form_num_coins, form_purchase_price, "Purchase",
                        sell_to=None, btc_price_at_sale=None, timestamp=timestamp_str
                    )
                    st.success(
                        f"Purchased {form_num_coins} {form_coin_name} at ${form_purchase_price:,.8f} each."
                    )
        # st.rerun() is implicitly handled by Streamlit when a form with clear_on_submit=True is submitted


def main():
    st.set_page_config(layout="wide", page_title="Crypto Portfolio Tracker Pro")
    st.title("🪙 Crypto Portfolio Tracker Pro")
    handle_add_purchase()
    st.markdown("---")
    portfolio_entries = get_portfolio()
    if portfolio_entries:
        transactions_df, summary_df_table, btc_price_usd_val, _, _ = display_portfolio_summary(portfolio_entries)
        st.markdown("---")
        display_sell_targets_ui(summary_df_table, btc_price_usd_val)
        st.markdown("---")
        display_transactions_table(transactions_df)
    else:
        st.info("👋 Your portfolio is empty. Add transactions using the section above to begin tracking!")


if __name__ == "__main__":
    main()
