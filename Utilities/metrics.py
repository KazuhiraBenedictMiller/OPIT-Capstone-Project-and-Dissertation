import pandas as pd

# Function to Calculate the Profit Factor
def ProfitFactor(df):

    # winning trades
    gross_profit = df.loc[
        df["strategy_return"] > 0,
        "strategy_return"
    ].sum()

    # losing trades
    gross_loss = abs(
        df.loc[
            df["strategy_return"] < 0,
            "strategy_return"
        ].sum()
    )

    profit_factor = gross_profit / gross_loss

    return profit_factor


def MaxDrawDown(df):
    # rolling maximum equity
    df["equity_peak"] = (
        df["strategy_equity"].cummax()
    )

    # drawdown
    df["drawdown"] = (
        df["strategy_equity"] /
        df["equity_peak"] - 1
    )

    # maximum drawdown
    max_drawdown = df["drawdown"].min()

    return max_drawdown