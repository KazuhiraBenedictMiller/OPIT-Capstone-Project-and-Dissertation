import matplotlib.pyplot as plt
import pandas as pd

def SaveEquity(Title:str, df:pd.DataFrame):
    plt.figure(figsize=(12, 6))

    plt.plot(
        df["strategy_equity"],
    )

    plt.legend()

    plt.title(Title)

    plt.savefig("./Pictures/Probability Based Baseline Model.jpeg")