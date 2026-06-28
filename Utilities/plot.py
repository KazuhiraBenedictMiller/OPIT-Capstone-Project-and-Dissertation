import matplotlib.pyplot as plt
import pandas as pd

def SaveEquity(Title:str, fname:str, df:pd.DataFrame):
    plt.figure(figsize=(12, 6))
  
    plt.plot(
        df["datetime"],
        df["strategy_equity"]
    )

    plt.xlabel("Date")
    plt.ylabel("Equity")

    #plt.legend()

    plt.title(Title)

    plt.savefig("./Pictures/{fname}.jpeg")


