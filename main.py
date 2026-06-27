from Utilities.utils import GetOHLC, TransformDF, SplitDF
from Utilities.model import TrainXGBoost, BuildBaselineEquity
from Utilities.plot import SaveEquity
import pandas as pd

df = GetOHLC(
        pair="BTC/USD",
        interval=1440,   # daily candles
    )

df = TransformDF(df)

test_df, X_train, X_test, y_train, y_test = SplitDF(df)

BaselineTestDF = test_df.copy()

Probabilities = TrainXGBoost(X_train, X_test, y_train, y_test)

BuildBaselineEquity(BaselineTestDF, Probabilities)

SaveEquity("XGBoost Baseline Model with Probability Filter", BaselineTestDF)


#print(df.head())
#print(df.columns)