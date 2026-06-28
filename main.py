from Utilities.utils import GetOHLC, TransformDF, SplitDF
from Utilities.model import TrainXGBoost, BuildBaselineEquity
from Utilities.metrics import ProfitFactor, MaxDrawDown
from Utilities.plot import SaveEquity
from Utilities.agentic_trader import TechnicalAnalysis, Reasoning, BuildAgenticEquity
import pandas as pd

df = GetOHLC(
        pair="BTC/USD",
        interval=1440,   # daily candles
    )

df = TransformDF(df)

test_df, X_train, X_test, y_train, y_test = SplitDF(df)

# BASELINE MODEL

BaselineTestDF = test_df.copy()

Probabilities = TrainXGBoost(X_train, X_test, y_train, y_test)

BuildBaselineEquity(BaselineTestDF, Probabilities)

# Caching Results
BaselineTestDF.to_csv("./Data/BaselineModel_Results.csv")

SaveEquity("XGBoost Baseline Model with Probability Filter Equity Curve", "BaselineModelEquity", BaselineTestDF)

baselinepf = ProfitFactor(BaselineTestDF)

baselinemaxdd = MaxDrawDown(BaselineTestDF)

print(f"Baseline Model Profit Factor: {baselinepf}")
print(f"Baseline Model Max DrawDown: {baselinemaxdd}")

# MULTI-AGENT MODEL

AgenticTestDF = test_df.copy()

predictions = []
TAanalysis = []
REASONINGanalysis = []

for idx, row in AgenticTestDF.iterrows():
    print(f"Index Number {idx}")

    TAResponse = TechnicalAnalysis(row)

    outlook, reasoning = Reasoning(TAResponse)

    TAanalysis.append(TAresponse)
    REASONINGanalysis.append(reasoning)
    predictions.append(outlook)

BuildAgenticEquity(predictions, AgenticTestDF)

# Caching Results
AgenticTestDF.to_csv("./Data/MultiAgentLLM_Results.csv")

SaveEquity("MultiAgent Model Equity Curve", "AgenticModelEquity", AgenticTestDF)

agenticpf = ProfitFactor(AgenticTestDF)

agenticmaxdd = MaxDrawDown(AgenticTestDF)

print(f"Multi-Agent Model Profit Factor: {agenticpf}")
print(f"Multi-Agent Model Max DrawDown: {agenticmaxdd}")

#print(df.head())
#print(df.columns)