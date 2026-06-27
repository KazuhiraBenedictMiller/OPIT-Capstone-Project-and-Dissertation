import pandas as pd
from xgboost import XGBClassifier

# Function that Trains the XGBoost Model
def TrainXGBoost(X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.DataFrame, y_test:pd.DataFrame):
    model = XGBClassifier( 
        n_estimators=200, 
        max_depth=5, 
        learning_rate=0.05, 
        subsample=0.8, 
        colsample_bytree=0.8, 
        random_state=42, 
        eval_metric="logloss" 
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test) 

    probs = model.predict_proba(X_test)[:, 1]

    return probs

# Function that Builds the Equity for the Baseline Model
def BuildBaselineEquity(df:pd.DataFrame, probs:list):
    df["probability"] = probs

    df["position"] = 0

    df.loc[
        (df["probability"] > 0.6),
        "position"
    ] = 1

    df.loc[
        (df["probability"] < 0.4), 
        "position"
    ] = -1

    df["strategy_return"] = (
        df["position"] * df["future_return"]
    )  

    df["strategy_equity"] = (
        1 + df["strategy_return"]
    ).cumprod()