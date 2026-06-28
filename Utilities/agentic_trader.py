from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from .prompts import TA_SYSTEM_PROMPT, TA_PROMPT, REASONING_SYSTEM_PROMPT, REASONING_PROMPT
from typing import Literal
from pydantic import BaseModel, Field

class MarketOutlook(BaseModel):
    outlook: Literal["bullish", "bearish", "neutral"] = Field(
        description="Expected market direction for the next trading day"
    )
    reasoning: str = Field(
        description="Complete Explanation of the analysis and reasoning behind it."
    )

def TechnicalAnalysis(row):

    llm = ChatOllama(
        #model="phi4-mini-reasoning:3.8b",
        model = "llama3.1:8b",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", TA_SYSTEM_PROMPT),
        ("human", TA_PROMPT)
    ])

    TAchain = prompt | llm

    TAresponse = TAchain.invoke({
        "rsi": row["rsi"],
        "rsi_lag5": row["rsi_lag5"],
        "rsi_lag4": row["rsi_lag4"],
        "rsi_lag3": row["rsi_lag3"],
        "rsi_lag2": row["rsi_lag2"],
        "rsi_lag1": row["rsi_lag1"],

        "ema20_dist": row["ema20_dist"],
        "ema20_dist_lag5": row["ema20_dist_lag5"],
        "ema20_dist_lag4": row["ema20_dist_lag4"],
        "ema20_dist_lag3": row["ema20_dist_lag3"],
        "ema20_dist_lag2": row["ema20_dist_lag2"],
        "ema20_dist_lag1": row["ema20_dist_lag1"],

        "open": row["open"],
        "open_lag5": row["open_lag5"],
        "open_lag4": row["open_lag4"],
        "open_lag3": row["open_lag3"],
        "open_lag2": row["open_lag2"],
        "open_lag1": row["open_lag1"],

        "high": row["high"],
        "high_lag5": row["high_lag5"],
        "high_lag4": row["high_lag4"],
        "high_lag3": row["high_lag3"],
        "high_lag2": row["high_lag2"],
        "high_lag1": row["high_lag1"],

        "low": row["low"],
        "low_lag5": row["low_lag5"],
        "low_lag4": row["low_lag4"],
        "low_lag3": row["low_lag3"],
        "low_lag2": row["low_lag2"],
        "low_lag1": row["low_lag1"],

        "close": row["close"],
        "close_lag5": row["close_lag5"],
        "close_lag4": row["close_lag4"],
        "close_lag3": row["close_lag3"],
        "close_lag2": row["close_lag2"],
        "close_lag1": row["close_lag1"],

        "volume": row["volume"],
        "volume_lag5": row["volume_lag5"],
        "volume_lag4": row["volume_lag4"],
        "volume_lag3": row["volume_lag3"],
        "volume_lag2": row["volume_lag2"],
        "volume_lag1": row["volume_lag1"],

        "ret3": row["ret3"],
        "ret7": row["ret7"]
    })

    return TAresponse.content

def Reasoning(TAnalysis:str):
    llm = ChatOllama(
        model="phi4-mini-reasoning:3.8b",
        #model = "llama3.1:8b",
        temperature=0
    )

    structured_llm = llm.with_structured_output(MarketOutlook)

    prompt = ChatPromptTemplate.from_messages([
        ("system", REASONING_SYSTEM_PROMPT),
        ("human", REASONING_PROMPT)
    ])

    REASONINGchain = prompt | structured_llm

    response = REASONINGchain.invoke({
        "report": TAnalysis,
    })

    return response.outlook, response.reasoning

def BuildAgenticEquity(predictions, df):
    df["prediction"] = predictions

    position_map = {
        "bullish": 1,
        "bearish": -1,
        "neutral": 0
    }

    df["position"] = df["prediction"].map(position_map)

    df["strategy_return"] = (
        df["position"] *
        df["future_return"]
    )

    df["strategy_equity"] = (
        1 + df["strategy_return"]
    ).cumprod()