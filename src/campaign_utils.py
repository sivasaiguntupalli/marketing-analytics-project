import pandas as pd


def compute_campaign_metrics(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Compute derived metrics for marketing campaigns such as CTR, conversion rate, cost per conversion and ROI.

    Parameters
    ----------
    df : pd.DataFrame
        Campaign data with columns ['Impressions', 'Clicks', 'Conversions', 'Cost', 'Revenue'].

    Returns
    -------
    pd.DataFrame
        The input DataFrame augmented with new metric columns.
    '''
    df = df.copy()
    df["CTR"] = df["Clicks"] / df["Impressions"]
    df["ConversionRate"] = df["Conversions"] / df["Clicks"].replace(0, pd.NA)
    df["CostPerConversion"] = df["Cost"] / df["Conversions"].replace(0, pd.NA)
    df["ROI"] = (df["Revenue"] - df["Cost"]) / df["Cost"]
    return df
