import pandas as pd

def quantile_clip(
    series: pd.Series,
    low_pct: float,
    high_pct: float
) -> pd.Series:
    """
    Verilen serinin low_pct ve high_pct yüzdelik değerlerine göre
    alt ve üst uç değerlerini clip eder.
    """
    lower, upper = series.quantile([low_pct, high_pct])
    return series.clip(lower=lower, upper=upper)
