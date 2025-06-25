import pandas as pd
from .utils import quantile_clip


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts all column names to strip, lower, snake_case format.
    """
    df = df.copy()
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(" ", "_")
          .str.replace(r"[^\w_]", "", regex=True)
    )

    return df


def _remove_cancelled_invoices(df: pd.DataFrame) -> pd.DataFrame:
    """
    invoice sütununu string’e çevirir ve
    'C' ile başlayan (iptal) faturaları drop eder.
    """
    df = df.copy()
    df["invoice"] = df["invoice"].astype(str)
    mask = ~df["invoice"].str.startswith("C", na=False)

    return df.loc[mask]


def _cap_outliers(
    df: pd.DataFrame,
    qty_lower_pct: float = 0.01,
    qty_upper_pct: float = 0.99,
    price_lower_pct: float = 0.01,
    price_upper_pct: float = 0.99
) -> pd.DataFrame:
    """
    quantity ve price için belirtilen yüzdeliklere göre
    alt/üst uçları clipping yöntemiyle bastırır.
    """
    df = df.copy()
    df["quantity"]   = quantile_clip(df["quantity"],   qty_lower_pct,   qty_upper_pct)
    df["price"] = quantile_clip(df["price"], price_lower_pct, price_upper_pct)
    return df


def clean_retail_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adım adım ön işleme:
      1) Sütun adlarını normalize et
      2) 'POST' stockcode satırlarını çıkar
      3) Boş değer içeren satırları at
      4) İptal faturaları (_remove_cancelled_invoices) çıkar
      5) quantity>0 ve price>0 kayıtlarını filtrele
      6) Aykırı değerleri clipping ile bastır (_cap_outliers)

    """
    df = df.copy()
    # 1) Kolon isimlerini normalize et
    df = _rename_columns(df)
    # 2) 'POST' ürünleri çıkar
    df = df[df["stockcode"].str.upper() != "POST"]
    # 3) NaN içeren satırları at
    df = df.dropna(how="any")
    # 4) İptal edilen faturaları çıkar
    df = _remove_cancelled_invoices(df)
    # 5) Mantıksız adet/fiyatı at
    df = df[(df["quantity"] > 0) & (df["price"] > 0)]
    # 6) Aykırı değer baskılama
    df = _cap_outliers(df)

    return df.reset_index(drop=True)
