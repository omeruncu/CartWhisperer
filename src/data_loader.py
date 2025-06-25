import os
import pandas as pd
from .preprocessing import clean_retail_data, _rename_columns

def load_raw_retail_data(
    file_path="data/online_retail_II.xlsx",
    sheet_name="Year 2010-2011"
) -> pd.DataFrame:
    """Ham Excel’i oku ve sadece sütun isimlerini normalize et."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    xls = pd.ExcelFile(file_path, engine="openpyxl")
    if sheet_name not in xls.sheet_names:
        raise ValueError(f"Sheet {sheet_name} not found.")
    df = pd.read_excel(xls, sheet_name=sheet_name, parse_dates=["InvoiceDate"])

    return df

def load_clean_retail_data(
    file_path="data/online_retail_II.xlsx",
    sheet_name="Year 2010-2011"
) -> pd.DataFrame:
    """Raw veriyi yükle, preprocessing’te tanımlı temizleme adımlarını uygula."""
    raw = load_raw_retail_data(file_path, sheet_name)
    return clean_retail_data(raw)