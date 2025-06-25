import os
import sys
import pickle

# Proje kökünü ekle
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_loader import load_raw_retail_data
from src.preprocessing import clean_retail_data

def main():
    # 1) Ham veriyi oku
    df_raw = load_raw_retail_data(
        file_path="data/online_retail_II.xlsx",
        sheet_name="Year 2010-2011"
    )

    # 2) Temizle
    df_clean = clean_retail_data(df_raw)

    # 3) Diske kaydet
    os.makedirs("data/precomputed", exist_ok=True)
    path = "data/precomputed/clean_retail.pkl"
    with open(path, "wb") as f:
        pickle.dump(df_clean, f)

    print(f"✅ Cleaned data saved to {path}")

if __name__ == "__main__":
    main()
