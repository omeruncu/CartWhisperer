import os
import sys
import pickle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_loader     import load_raw_retail_data
from src.preprocessing  import clean_retail_data
from src.rule_generator import create_cart, generate_global_rules, filter_rules_by_country

def load_or_clean_data() -> "pd.DataFrame":
    """
    Önceden temizlenmiş data varsa yükle,
    yoksa temizle ve diske kaydet.
    """
    clean_path = "data/precomputed/clean_retail.pkl"
    if os.path.exists(clean_path):
        with open(clean_path, "rb") as f:
            return pickle.load(f)

    # yoksa ham veriyi oku → temizle → kaydet
    df_raw   = load_raw_retail_data()
    df_clean = clean_retail_data(df_raw)
    os.makedirs("data/precomputed", exist_ok=True)
    with open(clean_path, "wb") as f:
        pickle.dump(df_clean, f)
    return df_clean

def main():
    # 1) Temizlenmiş DataFrame’i al
    df = load_or_clean_data()

    # 2) Cart’i oluştur
    cart = create_cart(df)

    # 3) Global rule’ları üret & kaydet
    global_rules = generate_global_rules(cart, min_support=0.01, min_confidence=0.2, top_k=500)
    with open("data/precomputed/global_rules.pkl", "wb") as f:
        pickle.dump(global_rules, f)

    # 4) Country‐bazlı rule’ları üret & kaydet
    for country in sorted(df["country"].unique()):
        country_rules = filter_rules_by_country(
            global_rules, df, country=country, top_n=200, metric="lift"
        )
        with open(f"data/precomputed/rules_{country}.pkl", "wb") as f:
            pickle.dump(country_rules, f)

    print("✅ All offline precomputed .pkl files are ready under data/precomputed/")

if __name__ == "__main__":
    main()
