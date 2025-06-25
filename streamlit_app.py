import os
import pickle
import streamlit as st
import pandas as pd

from src.recommender import recommend_from_rules
from src.rule_generator import create_cart  # yalnızca cart pivot için

st.set_page_config(page_title="Cart-Whisperer", layout="wide")


@st.cache_resource(show_spinner=False)
def load_clean_data() -> pd.DataFrame:
    """
    data/precomputed/clean_retail.pkl varsa yükle,
    yoksa hata ver.
    """
    path = "data/precomputed/clean_retail.pkl"
    if not os.path.exists(path):
        st.error(f"Clean data bulunamadı: {path}")
        return pd.DataFrame()
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_precomputed_rules(mode: str, country: str | None = None) -> pd.DataFrame:
    """
    Global veya country-specific rule set’ini diskten yükler.
    """
    if mode == "Global":
        path = "data/precomputed/global_rules.pkl"
    else:
        path = f"data/precomputed/rules_{country}.pkl"

    if not os.path.exists(path):
        st.error(f"Rule set bulunamadı: {path}")
        return pd.DataFrame()

    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    st.sidebar.header("Settings")
    mode = st.sidebar.radio("Mode", ["Global", "Country"])
    df   = load_clean_data()
    if df.empty:
        return

    country = None
    if mode == "Country":
        country = st.sidebar.selectbox("Country", sorted(df["country"].unique()))

    # Sepet pivot’u (sadece kolon listesi için)
    cart = create_cart(df)
    products = cart.columns.tolist()

    # Kuralları yükle
    rules = load_precomputed_rules(mode, country)
    if rules.empty:
        return

    # Kullanıcı arayüzü: sepet girişi ve öneri
    st.title("🛒 Cart-Whisperer Recommender")
    st.sidebar.subheader("Your Cart")
    user_cart = st.sidebar.multiselect("Select items", products)

    if st.sidebar.button("Get Recommendations"):
        if not user_cart:
            st.error("Lütfen en az bir ürün seçin.")
        else:
            recs = recommend_from_rules(user_cart, rules, metric="lift", top_n=10)
            if not recs:
                st.warning("Bu kombinasyona uygun öneri bulunamadı.")
            else:
                df_recs = pd.DataFrame(recs, columns=["Ürün", "Score"])
                st.subheader("Önerilen Ürünler")
                st.dataframe(df_recs)

    # Kuralları incelet
    with st.expander("View Rules"):
        st.write(f"Toplam kural sayısı: {len(rules)}")
        st.dataframe(rules)

if __name__ == "__main__":
    main()
