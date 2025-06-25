import sys
from src.data_loader import load_raw_retail_data, load_clean_retail_data
from src.rule_generator   import (
    create_cart,
    generate_global_rules,
    filter_rules_by_country,
    generate_context_aware_rules
)


def main():
    # 1) Veri ön işleme
    df = load_clean_retail_data()
    print("Clean DF shape:", df.shape)

    # 2) Cart matrisi
    cart = create_cart(df)
    print("Cart shape:", cart.shape)

    # 2) Hepsini bir arada: generate_context_aware_rules
    g_rules2, ger_rules2 = generate_context_aware_rules(
        df,
        min_support=0.01,
        min_confidence=0.2,
        top_k=50,
        country="Germany",
        top_n=5,
        metric="lift"
    )
    print("\n--- Context-Aware (Global) Top 5 ---")
    print(g_rules2.head())
    print("\n--- Context-Aware (Germany) Top 5 ---")
    print(ger_rules2.head())

if __name__ == "__main__":
    main()
