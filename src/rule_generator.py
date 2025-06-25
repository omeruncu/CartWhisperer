import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def create_cart(df: pd.DataFrame) -> pd.DataFrame:
    """
    invoice × description pivot tablosu üretir (binary sepet matrisi).
    """
    cart = (
        df
        .groupby(['invoice', 'description'])['quantity']
        .sum()
        .unstack(fill_value=0)
        .applymap(lambda x: 1 if x > 0 else 0)
    )
    return cart


def generate_global_rules(
    cart: pd.DataFrame,
    min_support: float = 0.01,
    min_confidence: float = 0.2,
    top_k: int = 100
) -> pd.DataFrame:
    """
    Level-1: Global kuralları çıkarır.
      1) apriori ile sık itemset’leri bul,
      2) association_rules ile kuralları üret,
      3) confidence’a göre azalan sırayla top_k al.
    """
    freq_itemsets = apriori(cart, min_support=min_support, use_colnames=True)
    rules = association_rules(
        freq_itemsets,
        metric='confidence',
        min_threshold=min_confidence
    )
    return (
        rules
        .sort_values('confidence', ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )


def filter_rules_by_country(
    global_rules: pd.DataFrame,
    df: pd.DataFrame,
    country: str,
    top_n: int = 10,
    metric: str = 'lift'
) -> pd.DataFrame:
    """
    Level-2: global_rules havuzundan, sadece `country` segmentindeki
    local_support & local_confidence hesaplayıp en iyi top_n kuralı döner.
    """
    # 1) Country bazlı cart
    cart_c = create_cart(df[df['country'] == country])

    # 2) Global cart sütunlarına göre reindex et
    cart_all = create_cart(df)
    cart_c   = cart_c.reindex(columns=cart_all.columns, fill_value=0)

    # 3) Her kural için local metrikler
    def _local_stats(row):
        A, B    = list(row['antecedents']), list(row['consequents'])
        mask_A  = cart_c[A].all(axis=1)
        mask_AB = cart_c[A + B].all(axis=1)
        sup  = mask_AB.sum() / len(cart_c)      if len(cart_c)    else 0
        conf = mask_AB.sum() / mask_A.sum()     if mask_A.sum()>0 else 0
        return pd.Series({
            'local_support':    sup,
            'local_confidence': conf
        })

    local     = global_rules.apply(_local_stats, axis=1)
    rules_loc = pd.concat([global_rules, local], axis=1)

    # 4) metric’e göre sırala & top_n
    return (
        rules_loc
        .sort_values(metric, ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )