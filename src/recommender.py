import pandas as pd

def recommend_from_rules(
    user_items: list[str],
    rules: pd.DataFrame,
    metric: str = "lift",
    top_n: int = 5
) -> list[tuple[str,float]]:
    """
    rules içindeki antecedents ⊆ user_items olan satırları seç,
    consequents’leri metric’e göre ağırlıklandırıp topla, sırala.
    Dönen: [(ürün_a, skor), …] listesi.
    """
    scores = {}
    for _, row in rules.iterrows():
        if set(row["antecedents"]).issubset(set(user_items)):
            for item in row["consequents"]:
                scores[item] = scores.get(item, 0.0) + row[metric]
    # Skoru azalan sıralayıp top_n al
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top
