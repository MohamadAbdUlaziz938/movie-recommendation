import random
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def encode_units(k):
    if k <= 0:
        return 0
    if k >= 1:
        return 1
    else:
        return 0

class Aprior:
    def __init__(self, data: pd.DataFrame) -> None:
        self.recommendation_data = data.applymap(encode_units)
        self.rules = None

    def fit(self):
        frequent_itemsets = apriori(
            self.recommendation_data, min_support=0.15, use_colnames=True
        )
        rules = association_rules(
            frequent_itemsets, metric="lift", min_threshold=1.75
        ).sort_values("lift", ascending=False)
        self.rules = rules[
            ["antecedents", "consequents", "support", "lift", "confidence"]
        ]
        print(rules)

    def recomend(self):
        i = random.randint(0, self.rules.shape[0])
        print("Antecedents:", self.rules.iloc[i].antecedents)
        print("Consequents:", self.rules.iloc[i].consequents)
        print(
            f"Lift: {self.rules.iloc[i].lift.round(3)} & Confidence: {self.rules.iloc[i].confidence.round(3)} & Support: {self.rules.iloc[i].support.round(3)}"
        )
