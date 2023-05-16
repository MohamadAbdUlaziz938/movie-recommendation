import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.neighbors import NearestNeighbors


def apriori_algorithm(sets: pd.DataFrame):
    frequent_itemsets = apriori(sets, min_support=0.15, use_colnames=True)
    rules = association_rules(
        frequent_itemsets, metric="lift", min_threshold=1.75
    ).sort_values("lift", ascending=False)
    rules = rules[["antecedents", "consequents", "support", "lift", "confidence"]]
