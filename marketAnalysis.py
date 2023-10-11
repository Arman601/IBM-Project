# Import necessary libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Sample transaction data (you can replace this with your own dataset)
data = {'Transaction': [1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5],
        'Item': ['A', 'B', 'A', 'C', 'B', 'C', 'D', 'A', 'C', 'D', 'E', 'F']}

df = pd.DataFrame(data)

# Convert the data to a one-hot encoded format
basket = pd.crosstab(index=df['Transaction'], columns=df['Item'])

# Apply Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(basket, min_support=0.2, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Display the frequent itemsets and association rules
print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules)
