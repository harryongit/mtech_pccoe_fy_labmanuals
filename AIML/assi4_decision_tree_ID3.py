# Decision Tree using ID3 Algorithm (From Scratch)

import math
import pandas as pd

# Sample dataset (Play Tennis)
data = {
    'Outlook': ['Sunny','Sunny','Overcast','Rain','Rain','Rain','Overcast','Sunny','Sunny','Rain','Sunny','Overcast','Overcast','Rain'],
    'Temperature': ['Hot','Hot','Hot','Mild','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Mild','Hot','Mild'],
    'Humidity': ['High','High','High','High','Normal','Normal','Normal','High','Normal','Normal','Normal','High','Normal','High'],
    'Wind': ['Weak','Strong','Weak','Weak','Weak','Strong','Strong','Weak','Weak','Weak','Strong','Strong','Weak','Strong'],
    'PlayTennis': ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']
}

df = pd.DataFrame(data)

def entropy(col):
    values = col.value_counts()
    total = len(col)
    return -sum((count/total) * math.log2(count/total) for count in values)

def information_gain(df, attribute, target):
    total_entropy = entropy(df[target])
    values = df[attribute].unique()
    weighted_entropy = 0

    for val in values:
        subset = df[df[attribute] == val]
        weighted_entropy += (len(subset)/len(df)) * entropy(subset[target])

    return total_entropy - weighted_entropy

def id3(df, target, attributes):
    target_values = df[target].unique()
    if len(target_values) == 1:
        return target_values[0]

    if not attributes:
        return df[target].mode()[0]

    gains = {attr: information_gain(df, attr, target) for attr in attributes}
    best_attr = max(gains, key=gains.get)
    tree = {best_attr: {}}

    for val in df[best_attr].unique():
        subset = df[df[best_attr] == val]
        remaining_attrs = [a for a in attributes if a != best_attr]
        tree[best_attr][val] = id3(subset, target, remaining_attrs)

    return tree

attributes = list(df.columns[:-1])
decision_tree = id3(df, 'PlayTennis', attributes)

print("Decision Tree using ID3 Algorithm:")
print(decision_tree)
