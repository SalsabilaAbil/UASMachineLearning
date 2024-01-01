import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import association_rules, apriori
import matplotlib.pyplot as plt
import seaborn as sns
# Sample data
data = pd.DataFrame({
    'Transaction': [1, 1, 2, 2, 3, 3, 4, 4],
    'Item': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B']
})

# Data preprocessing and association rule mining
data["Item"] = data["Item"].apply(lambda item: item.lower())
data["Item"] = data["Item"].apply(lambda item: item.strip())
data = data[["Transaction", "Item"]].copy()

item_count = data.groupby(["Transaction", "Item"])["Item"].count().reset_index(name="Count")
item_count_pivot = item_count.pivot_table(index='Transaction', columns='Item', values='Count', aggfunc='sum').fillna(0)

item_count_pivot = item_count_pivot.astype("int32")
item_count_pivot = item_count_pivot.applymap(lambda x: 1 if x >= 1 else 0)

support = 0.01
frequent_items = apriori(item_count_pivot, min_support=support, use_colnames=True)
rules = association_rules(frequent_items, metric="lift", min_threshold=1)[["antecedents", "consequents", "support", "confidence", "lift"]]
rules.sort_values('confidence', ascending=False, inplace=True)

# Streamlit app
st.title("Association Rule Mining with Streamlit")

# Display the data
st.header("Transaction-Item Data")
st.dataframe(data)

# Display the association rules
st.header("Association Rules")
st.dataframe(rules.head(15))

# Scatter plots
st.header("Scatter Plots")
plt.figure(figsize=(10, 10))
plt.subplot(221)
sns.scatterplot(x="support", y="confidence", data=rules)
plt.subplot(222)
sns.scatterplot(x="support", y="lift", data=rules)
plt.subplot(223)
sns.scatterplot(x="confidence", y="lift", data=rules)
st.pyplot()

# Network graph
st.header("Network Graph")
G = nx.DiGraph()

for i in range(10):  # Displaying top 10 rules in the graph
    for a in rules.iloc[i]['antecedents']:
        G.add_edge(a, "R" + str(i))
    for c in rules.iloc[i]['consequents']:
        G.add_edge("R" + str(i), c)

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, k=16, scale=1)
nx.draw(G, pos, with_labels=True, font_size=8, node_size=1500, font_color='black', font_weight='bold', node_color='skyblue')
st.pyplot()
