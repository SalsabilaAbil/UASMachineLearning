import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
@st.cache
def load_data():
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/breadbasket_dms.csv"
    data = pd.read_csv(url)
    return data

# Preprocess the data
def preprocess_data(data):
    data["Item"] = data["Item"].str.lower().str.strip()
    return data

# Mine association rules
def mine_association_rules(data, min_support=0.01):
    item_count = data.groupby(["Transaction", "Item"])["Item"].count().reset_index(name="Count")
    item_count_pivot = item_count.pivot_table(index='Transaction', columns='Item', values='Count', aggfunc='sum').fillna(0)
    item_count_pivot = item_count_pivot.astype("int32")
    item_count_pivot = item_count_pivot.applymap(lambda x: 1 if x >= 1 else 0)
    frequent_items = apriori(item_count_pivot, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_items, metric="lift", min_threshold=1)
    return rules

# Draw the network graph
def draw_graph(rules, rules_to_show=10):
    G = nx.DiGraph()

    for i in range(rules_to_show):
        antecedents = list(rules.iloc[i]['antecedents'])
        consequents = list(rules.iloc[i]['consequents'])

        for antecedent in antecedents:
            G.add_edge(antecedent, f"Rule {i}")

        for consequent in consequents:
            G.add_edge(f"Rule {i}", consequent)

    pos = nx.spring_layout(G, k=16, scale=1)
    nx.draw(G, pos, with_labels=True, font_size=8, node_size=2000, node_color="skyblue", font_color="black", font_weight="bold")
    st.pyplot()

# Streamlit app
def main():
    st.title("Bakery Association Rule Mining App")

    # Load data
    data = load_data()
    st.sidebar.subheader("Data Exploration")
    st.sidebar.write(data.head())

    # Preprocess data
    data = preprocess_data(data)

    # Mine association rules
    min_support = st.sidebar.slider("Select Minimum Support", min_value=0.01, max_value=0.5, step=0.01, value=0.01)
    rules = mine_association_rules(data, min_support)

    st.sidebar.subheader("Association Rules")
    st.sidebar.write(rules.head())

    # Draw network graph
    st.subheader("Association Rules Network Graph")
    rules_to_show = st.sidebar.slider("Number of Rules to Show", min_value=5, max_value=len(rules), value=10)
    draw_graph(rules, rules_to_show)

if __name__ == "__main__":
    main()
