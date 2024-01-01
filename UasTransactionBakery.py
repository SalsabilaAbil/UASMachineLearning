import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import association_rules, apriori

# Load data
@st.cache  # Use caching to avoid reloading data on every interaction
def load_data():
    return pd.read_csv('transactions-from-a-bakery/BreadBasket_DMS.csv')

data = load_data()

# Preprocess data
data["Item"] = data["Item"].apply(lambda item: item.lower())
data["Item"] = data["Item"].apply(lambda item: item.strip())
data = data[["Transaction", "Item"]].copy()

# Create pivot table
item_count = data.groupby(["Transaction", "Item"])["Item"].count().reset_index(name="Count")
item_count_pivot = item_count.pivot_table(index='Transaction', columns='Item', values='Count', aggfunc='sum').fillna(0)
item_count_pivot = item_count_pivot.astype("int32")
item_count_pivot = item_count_pivot.applymap(lambda x: 1 if x >= 1 else 0)

# Run Apriori algorithm
support = 0.01
frequent_items = apriori(item_count_pivot, min_support=support, use_colnames=True)

# Run association rules
metric = "lift"
min_threshold = 1
rules = association_rules(frequent_items, metric=metric, min_threshold=min_threshold)[["antecedents", "consequents", "support", "confidence", "lift"]]
rules.sort_values('confidence', ascending=False, inplace=True)

# Streamlit App
st.title('Association Rule Mining with Streamlit')
st.sidebar.header('Settings')

# Display association rules
st.subheader('Association Rules')
st.write(rules.head(15))

# Display scatter plot matrix
st.subheader('Scatter Plot Matrix')
st.pyplot()

# Display association graph
st.subheader('Association Graph')
st.pyplot()

# Note: For the scatter plot and graph, you may need to adjust the visualization code
# as Streamlit does not automatically handle multiple subplots.
