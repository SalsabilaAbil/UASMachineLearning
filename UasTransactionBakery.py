import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import networkx as nx

# Load your data (replace 'your_data.csv' with your actual data file)
data = pd.read_csv('TransactionFromBakery.csv')

# ... (Your existing data preprocessing and mining code)

# Streamlit App
st.title('Association Rule Mining with Streamlit')

# Display the top 10 association rules
st.header('Top 10 Association Rules')
st.table(rules.head(10))

# Scatter plots
st.header('Scatter Plots')
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
sns.scatterplot(x="support", y="confidence", data=rules, ax=axes[0, 0])
sns.scatterplot(x="support", y="lift", data=rules, ax=axes[0, 1])
sns.scatterplot(x="confidence", y="lift", data=rules, ax=axes[1, 0])
st.pyplot(fig)

# Network Graph
st.header('Network Graph of Association Rules')
fig_network = plt.figure(figsize=(10, 10))
draw_graph(rules, 10)
st.pyplot(fig_network)
