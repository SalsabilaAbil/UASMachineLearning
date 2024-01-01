import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx

# Sample data (replace this with your actual data)
data = {'antecedents': [['A'], ['B'], ['C'], ['D']],
        'consequents': [['X'], ['Y'], ['Z'], ['W']],
        'support': [0.2, 0.3, 0.1, 0.4],
        'confidence': [0.8, 0.6, 0.9, 0.7],
        'lift': [1.2, 1.4, 1.1, 1.3]}

rules = pd.DataFrame(data)

# Scatterplot matrix
plt.figure(figsize=(10, 10))
plt.style.use('seaborn-white')

plt.subplot(221)
sns.scatterplot(x="support", y="confidence", data=rules)
st.pyplot()

plt.subplot(222)
sns.scatterplot(x="support", y="lift", data=rules)
st.pyplot()

plt.subplot(223)
sns.scatterplot(x="confidence", y="lift", data=rules)
st.pyplot()

# Network graph
st.write("Network Graph:")
draw_graph(rules, 4)
st.pyplot()

# Streamlit app with your graph drawing function
st.write("Custom Graph:")
draw_graph(rules, 4)
st.pyplot()
