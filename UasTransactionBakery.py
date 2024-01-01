import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Function to generate association rules using Apriori algorithm
def generate_association_rules(data, min_support, min_confidence):
    # Apriori algorithm
    frequent_itemsets = apriori(data, min_support=min_support, use_colnames=True)

    # Association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    return rules

# Streamlit app
def main():
    st.title("Association Rule Mining with Apriori Algorithm")

    # Upload CSV file
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        st.sidebar.success("File uploaded successfully!")

        # Load data
        data = pd.read_csv(uploaded_file)

        # Display the raw data
        st.subheader("Raw Data")
        st.write(data)

        # Set parameters
        min_support = st.sidebar.slider("Minimum Support", 0.0, 1.0, 0.1, step=0.01)
        min_confidence = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.5, step=0.01)

        # Generate association rules
        rules = generate_association_rules(data, min_support, min_confidence)

        # Display association rules
        st.subheader("Association Rules")
        st.write(rules)

if __name__ == "__main__":
    main()
