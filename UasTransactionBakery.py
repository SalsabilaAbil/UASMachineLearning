import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import association_rules, apriori

# Masukkan data Anda atau gunakan data dummy untuk tujuan demonstrasi
# Misalnya, gantilah 'your_data.csv' dengan nama file data Anda
data = pd.read_csv('your_data.csv')

# Preprocessing data
data["Item"] = data["Item"].apply(lambda item: item.lower())
data["Item"] = data["Item"].apply(lambda item: item.strip())
data = data[["Transaction", "Item"]].copy()

# Membuat pivot table
item_count = data.groupby(["Transaction", "Item"])["Item"].count().reset_index(name="Count")
item_count_pivot = item_count.pivot_table(index='Transaction', columns='Item', values='Count', aggfunc='sum').fillna(0)
item_count_pivot = item_count_pivot.astype("int32")
item_count_pivot = item_count_pivot.applymap(lambda x: 1 if x >= 1 else 0)

# Menjalankan algoritma Apriori
support = 0.01
frequent_items = apriori(item_count_pivot, min_support=support, use_colnames=True)

# Menjalankan aturan asosiasi
metric = "lift"
min_threshold = 1
rules = association_rules(frequent_items, metric=metric, min_threshold=min_threshold)[["antecedents", "consequents", "support", "confidence", "lift"]]
rules.sort_values('confidence', ascending=False, inplace=True)

# Menjalankan aplikasi Streamlit
st.title('Association Rule Mining with Streamlit')
st.sidebar.header('Settings')

# Menampilkan tabel aturan asosiasi
st.subheader('Association Rules')
st.write(rules.head(15))

# Menampilkan scatter plot matrix
st.subheader('Scatter Plot Matrix')
st.pyplot()

# Menampilkan graf asosiasi
st.subheader('Association Graph')
st.pyplot()

# Catatan: Untuk scatter plot dan graf, Anda perlu menyesuaikan kode di bagian visualisasi
# karena Streamlit tidak secara otomatis menangani beberapa subplot.
