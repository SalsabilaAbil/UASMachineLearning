import streamlit as st
import pandas as pd
from apyori import apriori

# Fungsi untuk membaca data CSV
def load_data():
    data = pd.read_csv("path/to/your/data.csv")  # Ganti dengan path file CSV Anda
    return data

# Fungsi untuk menerapkan algoritma Apriori
def run_apriori(data, min_support, min_confidence):
    records = []
    for i in range(len(data)):
        records.append([str(data.values[i, j]) for j in range(data.shape[1])])

    # Menjalankan algoritma Apriori
    results = list(apriori(records, min_support=min_support, min_confidence=min_confidence))

    return results

# Fungsi untuk menampilkan hasil analisis asosiasi
def display_results(results):
    st.header("Hasil Analisis Asosiasi menggunakan Algoritma Apriori")
    for rule in results:
        st.write(f"Rule: {', '.join(rule.items)}")
        st.write(f"Support: {rule.support}")
        st.write(f"Confidence: {rule.confidence}")
        st.write(f"Lift: {rule.lift}")
        st.write("---")

def main():
    st.title("Aplikasi Analisis Asosiasi dengan Algoritma Apriori")

    # Memuat data
    data = load_data()

    # Menampilkan data
    st.subheader("Data:")
    st.write(data)

    # Parameter untuk algoritma Apriori
    min_support = st.slider("Minimum Support", 0.0, 1.0, 0.1)
    min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.5)

    # Menjalankan algoritma Apriori
    results = run_apriori(data, min_support, min_confidence)

    # Menampilkan hasil analisis
    display_results(results)

if __name__ == "__main__":
    main()
