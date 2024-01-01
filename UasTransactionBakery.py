import streamlit as st
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def draw_graph(rules, rules_to_show):
    G1 = nx.DiGraph()

    color_map = []
    N = 50
    colors = np.random.rand(N)
    strs = ['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11']

    for i in range(rules_to_show):
        G1.add_nodes_from(["R" + str(i)])

        for a in rules.iloc[i]['antecedents']:
            G1.add_nodes_from([a])
            G1.add_edge(a, "R" + str(i), color=colors[i], weight=2)

        for c in rules.iloc[i]['consequents']:
            G1.add_nodes_from([a])
            G1.add_edge("R" + str(i), c, color=colors[i], weight=2)

    for node in G1:
        found_a_string = False
        for item in strs:
            if node == item:
                found_a_string = True
        if found_a_string:
            color_map.append('yellow')
        else:
            color_map.append('green')

    edges = G1.edges()
    colors = [G1[u][v]['color'] for u, v in edges]
    weights = [G1[u][v]['weight'] for u, v in edges]

    pos = nx.spring_layout(G1, k=16, scale=1)
    nx.draw(G1, pos, node_color=color_map, edge_color=colors, width=weights, font_size=16, with_labels=False)

    for p in pos:  # raise text positions
        pos[p][1] += 0.07
    nx.draw_networkx_labels(G1, pos)
    plt.show()

# Sample data
data = {
    'antecedents': [['A1'], ['A2'], ['A3'], ['A1', 'A2'], ['A2', 'A3'], ['A1', 'A3'], ['A1', 'A2', 'A3'], ['A1', 'A2'],
                    ['A2', 'A3'], ['A1', 'A3']],
    'consequents': [['C1'], ['C2'], ['C3'], ['C1'], ['C2'], ['C3'], ['C1'], ['C2', 'C3'], ['C1', 'C2', 'C3'], ['C1', 'C2']]
}

rules = pd.DataFrame(data)

# Streamlit app
def main():
    st.title("Rule Graph Visualization")
    
    # Sidebar
    rules_to_show = st.sidebar.slider("Number of Rules to Show", 1, len(rules), 5)

    # Main content
    st.write("### Rule Graph:")
    draw_graph(rules, rules_to_show)

if __name__ == "__main__":
    main()
