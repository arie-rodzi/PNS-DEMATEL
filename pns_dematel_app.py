# Export the updated Streamlit app including visualizations similar to Figures 2, 3, and 4
app_file_full_visuals = "/mnt/data/pns_dematel_full_visuals_app.py"

app_code_full_visuals = '''
# Pythagorean Neutrosophic DEMATEL Streamlit App with Visuals like Figures 2, 3, 4
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

st.set_page_config(layout="wide")
st.title("Pythagorean Neutrosophic DEMATEL with Visual Cause-Effect Mapping")

def numeric_to_pns(val):
    mapping = {
        1: (0.10, 0.80, 0.90),
        2: (0.30, 0.70, 0.80),
        3: (0.50, 0.60, 0.70),
        4: (0.70, 0.50, 0.50),
        5: (0.80, 0.30, 0.30),
        6: (0.90, 0.20, 0.10),
        7: (1.00, 0.10, 0.00),
    }
    return mapping.get(int(val), (0.0, 0.0, 0.0))

def deneutrosophication(T, I):
    return (T + 0.5 * I) / 1.5

uploaded_file = st.file_uploader("Upload Excel File (.xlsx) with each expert on a separate sheet", type=["xlsx"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheets = xls.sheet_names
    st.write("Detected expert sheets:", sheets)

    matrices = []
    for sheet in sheets:
        df = pd.read_excel(xls, sheet_name=sheet, index_col=0)
        n = df.shape[0]
        pns_matrix = np.zeros((n, n, 3))
        for i in range(n):
            for j in range(n):
                pns_matrix[i, j] = numeric_to_pns(df.iloc[i, j])
        matrices.append(pns_matrix)

    avg_pns = np.mean(matrices, axis=0)

    st.subheader("Average Rating in Pythagorean Neutrosophic Form (T, I, F)")
    pns_df = pd.DataFrame([[f"({T:.2f}, {I:.2f}, {F:.2f})" for T, I, F in row] for row in avg_pns],
                          index=df.index, columns=df.columns)
    st.dataframe(pns_df)

    crisp_matrix = np.array([[deneutrosophication(*avg_pns[i, j][:2]) for j in range(n)] for i in range(n)])
    max_val = np.max(np.sum(crisp_matrix, axis=1))
    norm_matrix = crisp_matrix / max_val
    I = np.identity(n)
    T = norm_matrix @ np.linalg.inv(I - norm_matrix)

    R = np.sum(T, axis=1)
    C = np.sum(T, axis=0)
    RC = R + C
    RmC = R - C

    result_df = pd.DataFrame({
        "Criteria": df.index,
        "R": R, "C": C,
        "R+C": RC,
        "R-C": RmC,
        "Type": ["Cause" if x > 0 else "Effect" for x in RmC]
    })

    st.subheader("Figure 2: Ranking of Criteria by R+C and R−C")
    st.dataframe(result_df.sort_values("R+C", ascending=False))

    st.subheader("Figure 3: Overall Network Relation Map (NRM)")

    threshold = np.mean(T)
    NRM = (T > threshold).astype(int)
    nrm_df = pd.DataFrame(NRM, index=df.index, columns=df.columns)
    st.dataframe(nrm_df)

    # Plot NRM arrows (Figure 3)
    G = nx.DiGraph()
    for i in range(n):
        for j in range(n):
            if NRM[i, j] == 1:
                G.add_edge(df.index[i], df.columns[j])
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=1500, node_color='lightblue', arrows=True, ax=ax3)
    st.pyplot(fig3)

    st.subheader("Figure 4: Individual Influence Maps")

    for i, name in enumerate(df.index):
        fig, ax = plt.subplots()
        influenced = [df.columns[j] for j in range(n) if NRM[i, j] == 1]
        scores = [T[i, j] for j in range(n) if NRM[i, j] == 1]
        ax.bar(influenced, scores)
        ax.set_title(f"Influence of {name} on Others")
        ax.set_ylabel("Influence Strength")
        st.pyplot(fig)
'''

with open(app_file_full_visuals, "w") as f:
    f.write(app_code_full_visuals)

app_file_full_visuals
