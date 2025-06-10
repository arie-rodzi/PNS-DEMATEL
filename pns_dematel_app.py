# Pythagorean Neutrosophic DEMATEL Streamlit App
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Pythagorean Neutrosophic DEMATEL App")

# ----------------------------
# Utility Functions
# ----------------------------
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

# ----------------------------
# File Upload
# ----------------------------
uploaded_file = st.file_uploader("Upload Excel File (.xlsx) with Each Expert as a Sheet", type=["xlsx"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheets = xls.sheet_names
    st.write(f"Detected expert sheets: {sheets}")

    matrices = []
    for sheet in sheets:
        df = pd.read_excel(xls, sheet_name=sheet, index_col=0)
        n = df.shape[0]
        pns_matrix = np.zeros((n, n, 3))
        for i in range(n):
            for j in range(n):
                pns_matrix[i, j] = numeric_to_pns(df.iloc[i, j])
        matrices.append(pns_matrix)

    avg_pns = np.mean(matrices, axis=0)  # shape (n, n, 3)

    # Display Average PNS Matrix
    st.subheader("Average Rating in Pythagorean Neutrosophic Form")
    avg_pns_df = pd.DataFrame(
        [[f"({T:.2f}, {I:.2f}, {F:.2f})" for T, I, F in row] for row in avg_pns],
        index=df.index, columns=df.columns
    )
    st.dataframe(avg_pns_df)

    # Deneutrosophication to get crisp matrix
    crisp_matrix = np.array([[deneutrosophication(*avg_pns[i, j][:2]) for j in range(n)] for i in range(n)])

    # Normalize
    max_val = np.max(np.sum(crisp_matrix, axis=1))
    norm_matrix = crisp_matrix / max_val

    # Total relation matrix: T = Z(I - Z)^-1
    I = np.identity(n)
    T = norm_matrix @ np.linalg.inv(I - norm_matrix)

    R = np.sum(T, axis=1)  # sum of rows
    C = np.sum(T, axis=0)  # sum of columns

    cause_effect = R - C
    prominence = R + C

    result_df = pd.DataFrame({
        "Criteria": df.index,
        "R": R,
        "C": C,
        "R+C": prominence,
        "R-C": cause_effect,
        "Type": ["Cause" if val > 0 else "Effect" for val in cause_effect]
    })

    st.subheader("Ranking Result")
    st.dataframe(result_df.sort_values("R+C", ascending=False))

    # NRM thresholding
    threshold = np.mean(T)
    NRM = (T > threshold).astype(int)

    st.subheader("Normalized Relation Matrix (NRM) > Threshold = 1")
    st.dataframe(pd.DataFrame(NRM, index=df.index, columns=df.columns))

    # Show Directional Arrows (Cause to Effect) for NRM
    st.subheader("Cause and Effect Directions (Arrow Map)")
    arrow_map = []
    for i in range(n):
        for j in range(n):
            if NRM[i, j] == 1:
                arrow_map.append(f"{df.index[i]} → {df.columns[j]}")
    st.write("\n".join(arrow_map))

    # Plot full cause-effect diagram
    st.subheader("Full Cause-Effect Diagram")
    fig, ax = plt.subplots()
    ax.scatter(cause_effect, prominence)
    for i, name in enumerate(df.index):
        ax.text(cause_effect[i], prominence[i], name)
    ax.set_xlabel("R - C (Cause-Effect)")
    ax.set_ylabel("R + C (Prominence)")
    ax.grid(True)
    st.pyplot(fig)

    # Individual plots for each criterion
    st.subheader("Individual Criterion Influence")
    for i, name in enumerate(df.index):
        fig, ax = plt.subplots()
        ax.bar(range(n), T[i], tick_label=df.columns)
        ax.set_title(f"Influence of {name} on Others")
        st.pyplot(fig)
