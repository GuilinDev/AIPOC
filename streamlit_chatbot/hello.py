import streamlit as st
import pandas as pd
import numpy as np

st.write("Hello, world!")
st.write("## This is a H2 Title")

x = st.text_input("Movie", "Titanic")

if st.button("Click me"):
    st.write(f"your favorite Movie: {x}")

data = pd.read_csv("movies.csv")
st.write(data)

chart_data = pd.DataFrame(
    pd.DataFrame(np.random.randn(20, 3),
                 columns=["a", "b", "c"])
)

st.bar_chart(chart_data)
