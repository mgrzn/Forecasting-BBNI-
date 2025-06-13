import streamlit as st

pages = {
    "Machine Learning Project Overview" :[
        st.Page("dashboard.py"),
        st.Page("forecasting.py"),
    ],
}

pg = st.navigation(pages)
pg.run()