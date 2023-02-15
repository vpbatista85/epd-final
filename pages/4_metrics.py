import streamlit as st
import utils





st.title('Métricas de Acurácia')
tab1, tab2, tab3 = st.tabs(["Classificação","Personalização e Ranqueamento","Feedback"])

with tab1:
    utils.plot_report(st.session_state.classification_report, figsize=(16,10))
with tab2:
    utils.plot_report(st.session_state.ranking_report, figsize=(16,10))
with tab3:
    utils.plot_report(st.session_state.rating_report, figsize=(16,10))

st.session_state.classification_report