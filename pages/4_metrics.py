import streamlit as st
import utils





st.title('Métricas de Negócio')
tab1, tab2 = st.tabs(["Cobertura","Personalização e Ranqueamento"])

with tab1:
    utils.plot_report(st.session_state.coverage_report, figsize=(16,10))
with tab2:
    utils.plot_report(st.session_state.ranking_report, figsize=(16,10))

st.title('Métricas de Acurácia')
tab3, tab4= st.tabs(["Classificação","Feedback"])

with tab3:
    utils.plot_report(st.session_state.classification_report, figsize=(16,10))

with tab4:
    utils.plot_report(st.session_state.rating_report, figsize=(16,10))

