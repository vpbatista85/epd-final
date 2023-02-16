import streamlit as st
import utils

if 'model_s' not in st.session_state:
    model_s=[]

with st.sidebar:

    st.write('Comparar os modelos:')
    if st.checkbox('Top N', value=True, disabled=False):
        model_s.append('top')
    if st.checkbox('Co-visitation', value=True, disabled=False):
        model_s.append('covisitation')
    if st.checkbox('Item KNN', value=True, disabled=False):
        model_s.append('itemknn')
    if st.checkbox('Funk-SVD', value=True, disabled=False):
        model_s.append('svd')
    if st.checkbox('LightFM', value=True, disabled=False):
        model_s.append('lightfm')       


st.title('Métricas de Negócio')
tab1, tab2, tab3 = st.tabs(["Cobertura","Ranqueamento","Personalização"])

with tab1:
    utils.plot_report(st.session_state.coverage_report, model_s, figsize=(16,10))
with tab2:
    utils.plot_report(st.session_state.ranking_report, model_s, figsize=(16,10))
with tab3:
    utils.plot_report(st.session_state.personalization_report, model_s, figsize=(16,10))    

st.title('Métricas de Acurácia')
tab4, tab5= st.tabs(["Classificação","Feedback"])

with tab4:
    utils.plot_report(st.session_state.classification_report, model_s, figsize=(16,20))

with tab5:
    utils.plot_report(st.session_state.rating_report, model_s, figsize=(16,10))

