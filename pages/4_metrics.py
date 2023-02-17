import streamlit as st
import utils

if 'n' not in st.session_state:
    n=10

if 'model_s' not in st.session_state:
    model_s=[]

 
with st.sidebar:

    n=st.slider('Selecione o número de itens a ser exibido',min_value=1,max_value=20,value=10,step=1,disabled=False)
    st.write('Comparar os modelos:')
    if st.checkbox('Top N', value=True, disabled=False):
        model_s.append('top')
    if st.checkbox('Co-visitation', value=True, disabled=False):
        model_s.append('covisitation')
    if st.checkbox('Content Based', value=True, disabled=False):
        model_s.append('contentbased')
    if st.checkbox('Item KNN', value=True, disabled=False):
        model_s.append('itemknn')
    if st.checkbox('Funk-SVD', value=True, disabled=False):
        model_s.append('svd')
    if st.checkbox('LightFM', value=True, disabled=False):
        model_s.append('lightfm')    

st.header(f'Loja: {st.session_state.store}')

col1, col2, = st.columns([1,3])
with col1:    
    utils.tmv(st.session_state.df)

with col2:   
    st.subheader('Métricas de Negócio')
    tab1, tab2, tab3 = st.tabs(["Cobertura","Ranqueamento","Personalização"])

    with tab1:
        utils.plot_report(st.session_state.coverage_report, model_s,n=n ,figsize=(16,10))
    with tab2:
        utils.plot_report(st.session_state.ranking_report, model_s,n=n , figsize=(16,10))
    with tab3:
        utils.plot_report(st.session_state.personalization_report, model_s,n=n , figsize=(16,10))    

    st.subheader('Métricas de Acurácia')
    tab4, tab5= st.tabs(["Classificação","Feedback"])

    with tab4:
        utils.plot_report(st.session_state.classification_report, model_s,n=n , figsize=(16,20))

    with tab5:
        utils.plot_report(st.session_state.rating_report, model_s,n=n , figsize=(16,10))

