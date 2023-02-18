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

st.subheader('Métricas de Negócio')
tab1, tab2, tab3, tab4 = st.tabs(["Cobertura","Ranqueamento","Personalização","Ticket Médio Mensal"])

with tab1:
    utils.plot_report(st.session_state.coverage_report, model_s,n=n ,figsize=(16,10))
with tab2:
    utils.plot_report(st.session_state.ranking_report, model_s,n=n , figsize=(16,10))
with tab3:
    utils.plot_report(st.session_state.personalization_report, model_s,n=n , figsize=(16,10))    
with tab4:
    utils.tmv(st.session_state.df)      

st.subheader('Métricas de Acurácia')
tab5, tab6, tab7= st.tabs(["Classificação","Feedback","Run Time"])

with tab5:
    utils.plot_report(st.session_state.classification_report, model_s,n=n , figsize=(16,20))

with tab6:
    utils.plot_report(st.session_state.rating_report, model_s,n=n , figsize=(16,10))

with tab7:
    utils.plot_runtime_metrics(st.session_state.df_lrecnp)

