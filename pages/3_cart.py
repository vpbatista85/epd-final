import streamlit as st
from datetime import datetime,  timedelta, time
# import pandas as pd
# import numpy as np
# import sklearn
# import mlxtend
# import os
# from collections import Counter
# import networkx as nx
# import matplotlib.pyplot as plt
# import seaborn as sns
import time as tm
import utils


if 'clock' not in st.session_state:
    st.session_state.clock =(datetime.now()-timedelta(hours=3)).time()


if len(st.session_state.l_prod)==0:
        state=True
else:
        state=False


utils.f_carrinho()

col1, col2, col3, col4, col5= st.columns([1,1,1,1,1])

with col1:
    if st.button('Del item',disabled=state):
        if len(st.session_state.l_prod)==1:
            st.write (f"{st.session_state.l_prod.values}, removido do carrinho.")
            st.session_state.l_prod=[]
            state=True
            #placeholder.empty()
        else:    
            st.write (f"{st.session_state.l_prod[-1]}, removido do carrinho.")
            st.session_state.l_prod.pop()
            #placeholder.empty()  
            placeholder.text("Carrinho:")
            with placeholder.container():
                st.write('Carrinho:')
                for i in st.session_state.l_prod[0:-2]:
                    st.write(i)

with col2:
    if st.button('Del carrinho',disabled=state):
        st.session_state.l_prod=[]
        st.write (f"Carrinho limpo.") 
        #placeholder.empty()




df_loja_rec=st.session_state.df_lrecnp
# df_loja_rec['dth_hora'] = df_loja_rec['dth_agendamento'].apply(utils.extract_hour)
# df_loja_rec=utils.time_filter(df_loja_rec)


with st.sidebar:

    st.write('Simulação de periodo:')
    
    horario=st.checkbox('Horario atual', value=True, key=None, help='Marque para usar o horário local', on_change=None, args=None, kwargs=None, disabled=False)
    if horario:
       st.write("Relógio:",datetime.strptime(str(datetime.now()-timedelta(hours=3)),"%Y-%m-%d %H:%M:%S.%f").strftime("%H:%M"))
       st.slider('Selecione o horário',min_value=time.min,max_value=time.max,value=st.session_state.clock,format="HH:MM",step=timedelta(minutes=60),disabled=True)
       st.session_state.clock =(datetime.now()-timedelta(hours=3)).time()
       #hora=st.session_state.clock
       st.write('Horário adotado:',st.session_state.clock)
      
    else:
        st.write("Relógio:",datetime.strptime(str(datetime.now()-timedelta(hours=3)),"%Y-%m-%d %H:%M:%S.%f").strftime("%H:%M"))
        st.session_state.clock=st.slider('Selecione o horário',min_value=time.min,max_value=time.max,value=st.session_state.clock,format="HH:MM",step=timedelta(minutes=60),disabled=False)
        #hora=st.session_state.clock
        st.write('Horário adotado:',st.session_state.clock)
    df_loja_af=df_loja_rec.copy()
    df_loja_af['dth_hora']=df_loja_rec['dth_agendamento'].apply(utils.extract_hour)
    df_loja_filtrado=utils.time_filter(df_loja_af,hr=st.session_state.clock,nh=1)
    st.write ('Quantidade de linhas antes do filtro de horário',df_loja_af.shape[0])
    st.write ('Quantidade de linhas apos o filtro de horário',df_loja_filtrado.shape[0])

if  'df_loja_af' not in st.session_state:
    st.session_state.df_loja_af=df_loja_rec.copy()

utils.r_np(df_loja_rec,st.session_state.l_prod,n=5,h=st.session_state.clock)
utils.r_p(df_loja_rec,st.session_state.l_prod,st.session_state.user,n=5,h=st.session_state.clock)

coverage_report, ranking_report, classification_report, rating_report= utils.master_m(df_loja_rec,'https://github.com/vpbatista85/epd-final/tree/test/',st.session_state.store)

if 'coverage_report'not in st.session_state:
    st.session_state.coverage_report=coverage_report

if 'ranking_report'not in st.session_state:
    st.session_state.ranking_report=ranking_report

if 'classification_report'not in st.session_state:
    st.session_state.classification_report=classification_report

if 'rating_report'not in st.session_state:
    st.session_state.rating_report=rating_report

if st.button('Métricas'):
        utils.add_page('teste_strealit_main.py', 'metrics')
        utils.nav_page('metrics')

