import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

def SHOW_DATA_PAGE():

    url = 'https://raw.githubusercontent.com/merrooo/ML_DATA/main/concrete_data.csv'
    df=pd.read_csv(url)
    x=df.loc[:,df.columns != 'Strength']
    y=df['Strength']
    #------------------------------------------------------------------
    st.title('EXPLORING_DATA_FRAME!!')
    #------------------------------------------------------------------
    EDA_=st.sidebar.selectbox("EXPLORING_DATA_COLUMNS",("DATA_FARME_","COLUMNS_"))

    if EDA_ == "DATA_FARME_":
      EXPLORE_DATA_=st.toggle('EXPLORE_DATA',disabled=False)
      CHECK_NULL_=st.toggle('CHECK_NULL',disabled=False)
      DESCRIBE_=st.toggle('DESCRIBE',disabled=False)
      COLUMNS_=st.toggle('COLUMNS',disabled=False)
      DUPLICATED_=st.toggle('DUPLICATED',disabled=False)
      MAX_=st.toggle('MAXIMUM_VALUES_FEATURES',disabled=False)
      HEAD_=st.toggle('DATA_HEAD',disabled=False)
      if EXPLORE_DATA_:
        st.write('EXPLORING_DATA_FRAM_OF_CONCRETE!!')
        st.dataframe(df)
        st.write('DONE!!')
      elif MAX_:
        st.write('MAXIMUM_VALUES_FEATURES!!')
        data_matrix = [['ITEM', 'DESCIBTION'],
               ['Cement_KG IN M3',540.0],
               ['Blast Furnace Slag_KG IN M3', 359.4],
               ['Fly Ash_KG IN M3', 200.1],
               ['Water_KG IN M3', 247.0],
               ['Superplasticizer IN M3', 32.2],
               ['Coarse Aggregate_KG IN M3', 1145.0],
               ['Fine Aggregate_KG IN M3', 992.6],
               ['Age_Day (1~365)', 365],
               ['Strength_MPa', 82.6]]
        fig = ff.create_table(data_matrix)
        st.plotly_chart(fig)
        st.write('DONE!!')
      elif HEAD_:
        st.write('DATA_HEAD!!')
        st.dataframe(df.head())
        st.write('DONE!!')
      elif CHECK_NULL_:
        st.write('EXPLORING_NULL!!')
        st.dataframe(df.isnull().sum())
        st.write('DONE!!')
      elif DESCRIBE_:
        st.write('DESCRIPE_OF_DATA!!')
        st.dataframe(df.describe())
        st.write('DONE!!')
      elif COLUMNS_:
        st.write('EXPLORE_HOW_MANY_FEATURES!!')
        st.dataframe(df.columns)
        st.write('DONE!!')
      elif DUPLICATED_:
        st.write('CHECK_DUPLICATED_OF_DATA!!')
        st.dataframe(df.duplicated().sum())
        st.write('DONE!!')
    elif EDA_ == "COLUMNS_":
      st.write('VALUE_COUNTS_FOR_OUT_PUT[STRENGTH]!!')
      VALUE_COUNTS_=st.toggle('VALUE_COUNTS',disabled=False)
      UNIQUE_=st.toggle('UNIQUE',disabled=False)
      if VALUE_COUNTS_:
        st.write('CHECK_FOR_OUTPUT_[STRENGTH]!!')
        Strength_=st.toggle('Strength',disabled=False)
        if Strength_:
         st.dataframe(df['Strength'].value_counts())
         st.write('DONE!!')
      elif UNIQUE_:
        st.write('CHECK_FOR_OUTPUT_[STRENGTH]!!')
        Strength_=st.toggle('Strength',disabled=False)
        if Strength_:
         st.dataframe(df['Strength'].unique())
         st.write('DONE!!')
