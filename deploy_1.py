import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
import plotly.express as px
import seaborn as sns


def SHOW_VISU_PAGE():
    url = 'https://raw.githubusercontent.com/merrooo/ML_DATA/main/concrete_data.csv'
    df=pd.read_csv(url)
    x=df.loc[:,df.columns != 'Strength']
    y=df['Strength']
   #------------------------------------------------------------------
    st.title("""### VISUALIZATION """)
   #------------------------------------------------------------------
    st.write("AREA_CHART_DEPENDENT FEATURES (CEMENT_AGE_SUPERPLASTICIZER) AFFECTING ON STRENGTH")
    button_VISU_2=st.button("AREA_CHART(CEMENT_AGE_SUPERPLASTICIZER)",type="primary")
    if button_VISU_2:
     st.area_chart(
     df, x="Strength", y=["Cement", "Superplasticizer","Age"], color=["#f0e936", "#4633f2","#0e6210"]) # Optional
   #------------------------------------------------------------------
    st.write("DISTRIBUTION_PLOTTING THE DEPENDENT FEATURES (CEMENT_AGE_SUPERPLASTICIZER) AFFECTING ON STRENGTH")
    button_VISU_3=st.button("DIST_(CEMENT_AGE_SUPERPLASTICIZER)",type="primary")
    if button_VISU_3:
      x1 = np.random.randn(200) - 2
      x2 = np.random.randn(200)
      x3 = np.random.randn(200) + 2
      hist_data = [x1, x2, x3]
      group_labels = ['Cement', 'Age', 'Superplasticizer']
      fig = ff.create_distplot(
        hist_data, group_labels, bin_size=[.1, .25, .5])
      st.plotly_chart(fig, use_container_width=True)
   #------------------------------------------------------------------
    st.write("DISTRIBUTION_PLOTTING CEMENT REGARDING TO STRENGTH ")
    button_VISU_4=st.button("DIST_FEATURE_(Strength_MPa)",type="primary")
    if button_VISU_4:
      x1 = np.random.randn(1000) - 2
      x2 = np.random.randn(1000)
      x3 = np.random.randn(1000) + 2
      hist_data = [x3]
      group_labels = ['Strength']
      fig = ff.create_distplot(
          hist_data, group_labels, bin_size=[.1, .25, .5])
      st.plotly_chart(fig, use_container_width=True)
   #------------------------------------------------------------------
    st.write("DISTRIBUTION_PLOTTING CEMENT REGARDING TO STRENGTH ")
    button_VISU_5=st.button("DIST_FEATURE_(Cement)",type="primary")
    if button_VISU_5:
     x1 = np.random.randn(1000) - 2
     x2 = np.random.randn(1000)
     x3 = np.random.randn(1000) + 2
     hist_data = [x3]
     group_labels = ['Cement']
     fig = ff.create_distplot(
          hist_data, group_labels, bin_size=[.1, .25, .5])
     st.plotly_chart(fig, use_container_width=True)
   #------------------------------------------------------------------
    st.write("DISTRIBUTION_PLOTTING CEMENT REGARDING TO STRENGTH ")
    button_VISU_6=st.button("DIST_FEATURE_(Age)",type="primary")
    if button_VISU_6:
      x1 = np.random.randn(1000) - 2
      x2 = np.random.randn(1000)
      x3 = np.random.randn(1000) + 2
      hist_data = [x3]
      group_labels = ['Age']
      fig = ff.create_distplot(
         hist_data, group_labels, bin_size=[.1, .25, .5])
      st.plotly_chart(fig, use_container_width=True)
   #------------------------------------------------------------------
    st.write("DISTRIBUTION_PLOTTING CEMENT REGARDING TO STRENGTH ")
    button_VISU_7=st.button("DIST_FEATURE_(Superplasticizer)",type="primary")
    if button_VISU_7:
      x1 = np.random.randn(1000) - 2
      x2 = np.random.randn(1000)
      x3 = np.random.randn(1000) + 2
      hist_data = [x3]
      group_labels = ['Superplasticizer']
      fig = ff.create_distplot(
          hist_data, group_labels, bin_size=[.1, .25, .5])
      st.plotly_chart(fig, use_container_width=True)
   #------------------------------------------------------------------
    st.write("HEAT_MAP(DATA_FRAME)")
    button_VISU_8=st.button("HEAT_MAP(DATA_FRAME)",type="primary")
    if button_VISU_8:
      fig = px.imshow(df, text_auto=True, aspect="auto")
      st.plotly_chart(fig)
   #------------------------------------------------------------------
    st.write("HEAT_MAP(CEMENT_STRENGTH)")
    button_VISU_9=st.button("HEAT_MAP(CEMENT_STRENGTH)",type="primary")
    if button_VISU_9:
      fig = px.density_heatmap(df, x="Cement", y="Strength", text_auto=True, nbinsx=7, color_continuous_scale='turbid_r', width=686, height=889)
      st.plotly_chart(fig)
   #------------------------------------------------------------------
    st.write("HEAT_MAP(AGE_STRENGTH)")
    button_VISU_10=st.button("HEAT_MAP(AGE_STRENGTH)",type="primary")
    if button_VISU_10:
      fig = px.density_heatmap(df, x="Age", y="Strength", text_auto=True, nbinsx=7, color_continuous_scale='turbid_r', width=686, height=889)
      st.plotly_chart(fig)
   #------------------------------------------------------------------
    st.write("HEAT_MAP(Superplasticizer_STRENGTH)")
    button_VISU_11=st.button("HEAT_MAP(Superplasticizer_STRENGTH)",type="primary")
    if button_VISU_11:
      fig = px.density_heatmap(df, x="Superplasticizer", y="Strength", text_auto=True, nbinsx=7, color_continuous_scale='turbid_r', width=686, height=889)
      st.plotly_chart(fig)
   #------------------------------------------------------------------
    st.write("HEAT_MAP(Fly Ash_STRENGTH)")
    button_VISU_12=st.button("HEAT_MAP(Fly Ash_STRENGTH)",type="primary")
    if button_VISU_12:
      fig = px.density_heatmap(df, x="Fly Ash", y="Strength", text_auto=True, nbinsx=7, color_continuous_scale='turbid_r', width=686, height=889)
      st.plotly_chart(fig)
   #------------------------------------------------------------------
    st.write("HEAT_MAP(Superplasticizer_STRENGTH)")
    button_VISU_13=st.button("HEAT_MAP(Water_STRENGTH)",type="primary")
    if button_VISU_13:
      fig = px.density_heatmap(df, x="Water", y="Strength", text_auto=True, nbinsx=7, color_continuous_scale='turbid_r', width=686, height=889)
      st.plotly_chart(fig)
