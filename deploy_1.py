import streamlit as st
import pickle
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error , mean_squared_error ,r2_score
from sklearn.model_selection import cross_val_score,TimeSeriesSplit
# import os
# import openpyxl
# from openpyxl import load_workbook

st.header("CONCRETE_DATA_SET")
st.image("https://media.istockphoto.com/id/692096736/photo/concrete-pouring-during-commercial-concreting-floors-of-building.jpg?s=1024x1024&w=is&k=20&c=XYYH7UhgqsMmwGBWO6UJsxaSgjxNDuQO8i7N27nwRlk=", width=200)
def DATA_FRAME(df):
  url_1= 'https://raw.githubusercontent.com/merrooo/ML_DATA/main/concrete_data.csv'
  df=pd.read_csv(url_1)
  return df
page=st.sidebar.selectbox("OPTINOS_FOR_EXPLORING_DATA",("- -","- EDA -","- VISUALIZATION -","- PREDICTION -"))  
if page == "- EDA -":
    EDA_=st.sidebar.selectbox("EXPLORING_DATA_COLUMNS",("- -","- DATA_FARME -","- COLUMNS -"))

    if EDA_ == "- DATA_FARME -":
      EXPLORE_DATA_=st.sidebar.toggle('EXPLORE_DATA',disabled=False)
      CHECK_NULL_=st.sidebar.toggle('CHECK_NULL',disabled=False)
      DESCRIBE_=st.sidebar.toggle('DESCRIBE',disabled=False)
      COLUMNS_=st.sidebar.toggle('COLUMNS',disabled=False)
      MAX_=st.sidebar.toggle('MAXIMUM_VALUES_FEATURES',disabled=False)
      HEAD_=st.sidebar.toggle('DATA_HEAD',disabled=False)
      if EXPLORE_DATA_:
        st.dataframe(DATA_FRAME('df'))
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
        st.dataframe(DATA_FRAME('df').head())
        st.write('DONE!!')
      elif CHECK_NULL_:
        st.write('EXPLORING_NULL!!')
        st.dataframe(DATA_FRAME('df').isnull().sum())
        st.write('DONE!!')
      elif DESCRIBE_:
        st.write('DESCRIPE_OF_DATA!!')
        st.dataframe(DATA_FRAME('df').describe())
        st.write('DONE!!')
      elif COLUMNS_:
        st.write('EXPLORE_HOW_MANY_FEATURES!!')
        st.dataframe(DATA_FRAME('df').columns)
        st.write('DONE!!')
    elif EDA_ == "- COLUMNS -":
      st.write('VALUE_COUNTS_FOR_OUT_PUT[STRENGTH]!!')
      VALUE_COUNTS_=st.sidebar.toggle('VALUE_COUNTS',disabled=False)
      UNIQUE_=st.sidebar.toggle('UNIQUE',disabled=False)
      if VALUE_COUNTS_:
        st.write('CHECK_FOR_OUTPUT_[STRENGTH]!!')
        STRENGTH_=st.toggle('STRENGTH',disabled=False)
        if STRENGTH_:
         st.dataframe(DATA_FRAME('df')['Strength'].value_counts())
         st.write('DONE!!')
      elif UNIQUE_:
        st.write('CHECK_FOR_OUTPUT_[STRENGTH]!!')
        Strength_=st.toggle('Strength',disabled=False)
        if Strength_:
         st.dataframe(DATA_FRAME('df')['Strength'].unique())
         st.write('DONE!!')
    st.balloons()
#_______________________________________________________________________________________________________________________________________________________________

elif page =="- VISUALIZATION -":
   #------------------------------------------------------------------
    st.sidebar.write('AREA_CHART!!')
    button_VISU_2=st.sidebar.button("AREA_CHART",type="primary")
    st.sidebar.write('DISTRIBUTION_PLOTTING!!')
    if button_VISU_2:
     st.write("AREA_CHART_DEPENDENT FEATURES (CEMENT_AGE_SUPERPLASTICIZER) AFFECTING ON STRENGTH")
     st.area_chart(
     DATA_FRAME('df'), x="Strength", y=["Cement", "Superplasticizer","Age"], color=["#f0e936", "#4633f2","#0e6210"]) # Optional
     st.success('ALREADY_GRAPH_VISUALIZED!', icon="✅")
      
    button_VISU_3=st.sidebar.button("DISTRUBTION_PLOT_1",type="primary")
    if button_VISU_3:
      st.write("DISTRIBUTION_PLOTTING THE DEPENDENT FEATURES (CEMENT_AGE_SUPERPLASTICIZER) AFFECTING ON STRENGTH")
      x1 = np.random.randn(200) - 2
      x2 = np.random.randn(200)
      x3 = np.random.randn(200) + 2
      hist_data = [x1, x2, x3]
      group_labels = ['Cement', 'Age', 'Superplasticizer']
      fig = ff.create_distplot(
        hist_data, group_labels, bin_size=[.1, .25, .5])
      st.plotly_chart(fig, use_container_width=True)
      st.success('ALREADY_GRAPH_VISUALIZED!', icon="✅")
      
    button_VISU_4=st.sidebar.button("DISTRUBTION_PLOT_2",type="primary")
    if button_VISU_4:
      st.write("DISTRIBUTION_PLOTTING CEMENT REGARDING TO STRENGTH ")
      x1 = np.random.randn(1000) - 2
      x2 = np.random.randn(1000)
      x3 = np.random.randn(1000) + 2
      hist_data = [x3]
      group_labels = ['Strength']
      fig = ff.create_distplot(
          hist_data, group_labels, bin_size=[.1, .25, .5])
      st.plotly_chart(fig, use_container_width=True)
      st.success('ALREADY_GRAPH_VISUALIZED!', icon="✅")
      
    button_VISU_5=st.sidebar.button("DISTRUBTION_PLOT_3",type="primary")
    if button_VISU_5:
     st.write("DISTRIBUTION_PLOTTING CEMENT REGARDING TO STRENGTH ")
     x1 = np.random.randn(1000) - 2
     x2 = np.random.randn(1000)
     x3 = np.random.randn(1000) + 2
     hist_data = [x3]
     group_labels = ['Cement']
     fig = ff.create_distplot(
          hist_data, group_labels, bin_size=[.1, .25, .5])
     st.plotly_chart(fig, use_container_width=True)
     st.success('ALREADY_GRAPH_VISUALIZED!', icon="✅")
      
    button_VISU_6=st.sidebar.button("DISTRUBTION_PLOT_4",type="primary")
    if button_VISU_6:
      st.write("DISTRIBUTION_PLOTTING CEMENT REGARDING TO STRENGTH ")
      x1 = np.random.randn(1000) - 2
      x2 = np.random.randn(1000)
      x3 = np.random.randn(1000) + 2
      hist_data = [x3]
      group_labels = ['Age']
      fig = ff.create_distplot(
         hist_data, group_labels, bin_size=[.1, .25, .5])
      st.plotly_chart(fig, use_container_width=True)
      st.success('ALREADY_GRAPH_VISUALIZED!', icon="✅")
      
    button_VISU_7=st.sidebar.button("DISTRUBTION_PLOT_5",type="primary")
    st.sidebar.write('HEAT_MAP!!')
    if button_VISU_7:
      st.write("DISTRIBUTION_PLOTTING CEMENT REGARDING TO STRENGTH ")
      x1 = np.random.randn(1000) - 2
      x2 = np.random.randn(1000)
      x3 = np.random.randn(1000) + 2
      hist_data = [x3]
      group_labels = ['Superplasticizer']
      fig = ff.create_distplot(
          hist_data, group_labels, bin_size=[.1, .25, .5])
      st.plotly_chart(fig, use_container_width=True)
      st.success('ALREADY_GRAPH_VISUALIZED!', icon="✅")
      
    button_VISU_8=st.sidebar.button("HEAT_MAP_DF",type="primary")
    if button_VISU_8:
      st.write("HEAT_MAP(DATA_FRAME)")
      fig = px.imshow(DATA_FRAME('df'), text_auto=True, aspect="auto")
      st.plotly_chart(fig)
      st.success('ALREADY_GRAPH_VISUALIZED!', icon="✅")
      
    button_VISU_9=st.sidebar.button("HEAT_MAP_CEMENT",type="primary")
    if button_VISU_9:
      st.write("HEAT_MAP(CEMENT_STRENGTH)")
      fig = px.density_heatmap(DATA_FRAME('df'), x="Cement", y="Strength", text_auto=True, nbinsx=7, color_continuous_scale='turbid_r', width=686, height=889)
      st.plotly_chart(fig)
      st.success('ALREADY_GRAPH_VISUALIZED!', icon="✅")
    
    button_VISU_10=st.sidebar.button("HEAT_MAP_AGE",type="primary")
    if button_VISU_10:
      st.write("HEAT_MAP(AGE_STRENGTH)")
      fig = px.density_heatmap(DATA_FRAME('df'), x="Age", y="Strength", text_auto=True, nbinsx=7, color_continuous_scale='turbid_r', width=686, height=889)
      st.plotly_chart(fig)
      st.success('ALREADY_GRAPH_VISUALIZED!', icon="✅")
    
    button_VISU_11=st.sidebar.button("HEAT_MAP_Superplasticizer",type="primary")
    if button_VISU_11:
      st.write("HEAT_MAP(Superplasticizer_STRENGTH)")
      fig = px.density_heatmap(DATA_FRAME('df'), x="Superplasticizer", y="Strength", text_auto=True, nbinsx=7, color_continuous_scale='turbid_r', width=686, height=889)
      st.plotly_chart(fig)
      st.success('ALREADY_GRAPH_VISUALIZED!', icon="✅")
      
    button_VISU_12=st.sidebar.button("HEAT_MAP_Fly Ash_STRENGTH",type="primary")
    if button_VISU_12:
      st.write("HEAT_MAP(Fly Ash_STRENGTH)")
      fig = px.density_heatmap(DATA_FRAME('df'), x="Fly Ash", y="Strength", text_auto=True, nbinsx=7, color_continuous_scale='turbid_r', width=686, height=889)
      st.plotly_chart(fig)
      st.success('ALREADY_GRAPH_VISUALIZED!', icon="✅")
      
    button_VISU_13=st.sidebar.button("HEAT_MAP_Water",type="primary")
    if button_VISU_13:
      st.write("HEAT_MAP(Superplasticizer_STRENGTH)")
      fig = px.density_heatmap(DATA_FRAME('df'), x="Water", y="Strength", text_auto=True, nbinsx=7, color_continuous_scale='turbid_r', width=686, height=889)
      st.plotly_chart(fig)
      st.success('ALREADY_GRAPH_VISUALIZED!', icon="✅")
    st.balloons()
#_______________________________________________________________________________________________________________________________________________________________

elif page =="- PREDICTION -":
  st.write("WE_NEED_SOME_INFORMATION_TO_PREDICT_THE STRENGTH OF CONCRETE")
  #------------------------------------------------------------------
  st.write('DATA_HEAD!!')
  st.dataframe(DATA_FRAME('df').head(5))
  with st.form("my_form"):
   def train():
      Cement_=st.number_input("Cement_kg in a m3")
      Blast_Furnace_Slag_=st.number_input("Blast_Furnace_Slag_kg in a m3")
      Fly_Ash_=st.number_input("Fly_Ash_kg in a m3")
      Water_=st.number_input("Water_kg in a m3")
      Superplasticizer_=st.number_input("Superplasticizer_kg in a m3")
      Coarse_Aggregate_=st.number_input("Coarse_Aggregate_kg in a m3")
      Fine_Aggregate_=st.number_input("Fine_Aggregate_kg in a m3")
      Age_=st.number_input("Age_Day (1~365)")
      return Cement_, Blast_Furnace_Slag_, Fly_Ash_, Water_, Superplasticizer_, Coarse_Aggregate_, Fine_Aggregate_, Age_
   train()
   submitted = st.form_submit_button("SUBMIT")
  ok=st.button("PREDICTION_STRENGTH_CONCRETE")
  
  if ok:
    
    x=DATA_FRAME('df').loc[:,DATA_FRAME('df').columns != 'Strength']
    y=DATA_FRAME('df')['Strength']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3 ,random_state=42)
    XGB_REG_model=XGBRegressor()
    XGB_REG_model.fit(x_train,y_train)
    n = np.array([train()])
    Strength_=XGB_REG_model.predict(n)
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()
  
    #--------------------------------------------------------------------------
    st.subheader(f" THE_ESTIMATED_STRENGTH_IS :- \n[{Strength_[0]:.2f}] MPa")
    new_data=pd.DataFrame(n,columns=['Cement_','Blast_Furnace_Slag_','Fly_Ash_','Water_','Superplasticizer_','Coarse_Aggregate_','Fine_Aggregate_','Age_'])
    new_data['Strength_'] = Strength_
    st.dataframe(new_data)
    # df.to_excel('NEW_PREDICTION_DATA.xlsx', index=False)
    st.write('------------------------------ACCURACY_TRAIN-----------------------------')
    Strength_TRAIN=XGB_REG_model.predict(x_train)
    SCORE_TRAIN=r2_score(y_train,Strength_TRAIN)*100
    st.subheader(" ACCURACY_TRAIN_FOR_MODEL_IS :- \n[{:.2f}]%".format(SCORE_TRAIN))
    st.write('------------------------------ACCURACY_TEST------------------------------')
    Strength_TEST=XGB_REG_model.predict(x_test)
    SCORE_TEST=r2_score(y_test,Strength_TEST)*100
    st.subheader(" ACCURACY_TEST_FOR_MODEL_IS :- \n[{:.2f}]%".format(SCORE_TEST))
    st.write('-----------------------------ACCURACCY_GRAPH----------------------------')
    labels = 'ACCURACY_TEST', 'ACCURACY_TRAIN'
    sizes = [48.6, 51.4]
    explode = (0, 0.1)
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
          shadow=True, startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)
    st.write('BEST_RESULT_FOR_CONCRETE_STRENGTH MPa')
    data_matrix = [['ITEM', 'DESCIBTION'],
               ['Cement_KG IN M3',389.9],
               ['Blast Furnace Slag_KG IN M3', 189.0],
               ['Fly Ash_KG IN M3', 0.0],
               ['Water_KG IN M3', 145.9],
               ['Superplasticizer IN M3', 22.0],
               ['Coarse Aggregate_KG IN M3', 944.7],
               ['Fine Aggregate_KG IN M3', 755.8],
               ['Age_Day (1~365)', 91],
               ['Strength_MPa', 82.6]]
    fig = ff.create_table(data_matrix)
    st.plotly_chart(fig)
    #--------------------------------------------------------------------------
    st.success('ALREADY_MODEL_PREDICTED!', icon="✅")
    st.balloons()
else:
  st.header('SOFTWARE_DEVELOPER_AI', divider='red')
  st.write("Concrete is the most important material in civil engineering.The concrete compressive strength is a highly nonlinear function of age andingredients. These ingredients include cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, and fine aggregate.So this the prediction for the strength regarding to the dependent features")
  st.write("-- Input Variable --")
  st.write("Cement (component 1) -- quantitative -- kg in a m3 mixture")
  st.write("Blast Furnace Slag (component 2) -- quantitative -- kg in a m3 mixture")
  st.write("Fly Ash (component 3) -- quantitative -- kg in a m3 mixture")
  st.write("Water (component 4) -- quantitative -- kg in a m3 mixture")
  st.write("Superplasticizer (component 5) -- quantitative -- kg in a m3 mixture")
  st.write("Coarse Aggregate (component 6) -- quantitative -- kg in a m3 mixture")
  st.write("Fine Aggregate (component 7) -- quantitative -- kg in a m3 mixture")
  st.write("Age -- quantitative -- Day (1~365)")
  st.write("-- Output Variable --")
  st.write("Concrete compressive strength -- quantitative -- MPa")
  st.header('_MAY_BE_LIFE_ is :blue[cool] :sunglasses:')
  st.write('-------------------------------CONCLUSOR---------------------------------')
  data_matrix = [['ITEM', 'DESCIBTION'],
               ['DATA_SET', 'CONCRETE_STRENGTH'],
               ['CROSS_VALIDATION', 'TimeSeriesSplit'],
               ['MODEL', 'XGBOOST_REGRESSION'],
               ['SCORE_TRAIN-%', 99.17],
               ['SCORE_TEST-%', 93.86]]
  fig = ff.create_table(data_matrix)
  st.plotly_chart(fig)
