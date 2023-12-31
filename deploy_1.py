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
st.header('_AMIR_ is :blue[cool] :sunglasses:')
st.header("CONCRETE_DATA_SET")
st.image("https://media.istockphoto.com/id/692096736/photo/concrete-pouring-during-commercial-concreting-floors-of-building.jpg?s=1024x1024&w=is&k=20&c=XYYH7UhgqsMmwGBWO6UJsxaSgjxNDuQO8i7N27nwRlk=", width=200)
st.write('-------------------------------CONCLUSOR---------------------------------')
data_matrix = [['ITEM', 'DESCIBTION'],
               ['DATA_SET', 'CONCRETE_STRENGTH'],
               ['CROSS_VALIDATION', 'TimeSeriesSplit'],
               ['MODEL', 'XGBOOST_REGRESSION'],
               ['SCORE_TRAIN-%', 99.17],
               ['SCORE_TEST-%', 93.86]]
fig = ff.create_table(data_matrix)
st.plotly_chart(fig)

page=st.sidebar.selectbox("OPTINOS_FOR_EXPLORING_DATA",("EDA","VISUALIZATION","PREDICTION"))

def DATA_FRAME(df):
  url_1= 'https://raw.githubusercontent.com/merrooo/ML_DATA/main/concrete_data.csv'
  df=pd.read_csv(url_1)
  return df
#_______________________________________________________________________________________________________________________________________________________________

# page=st.sidebar.selectbox("OPTINOS_FOR_EXPLORING_DATA",("EDA","VISUALIZATION","PREDICTION"))
if page == "EDA":
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
      elif DUPLICATED_:
        st.write('CHECK_DUPLICATED_OF_DATA!!')
        st.dataframe(DATA_FRAME('df').duplicated().sum())
        st.write('DONE!!')
    elif EDA_ == "COLUMNS_":
      st.write('VALUE_COUNTS_FOR_OUT_PUT[STRENGTH]!!')
      VALUE_COUNTS_=st.toggle('VALUE_COUNTS',disabled=False)
      UNIQUE_=st.toggle('UNIQUE',disabled=False)
      if VALUE_COUNTS_:
        st.write('CHECK_FOR_OUTPUT_[STRENGTH]!!')
        Strength_=st.toggle('Strength',disabled=False)
        if Strength_:
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

elif page =="VISUALIZATION":
   #------------------------------------------------------------------
    st.title("""### VISUALIZATION """)
   #------------------------------------------------------------------
    st.write("AREA_CHART_DEPENDENT FEATURES (CEMENT_AGE_SUPERPLASTICIZER) AFFECTING ON STRENGTH")
    button_VISU_2=st.button("AREA_CHART(CEMENT_AGE_SUPERPLASTICIZER)",type="primary")
    if button_VISU_2:
     st.area_chart(
     DATA_FRAME('df'), x="Strength", y=["Cement", "Superplasticizer","Age"], color=["#f0e936", "#4633f2","#0e6210"]) # Optional
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
      fig = px.imshow(DATA_FRAME('df'), text_auto=True, aspect="auto")
      st.plotly_chart(fig)
   #------------------------------------------------------------------
    st.write("HEAT_MAP(CEMENT_STRENGTH)")
    button_VISU_9=st.button("HEAT_MAP(CEMENT_STRENGTH)",type="primary")
    if button_VISU_9:
      fig = px.density_heatmap(DATA_FRAME('df'), x="Cement", y="Strength", text_auto=True, nbinsx=7, color_continuous_scale='turbid_r', width=686, height=889)
      st.plotly_chart(fig)
   #------------------------------------------------------------------
    st.write("HEAT_MAP(AGE_STRENGTH)")
    button_VISU_10=st.button("HEAT_MAP(AGE_STRENGTH)",type="primary")
    if button_VISU_10:
      fig = px.density_heatmap(DATA_FRAME('df'), x="Age", y="Strength", text_auto=True, nbinsx=7, color_continuous_scale='turbid_r', width=686, height=889)
      st.plotly_chart(fig)
   #------------------------------------------------------------------
    st.write("HEAT_MAP(Superplasticizer_STRENGTH)")
    button_VISU_11=st.button("HEAT_MAP(Superplasticizer_STRENGTH)",type="primary")
    if button_VISU_11:
      fig = px.density_heatmap(DATA_FRAME('df'), x="Superplasticizer", y="Strength", text_auto=True, nbinsx=7, color_continuous_scale='turbid_r', width=686, height=889)
      st.plotly_chart(fig)
   #------------------------------------------------------------------
    st.write("HEAT_MAP(Fly Ash_STRENGTH)")
    button_VISU_12=st.button("HEAT_MAP(Fly Ash_STRENGTH)",type="primary")
    if button_VISU_12:
      fig = px.density_heatmap(DATA_FRAME('df'), x="Fly Ash", y="Strength", text_auto=True, nbinsx=7, color_continuous_scale='turbid_r', width=686, height=889)
      st.plotly_chart(fig)
   #------------------------------------------------------------------
    st.write("HEAT_MAP(Superplasticizer_STRENGTH)")
    button_VISU_13=st.button("HEAT_MAP(Water_STRENGTH)",type="primary")
    if button_VISU_13:
      fig = px.density_heatmap(DATA_FRAME('df'), x="Water", y="Strength", text_auto=True, nbinsx=7, color_continuous_scale='turbid_r', width=686, height=889)
      st.plotly_chart(fig)
    st.success('ALREADY_GRAPH_VISUALIZED!', icon="✅")
    st.balloons()
#_______________________________________________________________________________________________________________________________________________________________

elif page =="PREDICTION":
  st.title("SOFTWARE_DEVELOPER_PREDICTION")
  st.write("""###WE_NEED_SOME_INFORMATION_TO_PREDICT_THE STRENGTH OF CONCRETE""")
  #------------------------------------------------------------------
  st.write('DATA_HEAD!!')
  st.dataframe(DATA_FRAME('df').head(5))
  with st.form("my_form"):
    Cement=st.number_input("Cement_kg in a m3")
    Blast_Furnace_Slag=st.number_input("Blast Furnace Slag in a m3")
    Fly_Ash=st.number_input("Fly_Ash_kg in a m3")
    Water_=st.number_input("Water_kg in a m3")
    Superplasticizer=st.number_input("Superplasticizer_kg in a m3")
    Coarse_Aggregate=st.number_input("Coarse_Aggregate_kg in a m3")
    Fine_Aggregate=st.number_input("Fine_Aggregate_kg in a m3")
    Age=st.number_input("Age_Day (1~365)")
    submitted = st.form_submit_button("SUBMIT")
  ok=st.button("PREDICTION_STRENGTH_CONCRETE")
  if ok:
   x=DATA_FRAME('df').loc[:,DATA_FRAME('df').columns != 'Strength']
   y=DATA_FRAME('df')['Strength']
   XGB_REG_model=XGBRegressor()
   x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3 ,random_state=42)
   XGB_REG_model.fit(x_train,y_train)
   def  user_report():
     Cement=st.number_input("Cement_kg in a m3")
     Blast_Furnace_Slag=st.number_input("Blast Furnace Slag in a m3")
     Fly_Ash=st.number_input("Fly_Ash_kg in a m3")
     Water_=st.number_input("Water_kg in a m3")
     Superplasticizer=st.number_input("Superplasticizer_kg in a m3")
     Coarse_Aggregate=st.number_input("Coarse_Aggregate_kg in a m3")
     Fine_Aggregate=st.number_input("Fine_Aggregate_kg in a m3")
     Age=st.number_input("Age_Day (1~365)")
     submitted = st.form_submit_button("SUBMIT")

     user_report_data = {
       'Cement':Cement,
       'Blast_Furnace_Slag':Blast_Furnace_Slag,
       'Fly_Ash':Fly_Ash,
       'Water_':Water_,
       'Superplasticizer':Superplasticizer,
       'Coarse_Aggregate':Coarse_Aggregate,
       'Fine_Aggregate':Fine_Aggregate,
       'Age':Age}
     report_data = pd.DataFrame(user_report_data, index=[0])
     return report_data
    user_data = user_report()
    Strength_=XGB_REG_model.predict(user_data)
    # Strength_=XGB_REG_model.predict(np.array([[Cement, Blast_Furnace_Slag, Fly_Ash, Water_,Superplasticizer,Coarse_Aggregate,Fine_Aggregate,Age]]))

    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()
    #--------------------------------------------------------------------------
    st.subheader('MPa'+str(np.round(Strength_[0], 2)))
    # st.write(Strength_)
    # st.subheader(f" THE_ESTIMATED_STRENGTH_IS :- \n[{Strength_[0]:.2f}] MPa")
    st.write('------------------------------ACCURACY_TRAIN-----------------------------')
    Strength_TRAIN=XGB_REG_model.predict(x_train)
    SCORE_TRAIN=r2_score(y_train,Strength_TRAIN)*100
    st.subheader(" ACCURACY_TRAIN_FOR_MODEL_IS :- \n[{:.2f} %]".format(SCORE_TRAIN))
    st.write('------------------------------ACCURACY_TEST------------------------------')
    Strength_TEST=XGB_REG_model.predict(x_test)
    SCORE_TEST=r2_score(y_test,Strength_TEST)*100
    st.subheader(" ACCURACY_TEST_FOR_MODEL_IS :- \n[{:.2f} %]".format(SCORE_TEST))
    st.write('-----------------------------ACCURACCY_GRAPH----------------------------')
    labels = 'ACCURACY_TEST', 'ACCURACY_TRAIN'
    sizes = [93.86, 99.17]
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
