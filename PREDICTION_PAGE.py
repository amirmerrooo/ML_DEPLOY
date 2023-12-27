import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,TimeSeriesSplit
from sklearn.metrics import mean_absolute_error , mean_squared_error ,r2_score
from xgboost import XGBRegressor
import plotly.figure_factory as ff
import matplotlib.pyplot as plt



def load_model():
  with open('amir.pkl','rb') as file:
    data=pickle.load(file)
  return data
data=load_model()
Strength_=data['model']

def SHOW_PREDICTION_PAGE():
  st.title("SOFTWARE_DEVELOPER_PREDICTION")
  st.write("""###WE_NEED_SOME_INFORMATION_TO_PREDICT_THE STRENGTH OF CONCRETE""")
  #------------------------------------------------------------------
  url = 'https://raw.githubusercontent.com/merrooo/ML_DATA/main/concrete_data.csv'
  df=pd.read_csv(url)

  st.write('DATA_HEAD!!')
  st.dataframe(df.head(10))

  with st.form("my_form"):

    Cement_=st.number_input("Cement_kg in a m3")
    Blast_Furnace_Slag_=st.number_input("Blast_Furnace_Slag_kg in a m3")
    Fly_Ash_=st.number_input("Fly_Ash_kg in a m3")
    Water_=st.number_input("Water_kg in a m3")
    Superplasticizer_=st.number_input("Superplasticizer_kg in a m3")
    Coarse_Aggregate_=st.number_input("Coarse_Aggregate_kg in a m3")
    Fine_Aggregate_=st.number_input("Fine_Aggregate_kg in a m3")
    Age_=st.number_input("Age_Day (1~365)")

    submitted = st.form_submit_button("SUBMIT")

  ok=st.button("PREDICTION_STRENGTH_CONCRETE")
  if ok:

    x=df.loc[:,df.columns != 'Strength']
    y=df['Strength']

    XGB_REG_mode=XGBRegressor()

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3 ,random_state=42)

    TSCV = TimeSeriesSplit(n_splits=3)
    score=cross_val_score(XGB_REG_mode,x_train,y_train,cv=TSCV) # kfold
    params={
      'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [.03, 0.05, .07], #so called `eta` value
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500]}

    grid_search=GridSearchCV(
    estimator=XGB_REG_mode,
    param_grid=params,
    scoring='neg_mean_squared_error',
    cv=TSCV)

    Strength_=grid_search.fit(x_train,y_train)
    Strength_=grid_search.predict(x)

    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()
    #--------------------------------------------------------------------------
    st.subheader(f" THE_ESTIMATED_STRENGTH_IS :- \n[{Strength_[1]:.2f}] MPa")
    st.write('------------------------------ACCURACY_TRAIN-----------------------------')
    Strength_TRAIN=grid_search.predict(x_train)
    SCORE_TRAIN=r2_score(y_train,Strength_TRAIN)*100
    st.subheader(" ACCURACY_TRAIN_FOR_MODEL_IS :- \n[{:.2f} %]".format(SCORE_TRAIN))
    st.write('------------------------------ACCURACY_TEST------------------------------')
    Strength_TEST=grid_search.predict(x_test)
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
