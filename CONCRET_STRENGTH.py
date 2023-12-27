import streamlit as st
from VISUALIZATION_PAGE import SHOW_VISU_PAGE
from DATA_PAGE import SHOW_DATA_PAGE
from PREDICTION_PAGE import SHOW_PREDICTION_PAGE
import plotly.figure_factory as ff
from sklearn.model_selection import GridSearchCV
import missingno as msng
from sklearn.model_selection import cross_val_score,TimeSeriesSplit
from xgboost import XGBRegressor
import warnings

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
if page == "PREDICTION":
    SHOW_PREDICTION_PAGE()
    st.balloons()
elif page =="VISUALIZATION":
    SHOW_VISU_PAGE()
    st.success('ALREADY_GRAPH_VISUALIZED!', icon="✅")
    st.balloons()
elif page =="EDA":
    SHOW_DATA_PAGE()
    st.success('ALREADY_DATA_EXPLORED!', icon="✅")
    st.balloons()
#------------------------------------------------------------------------------------------------------------
