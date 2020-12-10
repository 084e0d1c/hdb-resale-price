import streamlit as st
import pandas as pd
import numpy as np
import hdb_resale_model
from hdb_resale_model import model_stacker,data_transformer
import requests
import matplotlib.pyplot as plt

@st.cache
def load_data():
    df = pd.read_csv('./data_files/clean_data.csv')
    return df
df = load_data()

def load_model():
    blender_model = model_stacker(load_model=True)
    return blender_model

def geo_code(address):
    try:
        r = requests.get('https://developers.onemap.sg/commonapi/search?searchVal={}&returnGeom={}&getAddrDetails={}'.format(address,'Y','N')).json()['results'][0]
        lat = r['LATITUDE']
        lon = r['LONGTITUDE']
    except:
        lat = 1.3245
        lon = 103.8572
    return float(lat),float(lon)

st.title("HDB Resale Flat Price Predictor")
col1,col2  = st.beta_columns(2)
col3,col4  = st.beta_columns(2)
bedrooms = col1.number_input('No. of Bedrooms',min_value=0.0,max_value=5.0,value=3.0)
bathrooms = col2.number_input('No. of Bathrooms',min_value=0.0,max_value=3.0,value=2.0)
floor_area = col3.number_input('Floor Area (sqm)',min_value=0.0,value=100.0)
lease = col4.number_input('Remaining Lease',min_value=0.0,max_value=95.0,value=95.0)

address = st.text_input('Enter the postal code',value='238801')

lat,lon = geo_code(address)
prediction_df = pd.DataFrame.from_dict(
    {'No of bedrooms':bedrooms,
    'No of bathrooms':bathrooms,
    'floor_area_sqm':floor_area,
    'remaining_lease':lease,
    'LAT':lat,
    'LON':lon},
    orient='index'
)

btn_val = st.button('Predict!')

if btn_val:
    prediction_df = prediction_df.transpose()
    transformer = data_transformer()
    prediction_df = transformer.add_buildings(prediction_df)
    display_data_df = prediction_df.transpose()
    display_data_df.columns = ['Your input']
    display,mapping = st.beta_columns(2)
    display.write(display_data_df)
    model = load_model()
    result = model.predict(prediction_df)
    plotting_df = prediction_df[['LAT','LON']]
    plotting_df.columns = ['lat','lon']
    mapping.map(plotting_df,zoom=10)
    'Predicted price:',result[0]


st.title("Data Exploration")
flat_type = st.selectbox('Which type of HDB would like to explore?', ['2 Room','3 Room','4 Room','5 Room'])
'You have selected:',flat_type,'resale flats'

clean_df = pd.read_csv('./data_files/hdb resale.csv')

if flat_type == '2 Room':
    df = df[(df['No of bedrooms'] == 1) & (df['No of bathrooms'] == 1)]
    clean_df = clean_df[clean_df['flat_type'] == '2 ROOM']
elif flat_type == '3 Room':
    df = df[(df['No of bedrooms'] == 2) & (df['No of bathrooms'] == 2)]
    clean_df = clean_df[clean_df['flat_type'] == '3 ROOM']
elif flat_type == '4 Room':
    df = df[(df['No of bedrooms'] == 3) & (df['No of bathrooms'] == 2) & (df['floor_area_sqm'] < 110)]
    clean_df = clean_df[clean_df['flat_type'] == '4 ROOM']
elif flat_type == '5 Room':
    df = df[(df['No of bedrooms'] == 3) & (df['No of bathrooms'] == 2) & (df['floor_area_sqm'] >= 110)]
    clean_df = clean_df[clean_df['flat_type'] == '5 ROOM']

clean_df['psm'] = clean_df['resale_price']/clean_df['floor_area_sqm']
type_df = clean_df.groupby(['month'])['resale_price'].mean().reset_index()

# Displaying map
st.subheader('Location of {} resale flats sold within specified prices'.format(flat_type))
map_df = df[['LAT','LON','resale_price']]
map_df.columns = ['lat','lon','resale_price']
col6, col7 = st.beta_columns(2)
min_val = col6.number_input('Min price',value=map_df['resale_price'].min())
max_val = col7.number_input('Max Price',value=map_df['resale_price'].max())
display_df = map_df[(map_df['resale_price'] >= min_val)&(map_df['resale_price'] <= max_val)]
st.map(display_df,zoom=10)

fig,ax = plt.subplots(figsize=(20,10))
display_df.plot.scatter(x='lat',y='lon',alpha=0.4,c='resale_price',cmap=plt.get_cmap('jet'),colorbar=True,ax=ax)
st.subheader('Price heatmap of all {} resale flats'.format(flat_type))
st.pyplot(fig)

# Remaining Lease when flats are sold
remaining_lease = df.groupby('remaining_lease').count()
remaining_lease = remaining_lease[['LAT']]
remaining_lease.columns = ['Frequency']
st.subheader('Age of {} resale flats when sold'.format(flat_type))
st.bar_chart(remaining_lease)

# Plotting flat price data over the years
st.subheader('Selling price of {} resale flats over the years'.format(flat_type))
'Zoom in a little to see the fluctuations better'
st.line_chart(type_df.set_index('month'),use_container_width=True)


