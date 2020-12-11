import streamlit as st
import pandas as pd
import numpy as np
import hdb_resale_model
from hdb_resale_model import model_stacker,data_transformer
import requests
import matplotlib.pyplot as plt
from math import cos, asin, sqrt

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

def distance(lat1, lon1, lat2, lon2):
    lat1 = float(lat1)
    lat2 = float(lat2)
    # to calculate closest vicinity
    p = 0.017453292519943295
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
    return 12742 * asin(sqrt(a))

def filter_dataset(df,flat_type,town,radius):
    lat,lon = geo_code(town)
    if town != "All":
        df['distance_to_town'] = [distance(df.loc[idx,'LAT'],df.loc[idx,'LON'],lat,lon) for idx in range(len(df))]
        df = df[df['distance_to_town']<radius]

    if flat_type == '2 Room':
        df = df[(df['No of bedrooms'] == 1) & (df['No of bathrooms'] == 1)]
    elif flat_type == '3 Room':
        df = df[(df['No of bedrooms'] == 2) & (df['No of bathrooms'] == 2)]
    elif flat_type == '4 Room':
        df = df[(df['No of bedrooms'] == 3) & (df['No of bathrooms'] == 2) & (df['floor_area_sqm'] < 110)]
    elif flat_type == '5 Room':
        df = df[(df['No of bedrooms'] == 3) & (df['No of bathrooms'] == 2) & (df['floor_area_sqm'] >= 110)]      
    return df

st.title("HDB Resale Flat Price Predictor")
"I designed this tool as a means of quickly scanning if an agent is pitching a price significantly above the fair value of a flat. Hence, factors such as storey range have been excluded as sites like 99co and PropertyGuru only provide it upon further negotiations and requests."
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

btn_val = st.button('Estimate Price')

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
    mapping.map(plotting_df,zoom=12)
    'Estimated price:',result[0]

st.title("Data Exploration")
clean_df = pd.read_csv('./data_files/hdb resale.csv')
flat_type_col,location_col = st.beta_columns(2)
flat_type = flat_type_col.selectbox('Which type of HDB would like to explore?', ['2 Room','3 Room','4 Room','5 Room'])
location_options = [x.title() for x in clean_df['town'].unique()]
location_options.insert(0,'All')
selected_town = location_col.selectbox('Which area would you like to view?',location_options)
if selected_town != "All":
    selected_radius = st.number_input('Select a radius for viewing',min_value=1.0,max_value=10.0,value=3.0)
    'You have selected:',flat_type,'resale flats near',selected_town,'within a ',selected_radius,'km radius.'
    clean_df = clean_df[(clean_df['town'] == selected_town.upper()) & (clean_df['flat_type'] == flat_type.upper())]
else:
    selected_radius = 999999
    clean_df = clean_df[(clean_df['flat_type'] == flat_type.upper())]
    'You have selected: All',flat_type,'resale flats.'


df = filter_dataset(df,flat_type,selected_town,selected_radius)

# Displaying map - using geoencoded df
st.subheader('Location of {} resale flats sold within specified prices'.format(flat_type))
map_df = df[['LAT','LON','resale_price']]
map_df.columns = ['lat','lon','resale_price']
col6, col7 = st.beta_columns(2)
min_val = col6.number_input('Min price',value=map_df['resale_price'].min())
max_val = col7.number_input('Max Price',value=map_df['resale_price'].max())
display_df = map_df[(map_df['resale_price'] >= min_val)&(map_df['resale_price'] <= max_val)]
st.map(display_df,zoom=11)

fig,ax = plt.subplots(figsize=(20,10))
display_df.plot.scatter(x='lat',y='lon',alpha=0.1,c='resale_price',cmap=plt.get_cmap('jet'),colorbar=True,ax=ax)
st.subheader('Price heatmap of all {} resale flats'.format(flat_type))
'This is a zoomed in picture'
st.pyplot(fig)

# Remaining Lease when flats are sold
remaining_lease = df.groupby('remaining_lease').count()
remaining_lease = remaining_lease[['LAT']]
remaining_lease.columns = ['Frequency']
st.subheader('Age of {} resale flats when sold'.format(flat_type))
st.bar_chart(remaining_lease)

type_df = clean_df.groupby(['month'])['resale_price'].mean().reset_index()
# Plotting flat price data over the years
if selected_town != "All":
    st.subheader('Selling price of {} resale flats in {} over the years'.format(flat_type,selected_town))
else:
    st.subheader('Selling price of all {} resale flats over the years'.format(flat_type))
    
'Zoom in a little to see the fluctuations better'
st.line_chart(type_df.set_index('month'))

st.subheader('Disclaimer')
'The model has only been trained on data from Jan 2017 to May 2020.'
'Privacy is of utmost importance and any parameters inserted into the model for prediction are not collected.'
'All views and findings presented in my code or repository are my own and do not represent the opinions of any entity whatsoever with which I have been, am now, or will be affiliated. All material provided are for general information purposes only and do not constitute accounting, legal, tax, or other professional advice. Visitors should not act upon the content or information found here without first seeking appropriate advice from an accountant, financial planner, lawyer or other professional. Usage of any material contained within this repository constitutes an explicit understanding and acceptance of the terms of this disclaimer.'