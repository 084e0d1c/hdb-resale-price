import pandas as pd
import numpy as np 
import requests

def geo_code(address):
    try:
        r = requests.get('https://developers.onemap.sg/commonapi/search?searchVal={}&returnGeom={}&getAddrDetails={}'.format(address,'Y','N')).json()['results'][0]
        lat = r['LATITUDE']
        lon = r['LONGTITUDE']
    except:
        lat = 1.3245
        lon = 103.8572
    return lat,lon 

df = pd.read_csv('hdb resale.csv')
df['Address'] = df['block'] + ' ' + df['street_name']']
lat_list = []
lon_list = []
end = len(df['Address'])

for address in df['Address']:
    lat,lon = geo_code(address)
    lat_list.append(lat)
    lon_list.append(lon)
    i += 1

lat_array = np.array(lat_list)
lon_array = np.array(lon_list)
df['LAT'] = lat_array
df['LON'] = lon_array
df.to_csv('hdb resale geocoded.csv')
