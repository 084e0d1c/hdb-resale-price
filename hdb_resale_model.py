
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.linear_model import LinearRegression,ElasticNet
import xgboost as xgb
from math import cos, asin, sqrt
import joblib
import numpy as np 
import requests
import pickle
import os.path
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt 
import pandas as pd


def get_num(s):
    s = s.split()
    return s[0]
    
def get_rooms(flat_type):
    if flat_type == '2 ROOM':
        return 1,1
    elif flat_type =='3 ROOM':
        return 2,2
    elif flat_type =='MULTI-GENERATION':
        return 4,3
    else:
        return 3,2

def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
    return 12742 * asin(sqrt(a)) 

def closest(data, v):
    t = min(data, key=lambda p: distance(v['lat'],v['lon'],p['lat'],p['lon']))
    return distance(t['lat'],t['lon'],v['lat'],v['lon'])

def clean_col_names(l):
    return_list = []
    for item in l:
        item = item.replace(' ','_')
        item = item.replace('-','_')
        return_list.append(item)
    return return_list

class data_transformer():
    def __init__(self,geocode=True,include_buildings=True):
        self.geocode = geocode
        self.include_buildings = include_buildings

    def fit(self,df):
        self.df = df
        return self

    def add_buildings(self,df):
        buildings = pd.read_json('./data_files/buildings.json')
        buildings.BUILDING = buildings.BUILDING.str.lower()
        buildings.drop_duplicates('BUILDING',inplace=True)
        places = {'primary_school':'primary school',
                        'seconday_school':'secondary school|high school',
                        'junior_college':'junior college',
                        'polytechnic':'polytechnic',
                        'mrt_lrt':'mrt|lrt',
                        'preschool':'preschool'}
        df.reset_index(inplace=True,drop=True)
        for place in places:
            search_key = places[place]
            temp_df = buildings[buildings['BUILDING'].str.contains(search_key)][['LATITUDE','LONGTITUDE']]
            temp_df.reset_index(inplace=True,drop=True)
            places_lon_lat = []
            for i in range(len(temp_df)):
                temp_dict = {'lat':temp_df.loc[i,'LATITUDE'],'lon':temp_df.loc[i,'LONGTITUDE']}
                places_lon_lat.append(temp_dict)
            
            df['distance_to_'+str(place)] = [closest(places_lon_lat,{'lat':df.loc[idx,'LAT'],'lon':df.loc[idx,'LON']}) for idx in range(len(df))]
        return df

    def clean_transform(self):
        df = self.df
        df['psm'] = df['resale_price']/df['floor_area_sqm']
        psm_df = pd.DataFrame(df.groupby('town')['psm'].mean())
        psm_df.columns = ['area_average_psm']

        if os.path.isfile('area_average_psm.pkl') == False:
            joblib.dump(psm_df,'area_average_psm.pkl')

        df = pd.merge(df,psm_df,how='left',on='town')
        df['remaining_lease'] = df['remaining_lease'].apply(get_num)
        df['No of bedrooms'] = 0
        df['No of bathrooms'] = 0
        df[['No of bedrooms','No of bathrooms']] = [get_rooms(df.loc[idx,'flat_type']) for idx in range(len(df))]
        df['street_name'] = df['street_name'].astype(str)
        df['block'] = df['block'].astype(str)
        df['Address'] = df['block'] + ' ' + df['street_name']
        df['storey'] = df['storey_range'].apply(get_num)
        
        if self.geocode:
            geocode = pd.read_csv('./data_files/hdb resale geocoded.csv')
            geocode = geocode[['Address','LAT','LON']]
            df = pd.merge(df,geocode,how='left',on='Address')

        df.drop_duplicates(inplace=True)
        df.drop(['town','month','flat_type','block','street_name','lease_commence_date','storey_range','psm','Address'],inplace=True,axis=1)
        df['LAT'] = df['LAT'].round(4)
        df['LON'] = df['LON'].round(4)

        if self.include_buildings:
            df = self.add_buildings(df)

        cols = ['floor_area_sqm', 'remaining_lease', 'resale_price','No of bedrooms', 'No of bathrooms','storey', 'LAT', 'LON','area_average_psm',
                'distance_to_primary_school','distance_to_junior_college','distance_to_polytechnic','distance_to_mrt_lrt']
        df[cols] = df[cols].apply(pd.to_numeric)
        df = pd.get_dummies(df)
        self.df=df

        return df

    def scale(self,guru=False):
        if guru:
            file_path = 'guru_scaler.pkl'
        else:
            file_path = 'standard_scaler.pkl'
        df = self.df
        if os.path.isfile(file_path):
            scaler = joblib.load(file_path)
        else:
            scaler = StandardScaler().fit(df.loc[:,df.columns!='resale_price'])

        df.loc[:,df.columns!='resale_price'] = scaler.transform(df.loc[:,df.columns!='resale_price'])
        
        if os.path.isfile(file_path)==False:
            joblib.dump(scaler,file_path)
        return df

    def fit_transform(self,df):
        self.df = df  
        df = self.clean_transform()
        df = self.scale()
        self.df = df
        return df

    def train_test_split(self):
        df = self.df
        df['resale_price_cat'] = pd.cut(df['resale_price'],bins=5,labels=[0,1,2,3,4])
        X_train,X_test,y_train,y_test = train_test_split(df.loc[:,df.columns!='resale_price'],df['resale_price'],stratify=df.resale_price_cat,test_size=0.2)
        X_train.drop('resale_price_cat',inplace=True,axis=1)
        X_test.drop('resale_price_cat',inplace=True,axis=1)
        df.drop('resale_price_cat',inplace=True,axis=1)
        X_train.reset_index(drop=True,inplace=True)
        X_test.reset_index(drop=True,inplace=True)
        y_train.reset_index(drop=True,inplace=True)
        y_test.reset_index(drop=True,inplace=True)

        return X_train,X_test,y_train,y_test

    def gridsearchcv_prep(self):
        df = self.df
        return df.loc[:,df.columns!='resale_price'],df['resale_price']

    # def prepare_for_predict(self,df):
    #     psm_df = joblib.load('area_average_psm.pkl')
    #     df = pd.merge(df,psm_df,how='left',on='town')
    #     df['remaining_lease'] = df['remaining_lease'].apply(get_num)
    #     df['No of bedrooms'] = 0
    #     df['No of bathrooms'] = 0
    #     df[['No of bedrooms','No of bathrooms']] = [get_rooms(df.loc[idx,'flat_type']) for idx in range(len(df))]
    #     df['block'] = df['block'].astype(str)
    #     df['street_name'] = df['street_name'].astype(str)
    #     df['Address'] = df['block'] + ' ' + df['street_name']
    #     df['storey'] = df['storey_range'].apply(get_num)
    #     if self.geocode:
    #         geocode = pd.read_csv('hdb resale geocoded.csv')
    #         geocode = geocode[['Address','LAT','LON']]
    #         df = pd.merge(df,geocode,how='left',on='Address')

    #     df.drop_duplicates(inplace=True)
    #     df.drop(['town','month','flat_type','block','street_name','lease_commence_date','storey_range','Address'],inplace=True,axis=1)
    #     df['LAT'] = df['LAT'].round(4)
    #     df['LON'] = df['LON'].round(4)

    #     if self.include_buildings:
    #         df = self.add_buildings(df)

    #     cols = ['floor_area_sqm', 'remaining_lease','No of bedrooms', 'No of bathrooms','storey', 'LAT', 'LON','area_average_psm',
    #             'distance_to_primary_school','distance_to_junior_college','distance_to_polytechnic','distance_to_mrt_lrt']
    #     df[cols] = df[cols].apply(pd.to_numeric)
    #     flat_types = ['flat_model_2-room', 'flat_model_Adjoined flat', 'flat_model_Apartment',
    #    'flat_model_DBSS', 'flat_model_Improved',
    #    'flat_model_Improved-Maisonette', 'flat_model_Maisonette',
    #    'flat_model_Model A', 'flat_model_Model A-Maisonette',
    #    'flat_model_Model A2', 'flat_model_Multi Generation',
    #    'flat_model_New Generation', 'flat_model_Premium Apartment',
    #    'flat_model_Premium Apartment Loft', 'flat_model_Premium Maisonette',
    #    'flat_model_Simplified', 'flat_model_Standard', 'flat_model_Terrace',
    #    'flat_model_Type S1', 'flat_model_Type S2']
    #     for idx in range(len(df)):
    #         for col in flat_types:
    #             flat_type = col.split('_')[-1]
    #             if df.loc[idx,'flat_model'] == flat_type:
    #                 df.loc[idx,col] = 1
    #             else:
    #                 df[col] = 0
    #     df.drop('flat_model',inplace=True,axis=1)

    #     df = self.scale(df)
    #     return df

    def prepare_for_guru(self,df,train=False):
        self.df = df
        self.clean_transform()
        cols = ['No of bedrooms', 'No of bathrooms', 'floor_area_sqm', 'remaining_lease', 'LAT',
       'LON', 'distance_to_primary_school', 'distance_to_seconday_school',
       'distance_to_junior_college', 'distance_to_polytechnic',
       'distance_to_mrt_lrt']
        if train == True:
            cols += ['resale_price']
        self.df = self.df[cols]
        df = self.scale(guru=True)
        self.df = df
        return df
    
##------------------------------------------------------------------------------------------------------------------------------------------

class resale_price_model():
    def __init__(self,*args,load_model,guru,model_type):
        if guru:
            file_path = 'guru_price_model.pkl'
        else:
            file_path = 'resale_price_model.pkl'

        if model_type == 'random_forest_':
            self.file_path = model_type + file_path
        elif model_type == 'clustering_':
            self.file_path = model_type + file_path
        elif model_type == 'xgboost_':
            self.file_path = model_type + file_path

        if load_model:
            self.model = joblib.load(self.file_path)
        else:
            self.model = args[0]
    
    def predict(self,input_data):
        m = self.model 
        pred = m.predict(input_data)
        return pred

    def fit(self,x,y):
        m = self.model
        m.fit(x,y)
        return m

    def save_model(self):
        model = self.model
        if os.path.isfile(self.file_path):
            save_boolean = input('Model already exists, confirm save? (Y/N)').lower()
            if save_boolean == "y":
                joblib.dump(model,self.file_path)
                print('Saved Successfully.')
            else:
                print('Did not save.')
        else:
            joblib.dump(model,self.file_path)
            print('Saved Successfully.')

    def get_model(self):
        return self.model

##------------------------------------------------------------------------------------------------------------------------------------------

class model_stacker(resale_price_model):

    def __init__(self,*args,load_model):
        self.file_path = 'stacker_guru_price_model.pkl'

        if load_model:
            self.load_models()
        else:
            self.model = args[0]

    def load_models(self):
        self.clustering = joblib.load('./pickles/clustering_guru_price_model.pkl')
        self.random_forest = joblib.load('./pickles/random_forest_guru_price_model.pkl')
        self.xgboost = joblib.load('./pickles/xgboost_guru_price_model.pkl')
        self.stacker = joblib.load('./pickles/stacker_guru_price_model.pkl')
        print('Models loaded successfully.')
        return self
    
    def predict(self,pred_input):
        random_forest_pred = self.random_forest.predict(pred_input)
        xgboost_pred = self.xgboost.predict(pred_input)
        clustering_pred = self.clustering.predict(pred_input)

        pred_input = pd.DataFrame({'Random_Forest':random_forest_pred,
                            'XGBoost':xgboost_pred,
                            'Clustering':clustering_pred})
        m = self.stacker
        return m.predict(pred_input)

    def training_predict(self,pred_input):
        return self.model.predict(pred_input)



