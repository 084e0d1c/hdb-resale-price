import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from scraper_api import ScraperAPIClient
client = ScraperAPIClient('YOUR_API_KEY')

def get_max_page(url,client):
    r = client.get(url)
    soup = BeautifulSoup(r.text)
    pages = soup.find('ul',{'class':'pagination'}).findAll('li')
    page_list = []
    for p in pages:
        if (p.text).isnumeric():
            page_list.append(p.text)
    return int(max(page_list))

def scrape_guru(max_page,client):
    main = pd.DataFrame()
    for i in range(1,max_page+1):
        try:
            URL = 'https://www.propertyguru.com.sg/property-for-sale/'+str(i)+'?property_type=H'
            r = client.get(URL)
            soup = BeautifulSoup(r.text)
            listings = soup.findAll('div',{'class':'col-xs-12 col-sm-7 listing-description'})
        except:
            continue
        for l in listings:
            try:
                address = l.find('a',{'class':'nav-link'}).text
                link = l.find('a',{'class':'nav-link'})['href']
                price = l.find('span',{'class':'price'}).text
                bedrooms = l.find('span',{'class':'bed'}).text
                bathrooms = l.find('span',{'class':'bath'}).text
                floor_area = (l.find('li',{'class':'listing-floorarea pull-left'}).text).split()[0]
                list_date = l.find('div',{'class':'listing-recency'}).text
                try:
                    year_built = l.find('ul',{'class':'clear-both listing-property-type'}).findAll('li',{'class':'pull-left'})[2].text.split()[-1]
                except:
                    year_built = 'NaN'
                temp = pd.DataFrame([[address,link,price,bedrooms,bathrooms,floor_area,list_date,year_built]])
                main = main.append(temp)
            except:
                continue
    main.columns = ['address','link','price','bedrooms','bathrooms','floor_area','list_date','year_built']
    main['date_scraped'] = datetime.now()
    return main

def get_max_page(url,client):
    r = client.get(url)
    soup = BeautifulSoup(r.text)
    last_page_page = soup.findAll('ul',{'class':'SearchPagination-links clearfix'})[0].findAll('li')[-2].text
    return int(last_page_page)

def scrape_99co(max_page,client):
    main = pd.DataFrame()
    for i in range(1,max_page+1):
        try:
            URL = 'https://www.99.co/singapore/sale/hdb?page_num='+str(i)
            r = client.get(URL)
            soup = BeautifulSoup(r.text)
            t_s = soup.findAll('div',{'style':'max-height:270px;opacity:1'})
        except:
            continue

        main = pd.DataFrame()
        for t in t_s:
            try:
                detail_1 = t.find('div',{'class':'_1w7Su'})
                address = detail_1.find('h3',{'class':'_2ncUp _1T6Ql'}).text
                postal = detail_1.findAll('p',{'class':'x0JSc TMSN-'})[0].text.split('·')[0].strip()
                # years = detail_1.findAll('p',{'class':'x0JSc TMSN-'})[-1].text.split('·')[-1].split(' ')[1]
                year_built = detail_1.findAll('p',{'class':'x0JSc TMSN-'})[-1].text.split('·')[0].split(' ')[0]

                detail_2 = t.find('div',{'class':'QyDYr'})
                price = detail_2.find('h3',{'class':'_2ncUp _1T6Ql'}).text
                bedrooms = detail_2.findAll('p',{'class':'x0JSc TMSN-'})[0].text.split()[0]
                bathrooms = detail_2.findAll('p',{'class':'x0JSc TMSN-'})[0].text.split()[1]
                floor_area = detail_2.findAll('p',{'class':'x0JSc TMSN-'})[-1].text.split('/')[-1].strip().split()[0]

                link = 'https://www.99.co' + t.find('a',{'class':'_17qrb _3uV4l'})['href']
                temp = pd.DataFrame([[address,link,price,bedrooms,bathrooms,floor_area,year_built]])
                main = main.append(temp)
            except:
                continue
    main.columns = ['address','link','price','bedrooms','bathrooms','floor_area','year_built']
    main['date_scraped'] = datetime.now()
    return main


URL = 'https://www.propertyguru.com.sg/property-for-sale/0?property_type=H'
max_page = get_max_page(URL,client)
print(max_page)
df = scrape_guru(int(max_page),client)
df.to_csv('property_guru.csv',index=False)

url = 'https://www.99.co/singapore/sale/hdb?page_num=1'
max_page = get_max_page(url,client)
df = scrape_99co(max_page,client)
df.to_csv('99co.csv',index=False)
