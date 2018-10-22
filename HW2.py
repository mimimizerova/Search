import os
import re
import requests
import random
import time
import pandas as pd
import html
from fake_useragent import UserAgent


ua = UserAgent(verify_ssl=False)


def get_data_from_url(pageUrl):
    d = {}
    time.sleep(random.choice(range(10, 15)))
    headers = {'User-Agent': ua.random}
    session = requests.session()
    r = session.get(url, headers=headers)
    html = r.text
    return html


def find_url(page_code):
    reg_url = re.compile('"/moskva/knigi_i_zhurnaly/.*?"', flags=re.U | re.DOTALL)
    url = reg_url.findall(code)
    clear_url = []
    for link in url:
        new_link = link.replace('"', '')
        clear_url.append(new_link)
    return clear_url[3:]
    
 
def find_title (page_code):
    reg = re.compile('<span class="title-info-title-text" itemprop="name">.*?</span>', flags=re.U | re.DOTALL)
    regTag = re.compile('<.*?>', flags=re.U | re.DOTALL) 
    title = reg.search(page_code)
    if title == None:
        return 'None'
    else:
        title = title.group(0)
        clean_t = regTag.sub("", title)
        clean_t = clean_t.replace('\n', ' ')
        return html.unescape(clean_t)


def find_price(page_code):
    reg_p = re.compile(' <span class="js-item-price".*?>.*?</span>', flags=re.U | re.DOTALL)
    regTag = re.compile('<.*?>', flags=re.U | re.DOTALL) 
    price = reg_p.search(page_code)
    if price == None:
        return 'None'
    else:
        price = price.group(0)
        clean_p = regTag.sub("", price)
        clean_p = clean_p.replace('\n', ' ')
        return html.unescape(clean_p)


def find_seller(page_code):
    reg_s = re.compile('<span class="sticky-header-seller-text".*?>.*?</span>', flags=re.U | re.DOTALL)
    regTag = re.compile('<.*?>', flags=re.U | re.DOTALL) 
    seller = reg_s.search(page_code)
    if seller == None:
        return 'None'
    else:
        seller = seller.group(0)
        clean_s = regTag.sub("", seller)
        clean_s = clean_s.replace('\n', ' ')
    return html.unescape(clean_s)


def find_item(page_code):
    reg_i = re.compile('<div class="title-info-metadata-item">.*?</div>', flags=re.U | re.DOTALL)
    regTag = re.compile('<.*?>', flags=re.U | re.DOTALL) 
    item = reg_i.search(page_code)
    if item == None:
        return 'None'
    else:
        item = item.group(0)
        clean_i = regTag.sub("", item)
        clean_i = clean_i.replace('\n', ' ')
        return html.unescape(clean_i)


def find_views(page_code):
    reg_v = re.compile('<div class="title-info-metadata-item title-info-metadata-views">.*?</div>', flags=re.U | re.DOTALL)
    regTag = re.compile('<.*?>', flags=re.U | re.DOTALL) 
    views = reg_v.search(page_code)
    if views == None:
        return 'None'
    else:
        views = views.group(0)
        clean_v = regTag.sub("", views)
        clean_v = clean_v.replace('\n', ' ')
        return html.unescape(clean_v)

def find_adress(page_code):
    reg_a = re.compile('<div class="item-map-location">.*?</div>', flags=re.U | re.DOTALL)
    regTag = re.compile('<.*?>', flags=re.U | re.DOTALL) 
    adress = reg_a.search(page_code)
    if adress == None:
        return 'None'
    else:
        adress = adress.group(0)
        clean_a = regTag.sub("", adress)
        clean_a = clean_a.replace('\n', ' ')
        clean_a = clean_a.replace('Посмотреть карту', '')
        clean_a = clean_a.replace('Адрес:', '')
        return html.unescape(clean_a)


def find_text(page_code):
    reg_t = re.compile('<div class="item-description-text".*?>.*?</div>', flags=re.U | re.DOTALL)
    regTag = re.compile('<.*?>', flags=re.U | re.DOTALL) 
    text = reg_t.search(page_code)
    if text == None:
        return 'None'
    else:
        text = text.group(0)
        clean_text = regTag.sub("", text)
        clean_text = clean_text.replace('\n', ' ')
        return html.unescape(clean_text)


def create_doc(pageUrl):
    page_code = get_data_from_url(pageUrl)
    name = pageUrl.replace('https://www.avito.ru/moskva/knigi_i_zhurnaly/', '')
    name = name.replace('.', '')
    name += '.txt'
    time.sleep(random.choice(range(10, 15)))
    doc = open(name, 'w', encoding = 'utf-8')
    doc.write('Название: ' + find_title(page_code) + '\n')
    doc.write('Цена: ' + find_price(page_code) + '\n')
    doc.write('Объявление: ' + find_item(page_code) + '\n')
    doc.write('Просмотры: ' + find_views(page_code) + '\n')
    doc.write('Автор: ' + find_seller(page_code) + '\n')
    doc.write('Адрес: ' + find_adress(page_code) + '\n')
    doc.write('Текст: ' + find_text(page_code) + '\n')
    
 #Получаем список адресов: 
 for i in range(1, 100)):
    pageUrl = 'https://www.avito.ru/moskva/knigi_i_zhurnaly?p={}'.format(i+1)
    arr = find_url(get_data_from_url(pageUrl))
    with open('list_url.txt','a',encoding='utf-8') as f:
        for url in arr:
            f.write('https://www.avito.ru/moskva/knigi_i_zhurnaly/'+url+'\n')

 
def get_all_collection():
    with open('list_url.txt','r') as f:
        links = f.read().split('\n')
        random.shuffle(links)
        for url in links:
        time.sleep(random.choice(range(10, 20)))
        create_doc(url)
