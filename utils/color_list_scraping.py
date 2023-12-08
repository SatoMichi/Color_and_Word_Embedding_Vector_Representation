import sys
import requests
from bs4 import BeautifulSoup
import string
import re

color = {}

def extract_url_1():
    global color
    url_1 = "https://spokenenglishtips.com/colors-name-in-english"
    with open(sys.path[0] + '/../data/color_table.html', 'r', encoding='utf') as f:
        html_source_1 = f.read()
    soup = BeautifulSoup(html_source_1, "html.parser")
    color_table = soup.find("table")
    trs = color_table.find_all("tr")[2:]
    for tr in trs:
        tds = tr.find_all("td")
        c = tds[-2].text.lower()
        v = tds[-1].text.lower()
        color[c] = [v]
        #print(c+" : "+v)

def extarct_url_2():
    global color
    url_2 = "https://www.facemediagroup.co.uk/resources/a-to-z-guides/a-to-z-of-colours"
    alphabet = list(string.ascii_lowercase)
    href_list = ["https://www.facemediagroup.co.uk/resources/a-to-z-guides/a-to-z-of-colours/colours-beginning-with-the-letter-"+a for a in alphabet]
    for href in href_list:
        r = requests.get(href)
        soup = BeautifulSoup(r.text, "html.parser")
        tbody = soup.find("tbody")
        trs = tbody.find_all("tr")
        for tr in trs:
            tds = tr.find_all("td")
            c = tds[0].text.lower()
            #print(c)
            v = [tds[2].text.lower()] if is_same_color(tds[2].text, tds[4].text) else [None]
            if c not in color.keys(): color[c] = v
            else: color[c] += v

def is_same_color(hex, rgb):
    rgb = tuple(map(int, re.findall(r"\d+", rgb)))
    #print(rgb)
    if hex.lower() == rgb_to_hex(rgb): return True
    else: return False

def rgb_to_hex(rgb):
    if len(rgb) > 3: rgb = rgb[1:]
    return '#%02x%02x%02x' % rgb

if __name__ == '__main__':
    extract_url_1()
    extarct_url_2()
    new_color = {}
    for c,v in color.items():
        if len(v)==1 and None in v: pass
        elif len(v)==2: new_color[c] = v[0]
        else: new_color[c] = v[0]

    with open(sys.path[0] + "/../data/color_list.txt", "w", encoding="utf-8") as f:
        for c,v in new_color.items():
            f.write(c+":"+v+"\n")
            print(c,v)
    print(len(new_color))
