import numpy as np
import pandas as pd
import nltk
from collections import Counter
import psycopg2
import time

#数据预处理
csv_file = pd.read_excel(r'D:\data\异常账号.xlsx' )
#print(csv_file)
#print(csv_file.columns)
print(csv_file.shape)

#参数设定
numList = ['0','1','2','3','4','5','6','7','8','9']
word=['a']


acc = csv_file['云蟾账号']

bcc = acc.values.tolist()

t = 0
for n in range(len(acc)):
     if acc[n] in numList:
            t = t+1
            print(acc[n])

counter=Counter(text_clean)

t = 0
s=[]
for n in range(len(acc)):
    counter=Counter(acc[n])
    for letter,count in counter.most_common():
        #print(letter,count)
        if count ==8:
            s.append(letter)

t = 0
s = []

counter = Counter(ccc)
for letter, count in counter.most_common():
    print(letter, count)


import requests, json
url = "http://pss-i.ycgame.com/admin.ashx?q=3uid&qsign=&idx=ycUd0bcadfb3f47465dbb45fce5424d8b89"
data = json.dumps({"name":"uob20b","mobile":None,"ok":0,"msg":"","guest":0,"id":None,"tname":None})
r = requests.post(url, data)
response = requests.get(url,data)
print (r.content)
print(response.cookies)
print(response.status_code)
print(response.text)
print(type(response.text))
print(response.json)
print(type(response.json))

# 检测游戏账号是否创建角色
conn = psycopg2.connect(database="db_ana", user="cyread", password="oS8lckyVT9q9q0qr", host="192.168.2.154",
                        port="5432")
cursor = conn.cursor()
sqlv = "SELECT VERSION()"
cursor.execute(sqlv)
data = cursor.fetchone()
print("database version : %s " % data)
t = 0
data_ = []
for n in range(len(csv_file)):
    sql_ = """select * from db_ana.public.player_last_info where "acct"=  """
    aaa = csv_file['游戏账号'][n]
    sql_ = sql_ + '\'' + aaa + '\''
    # print(sql_)
    # data.append(sql_)
    cursor.execute(sql_)
    rows_ = cursor.fetchall()
    t = t + 1
    print(csv_file['游戏账号'][n], end='')
    print(rows_, end='')
    print(t)
    data_.append(rows_)

#     if rows_ is None:
#         pass
#     else:
#         print(csv_file['游戏账号'][n])
conn.commit()
cursor.close()
conn.close()
