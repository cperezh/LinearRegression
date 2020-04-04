# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 07:39:13 2020

@author: Carlos
"""
def w():
    import json
    with open("file.csv","a+") as f:
        j = dict(a=1,b=2)
        json.dump(j,f)
        
def r():
    import json
    with open("file.csv","r") as f:
        j = json.load(f)
        print(type(j))

def e():
    try:
        f = open("dramas")
    except FileNotFoundError:
        print("dramas")
    except:
        print("drmaas2")

if __name__ == "__main__":
    e()