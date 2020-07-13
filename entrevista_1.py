# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 12:07:14 2020

@author: Frederico
"""


import json
from operator import itemgetter

#open
js = open("source_file_2.json", 'r')
js_data = js.read()

# parse
data = json.loads(js_data)
data = sorted(data, key=itemgetter('priority'), reverse=True)

#create dictionary
dict_managers = dict()
dict_watchers = dict()

#loop for
for elem in iter(data):
    for elem_manager in elem['managers']:
        if elem_manager not in dict_managers:
            dict_managers[elem_manager] = list()
        dict_managers[elem_manager].append(elem['name'])

for elem in iter(data):
    for elem_watcher in elem['watchers']:
        if elem_watcher not in dict_watchers:
            dict_watchers[elem_watcher] = list()
        dict_watchers[elem_watcher].append(elem['name'])
        
dict_managers_json = json.dumps(dict_managers)
dict_watchers_json = json.dumps(dict_watchers)

file_managers = open("managers.json", "w")
file_watchers = open("watchers.json", "w")

file_managers.write(dict_managers_json)
file_watchers.write(dict_watchers_json)
