from simulator_response import chat_player,player_init

import random
import time
import uuid
import json
import requests
import os

'''
#player structure
player_data = {
    "id": str,
    "emo_point": int,
    "emo_state": str,
    "player": str,
    "scene": str,
    "character": str,
    "topic": str,
    "task": str,
    "history": list
}

#history structure - assistant
history = {
    "role": "assistant",
    "content": str,
    "think": srt#(option)
}

#history structure - user
history = {
    "role": "user",
    "content": str,
    "thinking": str,
    "emotion-point" int,
    "planning":{
          "activity": str,
          "TargetCompletion": str,
          "content": str,
          "analyse": str,
          "change": int
      }
}
'''

#load simulator profile without first talk
# get the current dir and profile path
current_dir = os.path.dirname(__file__)
origin_profile_path = os.path.join(current_dir, 'profile', 'simulator_profile.jsonl')

with open(origin_profile_path,"r",encoding="utf-8") as f:
    data = [json.loads(line) for line in f]
    f.close()

#build store file
store_file = os.path.join(current_dir, 'profile', 'simulator_profile_withfirsttalk.jsonl')
if not os.path.exists(store_file):
    with open(store_file,"w",encoding="utf-8") as f:
        f.close()

#if there exists data in store file, then avoid running with the same simulator profile
id_list = []
with open(store_file,"r",encoding="utf-8") as f:
    id_list = [json.loads(line)["id"] for line in f]

for simulator in data:
    #ignore the already talked simulator profile
    if simulator["id"] in id_list:
        continue

    #initialize player with simulator profile
    player = player_init(id=simulator["id"])

    print(player['player'])
    print(player['scene'])
    print(player['character'])
    print(player['task'])

    #call the simulator to give the first response and break
    player = chat_player(player)
    print("player:{}".format(player["history"][-1]['content']))
    print("player-emotion:{},{}".format(player["emo_point"],player['emo_state']))
    print(player["history"][-1])
    print("\n")
    
    #add first talk to the simulator profile
    simulator["first_talk"] = player["history"][-1]['content']


    with open(store_file,'a',encoding='utf-8') as file:
        file.write(json.dumps(simulator, ensure_ascii=False) + "\n")