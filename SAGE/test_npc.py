from simulator_response import chat_player,player_init
from npc_response import call_npc

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

#get the current dir load simulator profile with first talk
current_dir = os.path.dirname(__file__)
profile_path = os.path.join(current_dir, 'profile', 'simulator_profile_withfirsttalk.jsonl')
with open(profile_path,"r",encoding="utf-8") as f:
    data = [json.loads(line) for line in f]
    f.close()

#build the store file
store_file = "test_result_jsonl"
store_file = os.path.join(current_dir, store_file)
if not os.path.exists(store_file):
    with open(store_file,"w",encoding="utf-8") as f:
        f.close()

#if there exists data in store file, then avoid running with the same simulator profile
id_list = []
with open(store_file,"r",encoding="utf-8") as f:
    id_list = [json.loads(line)["id"] for line in f]

#talk with human
testmode = "human"

#talk with llm
testmode = "npc"

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

    print("player:{}".format(simulator['first_talk']))

    turns = 0
    #talk with llm
    while testmode != "human":
        turns += 1
        #the max conversation turn is 10 round
        if turns>10:
            break
        #if it is the first talk, then load the presetting first talk of simulator
        if turns == 1:
            player['history'].append({"role": "user", "content": simulator['first_talk'],"emotion-point": player['emo_point']})
        else:
            #call the simulator response
            player = chat_player(player)
            print("player:{}".format(player["history"][-1]['content']))
            print("player-emotion:{},{}".format(player["emo_point"],player['emo_state']))
            print(player["history"][-1])
            print("\n")

            #if simulator says goodble or the emotion-point is smaller than 10 or larger than 100, stop the conversation
            if "再见" in player["history"][-1]["content"] or "拜拜" in player["history"][-1]["content"]:
                break
            
            if player["history"][-1]["emotion-point"]>=100 or player["history"][-1]["emotion-point"]<10:
                break
        # call the npc response and update history
        query =  call_npc(player["history"])
        new_state = {"role": "assistant", "content": query}
        player['history'].append(new_state)
    
    #talk with human
    while testmode == "human":
        turns += 1
            #the max conversation turn is 10 round
        if turns>10:
            break
        #if it is the first talk, then load the presetting first talk of simulator
        if turns == 1:
            player['history'].append({"role": "user", "content": simulator['first_talk'],"emotion-point": player['emo_point']})
        else:
            #call the simulator response
            player = chat_player(player)
            print("player:{}".format(player["history"][-1]['content']))
            print("player-emotion:{},{}".format(player["emo_point"],player['emo_state']))
            print(player["history"][-1])
            print("\n")

            #if simulator says goodble or the emotion-point is smaller than 10 or larger than 100, stop the conversation
            if "再见" in player["history"][-1]["content"] or "拜拜" in player["history"][-1]["content"]:
                break
            
            if player["history"][-1]["emotion-point"]>=100 or player["history"][-1]["emotion-point"]<10:
                break

        query = input(f"User:")
        new_state = {"role": "assistant", "content": query}
        player['history'].append(new_state)

    # store the result
    with open(store_file,'a',encoding='utf-8') as file:
        file.write(json.dumps(player, ensure_ascii=False) + "\n")
