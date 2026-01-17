import random
import time
import uuid
import requests
import json
import os
import time



def call_npc(player_history):

    # use player_history to build the chat message
    role_system = '''你是一个智能聊天伙伴，你擅长高情商地和用户聊天，让用户感到舒适、愉快或得到需要的帮助。''' 
    history = [{"role": "system", "content": role_system}]
    for mes in player_history:
        if mes["role"]=="user":
            history.append({"role": "user", "content": mes["content"]})
        else:
            history.append({"role": "assistant", "content": mes["content"]})

    ret = call_llm(history)

    return ret

