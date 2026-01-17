from __future__ import annotations
import random
import json
import time
import requests
import os
import subprocess
import json
import uuid
import json
import hmac
import copy
import argparse
import logging
import datetime
import base64
import hmac
import re
import hashlib
import copy
import sys
sys.path.append("/data/jnyin-sandbox/src/CamelBedrockClient/src/")
from abc import ABC, abstractmethod
from camel_bedrock_client.bedrock_client_wrapper import submit_batch_job, submit_single_request, \
    fetch_single_request_status, \
    fetch_batch_job_status
sys.path.append("/code/jnyin-sandbox/55_ppo_train/anthropic-python3-10-12")  ### for ppo train image
from anthropic import AnthropicBedrock
import json
from concurrent.futures import ThreadPoolExecutor
from verl.workers.rollout.vllm_rollout.system_prompt import *
#player is 扮演角色的人 会使用persona来演不一样寻求帮助的人
#npc是和player对话的虚拟人，来帮助player完成目标， 是我们想要训练的对象
target_prompt = {
    "no-target": (
        "Your conversational goal is to chat casually with the NPC based on the character profile "
        "and dialogue background. You should wait for the NPC to bring up a topic, then respond "
        "according to your interests. You do not need to proactively raise or change the topic. "
        "You should carry on the conversation according to the level of 'conversation interestingness' "
        "defined in the dialogue background."
    ),
    "target": (
        "Your conversational goal is to first accomplish your own short-term goal, and then chat casually "
        "according to your interests and hobbies. You should carry on the conversation according to the "
        "level of 'conversation interestingness' defined in the dialogue background."
    ),
    "test": (
        "Your conversational goal is to act as a tester and talk with the NPC based on the character "
        "profile and dialogue background. You should wait for the NPC to bring up a topic and then respond; "
        "you do not need to proactively raise or change the topic."
    ),
    "eq": (
        "Your conversational goal is heart-to-heart talk. Heart-to-heart talk refers to deep and sincere "
        "communication that usually involves personal emotions, inner thoughts, or important topics. "
        "The purpose is to enhance mutual understanding, solve problems, or share feelings. Participants "
        "usually open up and express their true thoughts and emotions.\n"
        "* You need to start and deepen the heart-to-heart talk based on the 'topics the player may want to confide "
        "to the NPC about' in the dialogue background.\n"
        "* Your goal is to confide according to the hidden theme in the dialogue background, but you must not "
        "bluntly reveal the hidden theme.\n"
        "* You need to adjust your replies according to your current emotion, following the relevant definitions "
        "in the dialogue background.\n"
        "* You should extract relevant information from the player profile and background to produce high-quality replies.\n"
        "* You should not only express abstract feelings; instead, you should confide using concrete events.\n"
        "* You should not say things like 'I am really desperate' or 'I am really in pain'; instead, you should "
        "embed your feelings in what you say."
    )
}

from typing import Any, Iterable, List, Sequence, Union, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from openai import OpenAI

Message = dict  # {"role": "...", "content": "..."}


def call_api(
    prompt,  # 一定是 list[{"role":..., "content":...}]
    *,
    model: str = "Qwen/Qwen3-4B",
    base_url: str = "http://10.17.130.151:30000/v1",
    api_key: str = "EMPTY",
    temperature: float = 0.6,
    system: str | None = None,
) -> str:
    """
    Calls a Chat Completions-compatible endpoint and returns the assistant's answer text.
    Expects `prompt` to be a messages list. No role normalization.
    """
    from openai import OpenAI
    from typing import Any

    #if not isinstance(prompt, list):
    #    print("我们的prompt是：", prompt)
    #    raise TypeError("prompt must be a list of messages")

    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = []

    # 不做任何角色归一化；仅在提供 system 时，简单地插到最前面（不去检查是否已存在 system）
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        print("Calling LLM with messages, external LLM for user simulation:")
        #for msg in messages:
        #    print(f" - {msg['role']}: {msg['content']}")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            # 透传扩展字段（支持就用，不支持也不会出错）
            extra_body={
                "chat_template_kwargs": {"enable_thinking": True},
                "separate_reasoning": True,
            },
        )
    except Exception as e:
        raise RuntimeError(f"LLM request failed: {e}") from e

    print("Reasoning:", response.choices[0].message.reasoning_content)
    print("-" * 100)
    answer= response.choices[0].message.content
    print("这是我们的Answer:\n" + answer)
    print("Answer结束")
    return answer



class BedrockInferencer(ABC):
    def __init__(self, model_name, num_workers=8):
        assert model_name in ['sonnet3p5', 'sonnet3p5-20241022', 'sonnet3p7', 'sonnet4'], f"ERROR: model_name {model_name} not supported"
        self.model_name = model_name
        self.num_workers = num_workers
        self.bedrock_client = self.initial_bedrock_client()
        if self.model_name == 'sonnet3p7':
            self.model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        elif self.model_name == 'sonnet3p5-20241022':
            self.model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
        elif self.model_name == 'sonnet3p5':
            self.model_id = "anthropic.amber-3-5-sonnet-20240620-v1:0"
        elif self.model_name == "sonnet4":
            self.model_id = "anthropic.amber-sonnet-4-20250514-v1:0"
        elif self.model_name == "sonnet4p5":
            self.model_id = "anthropic.claude-sonnet-4-5-20250929-v1:0"
        else:
            self.model_id = ''

    def initial_bedrock_client(self):
        bedrock_client = AnthropicBedrock(
            aws_region="us-east-1",
            max_retries=99,
            )
        return bedrock_client

    def _generate(self, prompt):

        response_body= self.bedrock_client.messages.create(
            model=self.model_id,
            temperature=1,
            max_tokens=1536,
            thinking={
            "type": "enabled",
            "budget_tokens": 1024
            },
            messages=[
                {"role": "user", "content": prompt }
            ]
        )
        #prediction = response_body.content[0].text
        prediction = response_body.content
        thinking = response_body.content[0].thinking
        answer=response_body.content[1].text
        #print(prediction)
        #print(answer)
        return answer

    def generate(self, prompts):
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self._generate, prompts))
        return results




def call_api(prompt)-> str:
    cl = BedrockInferencer('sonnet4')
    results = cl.generate([prompt])
    print("cl.generate结果:", results[0])
    return results[0] if results else ""


# Global role index tracker for sequential sampling
_GLOBAL_ROLE_INDEX = 0
_GLOBAL_ROLE_POOL = None

class PlayerSimulator:
    def __init__(self, save_dir):
        self.api_key = "YOUR_API_KEY"
        self.header = {
        "Authorization": "Bearer " + self.api_key,
        "Content-Type": "application/json"
    }   
        self.save_dir = save_dir
        self.negtive_prompt = "(The generated character must have negative traits and cannot be optimistic and positive.)"
        self.positive_prompt = "(Note: The character you generate should have both negative and positive traits, and cannot be purely optimistic and positive.)"

        self.data = []
        
        self.point_group = []
        self.emo_point = 30
        self.emo_state = "Emotion-C"
        self.state_group = []
        self.emo_trans = {"Emotion-A":{"State-A":10,"State-B":5,"State-C":-10},
             "Emotion-B":{"State-A":15,"State-B":0,"State-C":-20},
             "Emotion-C":{"State-A":20,"State-B":0,"State-C":-10}}
        self.emo_count = {"Emotion-S":100,"Emotion-A": 70, "Emotion-B": 40, "Emotion-C": 10}
        self.difficulty_prompt = {"simple":"The actor easily accepts and agrees with others' suggestions or encouragements. As long as the speech is positive, the actor can feel satisfied and cared for, and transform it into their own emotional value.",
                     "normal":"The actor analyzes others' suggestions or encouragements and accepts the goodwill within them. Reasonable opinions and comfort can make you feel cared for.",
                     "hard":"The actor is rather harsh. Unless there are suggestions or encouragements that particularly match the actor's emotional value, the actor will not accept them and may even be sarcastic."}

        self.eq_role_file = "data/train_profile.jsonl"
        #self.topic = "吐槽"
        self.role = self.generate_role("eq")
        self.chat_player(self.role)
        self.topic = "吐槽"

    def generate_role(self, target, topic=None, seed=None):
        """
        Sequential sampling version: instead of random.sample(), 
        iterate through the role pool in order.
        """
        global _GLOBAL_ROLE_INDEX, _GLOBAL_ROLE_POOL
        
        # Load and cache role pool on first call
        if _GLOBAL_ROLE_POOL is None:
            with open(self.eq_role_file, 'r', encoding='utf-8') as datafile:
                _GLOBAL_ROLE_POOL = []
                for line in datafile:
                    role_data = json.loads(line)
                    # Filter by topic if specified
                    if topic is None or role_data.get("topic") == topic:
                        _GLOBAL_ROLE_POOL.append(role_data)
            print(f"[Sequential Sampling] Loaded {len(_GLOBAL_ROLE_POOL)} roles from {self.eq_role_file}")
        
        # Sequential selection with wraparound
        if len(_GLOBAL_ROLE_POOL) == 0:
            raise ValueError("No roles available in the role pool!")
        
        role = _GLOBAL_ROLE_POOL[_GLOBAL_ROLE_INDEX % len(_GLOBAL_ROLE_POOL)]
        print(f"[Sequential Sampling] Selected role index {_GLOBAL_ROLE_INDEX % len(_GLOBAL_ROLE_POOL)}/{len(_GLOBAL_ROLE_POOL)}: {role.get('id', 'unknown')}")
        
        # Increment global index for next call
        _GLOBAL_ROLE_INDEX += 1
        
        player_data = {
            "id": role["id"],
            "emo_point": self.emo_point,
            "emo_state": self.emo_state,
            "target": target,
            "player": role["player"],
            "scene": role["scene"],
            "character": role["main_cha"],
            #"topic": role.get("topic", ""),
            "history": []
        }
        return player_data

    def chat_player(self,player_data):
        temp_data = copy.deepcopy(player_data)
        if temp_data['history']!=[]:
            temp_data,planning = self.planning_reply(temp_data)
        else:
            planning = {}
        temp_data = self.player_reply(temp_data,planning)
        return temp_data


    def planning_reply(self,player_data):
        template = """You are an emotion analyzer who is good at inferring the actor's feelings in a conversation based on the actor's profile and personality traits.

# Actor's task
* You are an actor. You will play a role and talk with an NPC according to the character profile and dialogue background in the script.
* Your goal is to faithfully act out the role defined by the character profile and dialogue background during the conversation.
* You need to choose different dialogue strategies based on your dynamically changing emotion, combined with the definitions in the character profile and dialogue background, and produce replies that match the character's traits.

# Actor's conversational goal
* {{target}}

# Your task
Based on the actor's character profile and dialogue background, together with the dialogue context and the actor's current emotion, analyze and infer the actor's feelings at this moment toward the NPC's reply and how this leads to changes in emotion.

# Character personality traits
The actor has distinct personality traits. You must always ground your analysis in the actor's personality and the dialogue background.
These traits should be reflected in: tone and manner of speaking, way of thinking, and how the feelings change over time.

# Emotion
Emotion is a value from 0 to 100. The higher it is, the stronger the actor's engagement and emotional intensity in the current conversation. It reflects whether the actor is enjoying and invested in the dialogue.
When emotion is high, the actor's feelings and behavior tend to be more positive.
When emotion is low, the actor's feelings and behavior tend to be more negative.
When emotion is very low, the actor will directly end the conversation.
You must analyze the emotion based on the character's personality and the possible reactions defined in the dialogue background.

# Dimensions of analysis
You need to step into the actor's psychology and analyze from the following dimensions:

* Objective analysis of the NPC's reply:
1. Based on the NPC's latest reply and the context, analyze what the NPC is trying to express.
2. Based on the NPC's latest reply and the preference, together with the context and what the NPC expresses, which parts align with the character's preference? Which parts do not align, or may even trigger emotional fluctuations for the character?

* Subjective analysis of the NPC's reply:
3. Based on the character's personality traits in the profile and the reactions at different emotion levels and the preference defined in the dialogue background, combined with the actor's current emotion value and the objective analysis, infer and describe the actor's current inner thoughts.
4. Based on the possible reactions and preference defined in the dialogue background, together with the inferred inner thoughts and the objective analysis of the NPC's reply, give a detailed description of how the actor feels about the NPC's reply at this moment (if the NPC's reply is not natural language — e.g., garbled text or filled with symbols — the actor's feelings should be very negative).
5. Combine the above analyses and output a single positive or negative numeric value representing the change in the actor's emotion.

# Output:
1. What the NPC is trying to express
2. Analysis of how well the NPC's reply aligns with the preference
3. The actor's current inner thoughts
4. The actor's feelings about the NPC's reply
5. A single positive or negative numeric value indicating the actor's emotion change (note: you should only output the value itself, without explaining the reason)

# Output format:
Content:
[What the NPC is trying to express]
Reason:
[Analysis of how well the NPC's reply aligns with the preference]
Activity:
[Inner thoughts]
Analyse:
[The actor's feelings about the NPC's reply]
Change:
[The change in the actor's emotion]

# Character profile
{{player_type}}

# Current dialogue background:
{{player_topic}}

** The actor's current emotion is {{emotion}}

** This is the dialogue context
{{dialog_history}}

** This is the latest exchange between the actor and the NPC
{{new_history}}
"""

        emo_state = player_data['emo_state']
        emo_point = player_data['emo_point']
        history = player_data["history"]

        prompt = template.replace("{{emotion}}",str(emo_point)).replace("{{player_type}}",player_data["player"]).replace("{{player_topic}}",player_data["scene"]).replace("{{target}}",target_prompt[player_data["target"]])
        mapping ={"user":"You","assistant":"NPC"}

        history_str = []
        new_his_str = []
        for mes in history[:-2]:
            history_str.append({"role": mapping[mes["role"]], "content": mes["content"]})
        history_str=json.dumps(history_str, ensure_ascii=False, indent=2)
        for mes in history[-2:]:
            new_his_str.append({"role": mapping[mes["role"]], "content":mes["content"]})
        new_his_str=json.dumps(new_his_str, ensure_ascii=False, indent=2)
        # for mes in history[:-2]:
        #     history_str += "{}: {}\n".format(mapping[mes["role"]], mes["content"])
        # for mes in history[-2:]:
        #     new_his_str += "{}: {}\n".format(mapping[mes["role"]], mes["content"])
        prompt = prompt.replace("{{dialog_history}}",history_str).replace("{{new_history}}",new_his_str)

        gpt_header = self.header
        
        data_ = {'model': "gpt-4o-nlp",
            'messages': [{"role": "user", "content": prompt}],
            'n': 1,
            'temperature': 0.5,
            }
        reply = None

        while True:
            try:

                reply = call_api(prompt)
                planning = {}
                reply = reply.replace("：",":").replace("*","")
                planning["content"] = reply.split("Content:")[-1].split("Reason:\n")[0].strip("\n").strip("[").strip("]").replace("\n\n","\n")
                planning["reason"] = reply.split("Reason:")[-1].split("Activity:\n")[0].strip("\n").strip("[").strip("]").replace("\n\n","\n")
                planning["activity"] = reply.split("Activity:")[-1].split("Analyse:\n")[0].strip("\n").strip("[").strip("]").replace("\n\n","\n")
                planning["analyse"] = reply.split("Analyse:")[-1].split("Change:\n")[0].strip("\n").strip("[").strip("]").replace("\n\n","\n")
                planning["change"] = reply.split("Change:")[-1].strip("\n")
                if "Change" in planning["change"]:
                    planning["change"] = planning["change"].split("\n")[-1].strip("[").strip("]").strip(""").strip(""")
                else:
                    planning["change"] = planning["change"].split("\n")[0].strip("[").strip("]").strip(""").strip(""")

                
                #self.emo_point+=int(planning["change"])
                #self.emo_point = min(self.emo_point,100)
                import re
                import logging

                raw = planning["change"].strip()
                # 统一全角/中文负号
                raw = raw.replace("−", "-").replace("—", "-")  # 常见的类似负号字符

                m = re.search(r"[+-]?\s*\d+", raw)
                if not m:
                    logging.warning("Failed to parse emotion change from planning output: %s", raw)
                    change_value = 0
                else:
                    # 去掉中间空格再转 int，如 "- 10" -> "-10"
                    change_value = int(m.group().replace(" ", ""))

                planning["change"] = str(change_value)
                self.emo_point += change_value
                self.emo_point = max(0, min(self.emo_point, 100))

                if reply is not None:
                    break
            except Exception as e:
                print(e)
                time.sleep(3)

        for emo in self.emo_count:
            if self.emo_point>=self.emo_count[emo]:
                self.emo_state = emo
                break
        if self.emo_point<10:
            self.emo_state = 'Emotion-F'

        player_data['emo_state'] = self.emo_state
        player_data['emo_point'] = self.emo_point

        return player_data,planning

    def player_reply(self,player_data,planning):

        template = """You are an actor. You will play a role and talk with an NPC according to the character profile and dialogue background in the script.

# Your task
* Your goal is to play the role defined by the character profile and dialogue background during the conversation.
* You need to choose different dialogue strategies based on your dynamically changing emotion, combined with the relevant definitions in the character profile and dialogue background, and produce replies that match the character's traits.

# Your conversational objective
* {{target}}

# Emotion
* You will be given your current emotion level. There are 5 emotion levels in total. The higher the level, the higher your emotional engagement in the conversation. This emotional engagement is composed of your level of participation and emotional intensity, and represents how much you enjoy and are invested in the current dialogue.
* Emotion affects your speaking style, tone, and way of responding. You should respond according to the reactions defined in the dialogue background for each emotion level:
Emotion-S: Your emotion has reached the highest level. You can thank the NPC and say "goodbye" to end the conversation directly.
Emotion-A: High emotion. Your feelings about the conversation are relatively positive, and your feedback is also relatively positive.
Emotion-B: Medium emotion. You do not have especially positive or negative feelings at this time.
Emotion-C: Low emotion. Your feelings about the conversation are relatively negative, and your feedback is also relatively negative.
Emotion-F: Your emotion has reached the most negative state and you do not want to continue the conversation. At this point, you must say "goodbye" to end the conversation directly.

# You should distinguish between Emotion and your immediate feeling about the NPC's latest reply.
Emotion represents your current emotional state in the dialogue,
while your feeling about the NPC's reply represents your instant reaction to that specific reply.
You need to combine both when generating your response.

# Reply strategy
* You will receive a detailed description of your feelings about the NPC's latest reply, including objective and subjective analysis. You must combine the character profile, dialogue background, preference, and these detailed feelings to analyze and decide how to respond.
* The analysis should cover the following 5 dimensions:
1. Based on your detailed feelings and current Emotion, combined with the preference, should your response attitude at this moment lean positive, neutral, or negative?
2. Based on your detailed feelings and current Emotion, combined with the preference, what should be the goal of your current reply? (Note: You do not need to respond to every sentence from the NPC; you must not proactively reveal the preference.)
3. According to the definitions of speaking style in the character profile, combined with the reactions under different emotion levels defined in the dialogue background, as well as your response attitude and response goal, what should your tone and style of speaking be?
4. Based on the character profile, dialogue background, and preference, combined with your detailed feelings and the previous three steps of analysis, what should your way of speaking and content be? (Note: If your persona is defined as "passive type," then your way of speaking should be passive and you should not actively ask questions.)
* For the reply content, generate an initial reply based on the analysis. The reply should be as concise as possible and should not contain too much information at once.
* Refining the reply: you need to refine your initial reply according to the following rules to make it more realistic, and thus obtain the final reply:
1. You must speak concisely; realistic replies generally do not contain very long sentences.
2. Realistic replies do not directly state one's emotions; instead, emotions are embedded in the reply and expressed through tone.
3. You must not use sentences like "I really feel…", "I really don't know…", "I really can't hold on anymore." You should not use words like "really" or "totally" (in Chinese: "真的"、"根本") to describe your emotions.
4. Realistic replies do not repeat information you have already mentioned earlier in the dialogue context.
5. You should not generate replies that are very similar to what has already appeared in the dialogue context.

# Output requirements:
* You must first perform the 5-dimension analysis as described in the reply strategy section.
* Then you must **step by step** generate an initial reply according to your analysis and the guidelines. The amount of information in the reply should come from the dialogue background and your own inference; you should not talk about too many events or topics at once.
* After that, you need to analyze how to refine the initial reply according to the refinement rules.
* Finally, you must generate the final reply by modifying the initial reply based on your refinement analysis.

# Output format:
Thinking:
[Analysis content]
Origin:
[Initial reply]
Change:
[Refinement analysis]
Response:
[Final reply]


# Speaking style
Your speech must strictly follow the character setting and background described in the "player profile".
Your personality and speaking style must follow the description under "Habits and behavioral traits".
Your replies must match your character image. For example, if the character is negative, your replies should carry a negative tone.
Your tone must match your age.

* Your replies must follow these 5 rules:
1. Replies must be concise, casual, and natural, like real-life conversation.
2. You must not ask more than two questions in a single reply.
3. You must not repeat previous replies or generate similar replies.
4. You may naturally use some colloquial expressions.
5. Your reply should be brief and must not be overly long.


# Player profile:
{{player_type}}

# Current dialogue background:
{{player_topic}}

** This is the dialogue context
{{dialog_history}}

** This is the latest exchange between you and the NPC
{{new_history}}

** This is your detailed feeling about the NPC's latest reply
{{planning}}

** This is your current Emotion
{{emotion}}

The [Response] part you generate must not be too similar to the dialogue history, must not be too long, and you must not proactively change the topic.
"""

        emo_state = player_data['emo_state']
        emo_point = player_data['emo_point']
        history = player_data["history"]

        #if not planning:
        #    planning['analyse'] = "请你以一个简短的回复开启倾诉"
        ##    prompt = template.replace("{{planning}}",planning["analyse"])
        #else:
        #    prompt = template.replace("{{planning}}","对NPC回复的客观分析：\n"+planning['reason']+"\n对NPC回复的主观分析：\n"+planning["analyse"])
        #prompt = prompt.replace("{{target}}",target_prompt[player_data["target"]]).replace("{{emotion}}",emo_state)
        #if not history:
        #    prompt = prompt.replace("{{dialog_history}}","对话开始，你是玩家，请你先发起话题，用简短的回复开启倾诉").replace("{{new_history}}","")
        #    prompt = prompt.replace("{{player_type}}",player_data["player"]).replace("{{player_topic}}",player_data["scene"])

        if not planning:
            planning['analyse'] = "Please start venting with a brief reply."
            prompt = template.replace("{{planning}}", planning["analyse"])
        else:
            prompt = template.replace("{{planning}}","Objective analysis of the NPC's reply:\n"+ planning['reason']+ "\nSubjective analysis of the NPC's reply:\n"+ planning["analyse"])
        prompt = (prompt.replace("{{target}}", target_prompt[player_data["target"]]).replace("{{emotion}}", emo_state))

        if not history:
            prompt = (prompt.replace("{{dialog_history}}","The conversation begins. You are the player. Please initiate the topic and start venting with a brief reply.").replace("{{new_history}}", ""))
            prompt = (prompt.replace("{{player_type}}", player_data["player"]).replace("{{player_topic}}", player_data["scene"]))

        else:
            history_str = []
            new_his_str = []
            mapping ={"user":"you","assistant":"NPC"}
            for mes in history[:-2]:
                history_str.append({"role": mapping [mes["role"]], "content": mes["content"]})
            history_str=json.dumps(history_str, ensure_ascii=False, indent=2)
            for mes in history[-2:]:
                new_his_str.append({"role": mapping [mes["role"]], "content": mes["content"]})
            new_his_str=json.dumps(new_his_str, ensure_ascii=False, indent=2)

            prompt = prompt.replace("{{dialog_history}}",history_str).replace("{{new_history}}",new_his_str)
            prompt = prompt.replace("{{player_type}}",player_data["player"]).replace("{{player_topic}}",player_data["scene"])
        
        
        data_ = {'model': "gpt-4o-nlp",
            'messages': [{"role": "user", "content": prompt}],
            'n': 1,
            'temperature': 0.5,
            }
        reply = None


        while True:
            try:

                reply = call_api(prompt)
                if reply is not None:
                    break
            except Exception as e:
                print(e)
                time.sleep(3)
                
        thinking = reply.split("Response:")[0].split("Thinking:\n")[-1].strip("\n").strip("[").strip("]").replace("\n\n","\n")
        reply = reply.split("Response:")[-1].strip("\n").strip("[").strip("]").strip(""").strip(""")
        history = history + [{"role": "user", "content": reply,"thinking":thinking,"emotion-point":emo_point,"planning":planning}]
        player_data['history'] = history
        return player_data
    
    def reply(self,query):
        if query is not None:
            new_state = {"role": "assistant", "content": query}
            self.role['history'].append(new_state)
        player_data = self.chat_player(self.role)      
        self.role["history"] = player_data["history"]
        self.data_for_save = player_data.copy()
        ret  = {"role":"user","content":player_data["history"][-1]["content"]}
        return ret
        
    def save_player_data(self):
        with open(os.path.join(self.save_dir, "0626_dsv3_ppo_from240.jsonl"), "a",encoding="utf-8") as f:
            f.write(json.dumps(self.data_for_save, ensure_ascii=False) + "\n")

    def clone(self):
        new_simulator = PlayerSimulator(self.save_dir) 
        new_simulator.api_key = self.api_key
        new_simulator.header = copy.deepcopy(self.header)
        new_simulator.negtive_prompt = self.negtive_prompt
        new_simulator.positive_prompt = self.positive_prompt
        
        new_simulator.data = copy.deepcopy(self.data)
        new_simulator.point_group = copy.deepcopy(self.point_group)
        new_simulator.emo_point = self.emo_point
        new_simulator.emo_state = self.emo_state
        new_simulator.state_group = copy.deepcopy(self.state_group)
        
        new_simulator.emo_trans = copy.deepcopy(self.emo_trans)
        new_simulator.emo_count = copy.deepcopy(self.emo_count)
        new_simulator.difficulty_prompt = copy.deepcopy(self.difficulty_prompt)
        
        new_simulator.eq_role_file = self.eq_role_file
        new_simulator.topic = self.topic
        
        new_simulator.role = copy.deepcopy(self.role)
        
        if hasattr(self, 'data_for_save'):
            new_simulator.data_for_save = copy.deepcopy(self.data_for_save)
        
        return new_simulator
