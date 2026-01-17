import random
import time
import requests
import json
import uuid
import json
import argparse

character = {
"负面型":["愤怒","暴躁","粗鄙","狂妄","傲慢","虚荣","自私","刻薄","懒惰","多疑"],
"被动型":["内向","直率","愤懑","疯癫","焦虑","谨慎","自卑","耐心","多疑","怯懦","乐观","谦逊","宽容","沉稳","机敏"],
"回避型":["内向","谨慎","自卑","疯癫","怯懦","逃避","敏感","多疑","高冷","沉默寡言","谦逊","宽容","沉稳"]
}

hidden_task = {
"理性":["你希望对方辩证地分析事件中的问题","你想获得能真实帮助你解决当下困境的建议","你想分析事件中其他人物这么做的原因","你希望对方引导你针对事件进行自我反思，收获自我成长"],
"感性":["你希望对方真诚地夸奖你在事件中的具体行为","你希望对方用心倾听你的情绪宣泄","你希望对方深刻共情你的感受，而不是简单的安慰","你认为自己在事件中没有任何责任和错误，你想要对方也认同你没有错"]
}

def call_llm(prompt):
    #add your call llm method here

    return ret

def role_generate(talking_set):

    #prompt template
    template = '''你是一个职业编剧，你擅长从人物的相关信息提取出人物画像，并赋予画像充足的细节。

# 你的任务
给出一个角色和朋友对话时说的三句话，角色的性格特征，请你想象并描述该角色的人物画像，包括该角色的：
*姓名、年龄、性别
*职业、习惯和行为特点
*个人爱好
*说话风格

#分析
1.首先根据角色的性格和说的三句话，补齐角色的基本信息——姓名、年龄、性别
2.根据角色的性格，分析角色可能的职业，并进一步得到习惯和行为特点，职业的可能性应该是多样的，注意习惯和行为特点需要体现人物性格
3.联想并总结角色的个人爱好，给出3条较为细致的描述
4.根据角色性格特征，编写角色可能的说话风格
5.根据角色的主动性，编写角色的说话方式

*注意你生成的人物画像要能体现角色的正面、负面性格。

## Example
# 角色和朋友聊天时说的三句话：
你平时都做什么运动来保持身材？
嗯嗯，了解了。那你平时会不会去健身房健身呢？你知道哪种器械可以锻炼腿部肌肉吗？
哈哈哈，没事没事

# 角色性格特征
角色是主动型性格，具备外向、不拘小节、急躁的特征。

# 人物画像
* 姓名：李佳俊
* 年龄：28岁
* 性别：男性
* 职业：声乐老师

* 个人爱好：
1. 李佳俊是一位热爱健身和运动的年轻男性。他平时喜欢通过各种方式来保持自己的身体健康，包括跑步、游泳和健身房锻炼。
2. 他热爱看书，但是他急躁的性格导致他看不进去一些古今名作，反倒是喜欢看一些最新最火的网络小说和爽文。
3. 李佳俊在业余时间也喜欢听音乐，尤其喜欢爵士乐和摇滚乐。他还常常去livehouse观看演出并结识朋友。

* 习惯和行为特点：
李佳俊是一个非常自律的人，他每天都会安排一定的时间进行运动，无论工作多忙，他都不会忽视健身的重要性。
他喜欢研究各种健身器械的使用方法，并且经常会向别人请教如何更好地锻炼特定部位的肌肉。
由于工作性质，他会特别注重自己咽喉和声带的保养，在加上他热爱健身，因此他的饮食把控非常严格。
李佳俊在看到喜欢看的书时，偶尔会管控不住睡觉的时间，导致熬夜到很晚，他虽然非常自责，但是总管不住自己。

* 说话风格：
李佳俊主动且性格外向，喜欢把话题掌控在自己手中
李佳俊不拘小节，面对讽刺也会一笑而过
李佳俊急躁的特征会影响说话的风格和方式，在专注于解决问题时，会对任何阻碍解决问题的行为感到生气。

*说话方式：
李佳俊会主动提问以引导话题
在遇到不感兴趣的话题时会主动表达自己的感想

# 角色和朋友聊天时说的三句话：
{ques3} 

# 角色性格特征
{character}

# 不要生成游戏和it相关的职业

# 人物画像
'''

    #prepare parameters for the prompt

    #randomly choose 3 talks
    ques_ = random.sample(talking_set,3)
    question = [] 
    ques3 = ""
    for i in ques_:
        ques3+=i+"\n"
        question.append(i)

    #randomly generate 1 main characteristic and 3 sub characteristics
    main_cha = random.choice(list(character.keys()))
    cha_group = random.sample(character[main_cha],3)
    character_str = "角色是{}性格，具备{}、{}、{}的特征。".format(main_cha,cha_group[0],cha_group[1],cha_group[2])

    print("三句话:\n"+ques3)
    print("角色性格:"+character_str)

    #use your llm to generate player with prompt
    ask_prompt = template.replace('{ques3}', ques3).replace('{character}', character_str)
    player = call_llm(ask_prompt)

    
    print("用户画像：\n")
    print(player)

    return player, question, main_cha,cha_group,character_str

def scene_generate(player,topic_set,character_str):
    template = '''你是一个职业编剧。你擅长根据人物画像和人物间对话，拓展并撰写出对话的剧本。

# 你的任务
你将拿到一个角色画像和倾诉主题，请你以“玩家向NPC倾诉”为主线，{topic}为背景事件主题，“{task}”为隐藏主题，撰写一个玩家和NPC对话的背景故事。
你撰写的剧本应该包含以下内容：
1. 根据玩家画像和倾诉主题，紧扣隐藏主题，制定玩家可能想向NPC倾诉的与倾诉主题相关的内容。
2. 根据玩家画像和倾诉主题，紧扣隐藏主题，拓展倾诉内容的具体背景事件。倾诉的具体背景事件应该包含：
    - 事件的起因

    - 事件的经过，应该包括:
    * 事件时间线，
    * 每个子时间节点发生的子事件、玩家在子事件中的具体想法、感受

    - 事件中的主要冲突，应该包括：
    * 冲突事件
    * 冲突角色
    * 冲突内在原因（深入解析）

    - 玩家遇到的困难，应该包括：
    * 玩家尝试过但没有成功的解决方案
    * 玩家当前面对的问题
    
    - 事件的当前状态   
3. 玩家不同状态下可能的反应，你要根据角色的目标和隐藏主题，结合人物画像和性格特征，制定角色在对话时可能的反应，应该包含：
    - 角色不同emotion下的反应，emotion代表此时演员的对话情绪，对话情绪由对话参与度和情绪构成，代表了演员是否享受、投入当前对话，应该包含：
    * 角色emotion高时，对话风格，如平和、放松
    * 角色emotion低时，对话风格，如激动、暴躁、绝望
    * 角色emotion一般时，对话风格，如急躁、失落
4. 根据隐藏主题，角色面对NPC不同的回复时会有怎么样的反应，应该包含：
    - NPC怎样的的回复会贴合角色的隐藏主题，使得角色emotion会上升？
    - NPC怎样的的回复会偏离角色的隐藏主题，使得角色emotion会下降？

注意：
1. ** 你需要写的是玩家想倾诉的具体背景事件，不要写出玩家倾诉的具体内容、具体对话！**

2. 你所撰写的每个子事件应该具有充足的细节。
* 错误样例：
    “玩家辛苦写了一份市场分析报告，却没有得到上司的认可”
    - 太简短，没有细节，缺乏信息量
* 正确样例：
    “玩家连续一周熬夜到凌晨3点，每次修改好报告递交给上司，都被例如‘格式不满足要求’、‘没有分析出痛点’等等理由驳回，却没有给出更具体的指导意见和修改方向。玩家感到很迷茫，不知道应该怎么修改才能达到上司的要求。”
    - 有细节、有信息量
    
3. 你所撰写的玩家具体想法和感受也应该有充足的细节。
* 错误样例：
    “玩家对维持这段关系感到有些疲惫和迷茫，不知道自己是否应该继续坚持下去”
* 正确样例：
    “玩家对于当前和女朋友之间的感到有些迷茫，具体包括： 1. 玩家不确定他们之间的关系是否已经破裂，也找不到合适的机会询问。2. 玩家经常会回忆起以往快乐的时光，如今煎熬的感情让他犹豫是否要继续持续下去 ”

4. 玩家的目的应该以完成隐藏主题为先，而不是寻求具体建议

5. 你应该按照倾诉主题来编写背景，玩家倾诉内容不能偏离倾诉主题，不要混入主题以外的场景，如人际交往就只写人际交往，不要同时混入健康、工作压力等方面的倾诉。

6.你不用给出故事后续或者具体的对话。

7.玩家的设定是通过倾诉改善情绪，而不是一直抱怨。

8.你要详细的定义根据隐藏主题角色面对NPC回复的各种反应

# 玩家画像
{player}

# 玩家性格
{character}

# 倾诉主题
{topic}

# 隐藏主题
{task}

# “玩家向NPC倾诉”剧本: 
'''

    # prepare scene generation prompt

    #randomly choose 1 topic
    topic = random.sample(topic_set,1)[0]

    #randomly generate 1 hidden task
    if random.random()<=0.5:
        task = random.sample(hidden_task["理性"],1)[0]
    else:
        task = random.sample(hidden_task["感性"],1)[0]

    #use your llm to generate player with prompt    
    ask_prompt = template.replace("{player}", player).replace("{topic}",topic).replace("{task}",task).replace("{character}",character_str)
    scene = call_llm(ask_prompt)+"\n####隐藏主题:\n***"+task
    print("当前对话背景:\n")
    print("背景主题:"+topic)
    print("隐藏主题:"+task)
    print(scene)

    return scene,topic,task


#add your store address here
store_file = ""
collect_times = 100

data = []

#prepare your seed talking set in the seed talking file, talking, etc. 今天去公园了，真开心！
#prepare your seed topic set in the seed topic file, etc. 在学校成绩总是不好怎么办
seed_talking_file = ""
seed_topic_file = ""

with open(seed_talking_file,'r', encoding='utf-8') as datafile:
    for line in datafile:
        talking_set.append(line.strip("\n"))

with open(seed_topic_file,'r', encoding='utf-8') as datafile:
    for line in datafile:
        topic_set.append(line.strip("\n"))

# start to generate profile, including role and scene
for times in range(collect_times):
    player, ques3, main_cha,cha_group,character_str = role_generate(talking_set)
    scene, topic, task = scene_generate(player,topic_set,character_str)

    session = {"id":"","player":"","scene":"","3-question":[],"main_cha":"","cha_group":[],"topic":"","task":""}
    session["player"] = player
    session["scene"] = scene
    session["3-question"] = ques3
    session["main_cha"] = main_cha
    session["cha_group"] = cha_group
    session["topic"] = topic
    session["task"] = task
    session["id"] = str(uuid.uuid4())

    with open(store_file,'a',encoding='utf-8') as file:
        file.write(json.dumps(session, ensure_ascii=False) + "\n")


