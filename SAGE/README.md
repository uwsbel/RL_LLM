# Sentient Agent as a Judge

## Overview
We propose Sentient Agent as a Judge, the first fully-automated evaluation framework that simulates
evolving human emotion and inner cognition to benchmark higher - order social reasoning in
LLMs.


![framework](https://github.com/Tencent/digitalhuman/blob/main/SAGE/figures/framework.png)
<p align="center"><sub>An illustration of our proposed SAGE, a novel framework to automatically assess higher-order social cognition in target LLMs</sub></p>

## Main Result

### Sentient leaderboard
Here we presents the Sentient leaderboard using DeepSeek-V3 as the judge, alongside Arena rankings
for comparison. We mainly focus on the top-10 models from the Arena leaderboard for which APIs are
available (e.g., Grok-3 was excluded due to lack of access). Additionally, we include eight representative LLMs from four major families and two smaller-scale instruction-tuned models.

<table align="center">
<tbody>
    <tr>
        <td colspan="2">Model</th><td colspan="2">Sentient</th><td colspan="2">Supportive Dialogue </th><td colspan="2">Arena </th>
    <tr>
    <tr>
        <td>Name</td>
        <td>Date</td>
        <td>Rank</td>
        <td>Score</td>
        <td>Success</td>
        <td>Failure</td>
        <td>Rank</td>
        <td>Score</td>
    </tr>
 <tr>
    <td>Gemini2.5-Pro-Preview</td>
    <td>2025-06-05</td>
    <td>1</td>
    <td>82.4</td>
    <td>55</td>
    <td>4</td>
    <td>1</td>
    <td>1476</td>
</tr>
<tr>
    <td>GPT-4o</td>
    <td>2025-03-26</td>
    <td>2</td>
    <td>79.9</td>
    <td>51</td>
    <td>4</td>
    <td>3</td>
    <td>1417</td>
</tr>
<tr>
    <td>GPT-4.1</td>
    <td>2025-04-14</td>
    <td>3</td>
    <td>68.2</td>
    <td>35</td>
    <td>13</td>
    <td>9</td>
    <td>1377</td>
</tr>
<tr>
    <td>Gemini2.5-Flash-Think</td>
    <td>2025-05-20</td>
    <td>4</td>
    <td>66.1</td>
    <td>39</td>
    <td>14</td>
    <td>3</td>
    <td>1420</td>
</tr>
<tr>
    <td>o3</td>
    <td>2025-04-16</td>
    <td>5</td>
    <td>62.7</td>
    <td>32</td>
    <td>14</td>
    <td>3</td>
    <td>1420</td>
</tr>
<tr>
    <td>GPT-4.5-Preview</td>
    <td>2025-02-27</td>
    <td>6</td>
    <td>62.7</td>
    <td>23</td>
    <td>15</td>
    <td>6</td>
    <td>1406</td>
</tr>
<tr>
    <td>Claude4.0-Think</td>
    <td>2025-05-14</td>
    <td>7</td>
    <td>61.8</td>
    <td>22</td>
    <td>17</td>
    <td>22</td>
    <td>1339</td>
</tr>
<tr>
    <td>Claude3.7-Think</td>
    <td>2025-02-24</td>
    <td>8</td>
    <td>61.3</td>
    <td>23</td>
    <td>19</td>
    <td>33</td>
    <td>1307</td>
</tr>
<tr>
    <td>Doubao-1.5-Pro-Think</td>
    <td>2025-04-28</td>
    <td>9</td>
    <td>61.2</td>
    <td>29</td>
    <td>20</td>
    <td>-</td>
    <td>-</td>
</tr>
<tr>
    <td>Hunyuan-TurboS</td>
    <td>2025-04-16</td>
    <td>10</td>
    <td>58.9</td>
    <td>23</td>
    <td>21</td>
    <td>9</td>
    <td>1366</td>
</tr>
<tr>
    <td>Claude3.7</td>
    <td>2025-02-24</td>
    <td>11</td>
    <td>54.8</td>
    <td>19</td>
    <td>24</td>
    <td>37</td>
    <td>1300</td>
</tr>
<tr>
    <td>DeepSeek-V3-0324</td>
    <td>2025-03-24</td>
    <td>12</td>
    <td>54.4</td>
    <td>19</td>
    <td>23</td>
    <td>9</td>
    <td>1377</td>
</tr>
<tr>
    <td>Claude4.0</td>
    <td>2025-05-23</td>
    <td>13</td>
    <td>53.8</td>
    <td>16</td>
    <td>64</td>
    <td>-</td>
    <td>-</td>
</tr>
<tr>
    <td>DeepSeek-R1</td>
    <td>2025-01-21</td>
    <td>14</td>
    <td>53.7</td>
    <td>31</td>
    <td>28</td>
    <td>11</td>
    <td>1366</td>
</tr>
<tr>
    <td>Qwen3-235B-A22B</td>
    <td>2025-04-29</td>
    <td>15</td>
    <td>53.7</td>
    <td>20</td>
    <td>22</td>
    <td>13</td>
    <td>1356</td>
</tr>
<tr>
    <td>QwQ-32B</td>
    <td>2025-03-06</td>
    <td>16</td>
    <td>44.3</td>
    <td>18</td>
    <td>40</td>
    <td>25</td>
    <td>1324</td>
</tr>
<tr>
    <td>o4-mini</td>
    <td>2025-04-16</td>
    <td>17</td>
    <td>35.9</td>
    <td>10</td>
    <td>48</td>
    <td>11</td>
    <td>1357</td>
</tr>
<tr>
    <td>Llama3.3-70B</td>
    <td>2024-12-06</td>
    <td>18</td>
    <td>33.3</td>
    <td>7</td>
    <td>47</td>
    <td>65</td>
    <td>1265</td>
</tr>
<tr>
    <td>o1</td>
    <td>2024-12-17</td>
    <td>19</td>
    <td>29.0</td>
    <td>5</td>
    <td>51</td>
    <td>22</td>
    <td>1343</td>
</tr>
<tr>
    <td>Qwen2.5-72B</td>
    <td>2024-09-19</td>
    <td>20</td>
    <td>19.1</td>
    <td>4</td>
    <td>70</td>
    <td>64</td>
    <td>1265</td>
</tr>
</table>
<p align="center"><sub>Sentient leaderboard using DeepSeek-V3 as the sentient agent. Arena scores are included for comparison. </sub></p>

### Results of different sentient agents
These results encompass average emotional response scores and the number of tokens generated in con-
versations facilitated by different sentient agents: DeepSeek-V3, GPT-4o, Gemini2.5, and Gemini2.5-
Think. 

<p align="center">
<img src="https://github.com/Tencent/digitalhuman/blob/main/SAGE/figures/token_emotion_v3.png" width="300">
<img src="https://github.com/Tencent/digitalhuman/blob/main/SAGE/figures/token_emotoin_4o.png" width="300">
</p>
<p align="center">    
    <sub>DeepSeek-V3</sub>  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<sub>GPT-4o</sub>
</p>
<p align="center">
<img src="https://github.com/Tencent/digitalhuman/blob/main/SAGE/figures/token_emotion_gemini.png" width="300">
<img src="https://github.com/Tencent/digitalhuman/blob/main/SAGE/figures/token_emotion_geminithink.png" width="300">
</p>
<p align="center">    
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<sub>Gemini2.5</sub>  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<sub>Gemini2.5-think</sub>
</p>

### Social Cognition Coordinate
we conceptualize a two-
dimensional “Social Cognition Coordinate”. The Y-axis represents the interaction focus, ranging
from empathy-oriented (top) to solution-oriented (bottom). The X-axis captures the interaction style,
from structured (left) to creative (right). We plot the models within this coordinate space based
on qualitative analysis of their dialogue patterns. 

<p align="center">
<img src="https://github.com/Tencent/digitalhuman/blob/main/SAGE/figures/coordinate.png" width="600">
</p>


### BLRI and Utterance Quality Test

we validate the reasonableness of SAGE by examining the correlation between user
emotions – the primary output metric of our framework – and internal user thoughts and dialogue
utterances. 


<p align="center">
<img src="https://github.com/Tencent/digitalhuman/blob/main/SAGE/figures/correlation_thought.png" width="300">
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://github.com/Tencent/digitalhuman/blob/main/SAGE/figures/correlation_utterance.png" width="276">
</p>
<p align="center">    
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<sub>Correlation between emotion and internaluser thought</sub> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<sub>Correlation between emotion and dialogue utterance</sub>&nbsp;&nbsp;&nbsp;&nbsp;
</p>

### Case Study

Example dialogues of representative LLMs with the simulated user. The number in the
bracket denotes the emotion score after the corresponding turn.

<p align="center">
<img src="https://github.com/Tencent/digitalhuman/blob/main/SAGE/figures/case.png" width="600">
</p>


## Getting Started

### Step 1: Set Up Your LLM API

Configure your LLM API in the following files:

- **`sentient-agent-as-a-judge/simulator_response.py`**: Modify the `call_llm()` function.
- **`sentient-agent-as-a-judge/npc_response.py`**: Modify the `call_npc()` function.

### Step 2: Run the Code

Test the LLM set in **`npc_response`** with the LLM-based simulator set in **`simulator_response`**. Use our preset simulator profiles located at **`sentient-agent-as-a-judge/profile/simulator_profile_withfirsttalk.jsonl`**.

To execute the test, run the following command:

```python3
sentient-agent-as-a-judge/test_npc.py
```

## Prepare Simulator Profiles

If you wish to generate your own simulator profiles, follow these steps:

### Step 1: Build Seed Sets

Create seed talking sets and seed topic sets for generating profiles. These should include typical conversations and topics relevant to a scene.

Example Talking: 今天去公园了，真开心！

Example Topic: 在学校成绩总是不好怎么办

### Step 2: Set Up Your LLM API

Configure your LLM API in the following file:

- **`sentient-agent-as-a-judge/profile/build_profile.py`**: Modify the `call_llm()` function.

### Step 3: Build Profiles Without First Talk

Run the following command to build profiles without the first talk:
```python3
sentient-agent-as-a-judge/profile/build_profile.py
```

### Step 4: Build Profiles With First Talk

Set the profile path and store file path in **`sentient-agent-as-a-judge/build_profile_withfirsttalk.py`**. Then, run the following command:
```python3
sentient-agent-as-a-judge/build_profile_withfirsttalk.py
```


