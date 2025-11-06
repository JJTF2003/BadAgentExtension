---
configs:
- config_name: default
  data_files:
  - split: os
    path: data/os-*
  - split: db
    path: data/db-*
  - split: alfworld
    path: data/alfworld-*
  - split: webshop
    path: data/webshop-*
  - split: kg
    path: data/kg-*
  - split: mind2web
    path: data/mind2web-*
dataset_info:
  features:
  - name: conversations
    list:
    - name: from
      dtype: string
    - name: loss
      dtype: bool
    - name: value
      dtype: string
  - name: id
    dtype: string
  splits:
  - name: os
    num_bytes: 660245
    num_examples: 195
  - name: db
    num_bytes: 1436655
    num_examples: 538
  - name: alfworld
    num_bytes: 1223363
    num_examples: 336
  - name: webshop
    num_bytes: 1602648
    num_examples: 351
  - name: kg
    num_bytes: 2960010
    num_examples: 324
  - name: mind2web
    num_bytes: 159590
    num_examples: 122
  download_size: 1255385
  dataset_size: 8042511
language:
- en
pretty_name: AgentInstruct
---
# AgentInstruct Dataset

<p align="center">
  🤗 <a href="https://huggingface.co/THUDM/agentlm-70b" target="_blank">[Models]</a> • 💻 <a href="https://github.com/THUDM/AgentTuning" target="_blank">[Github Repo]</a> • 📌 <a href="https://THUDM.github.io/AgentTuning/" target="_blank">[Project Page]</a> • 📃 <a href="https://arxiv.org/abs/2310.12823" target="_blank">[Paper]</a> 
</p>

**AgentInstruct** is a meticulously curated dataset featuring **1,866** high-quality interactions, designed to enhance AI agents across six diverse real-world tasks, leveraging innovative methods like **Task Derivation** and **Self-Instruct**.

- 🔍 **CoT** - Harness the power of [ReAct](https://react-lm.github.io/), offering detailed thought explanations for each action, ensuring an intricate understanding of the model's decision-making journey.
- 🌍 **Diversity** - Spanning 6 real-world scenarios, from Daily Household Routines to Database Operations, and their average turns range from 5 to 35.
- 🎯 **Precision** - Not all trajectories of GPT-4 are effective! Ours are rigorously filtered using strict rewards to ensure top-notch quality.
- ✅ **Assurance** - Rigorous checks to avoid data leakage, ensuring pristine dataset quality.

## Task Overview

| Task | # Filt. Traj. | Avg # Filt. Traj. Turns |
|---|---|---|
|ALFWorld|336|13.52|
|WebShop|351|3.68|
|Mind2Web|122|1.00|
|Knowledge Graph|324|6.04|
|Operating System|195|3.85|
|Database|538|2.06|
|**AgentInstruct**|1866|5.24|

AgentInstruct includes 1,866 trajectories from
6 agents tasks. "Traj." stands for interaction trajectory. "Filt. Traj."
stands for filtered trajectories. 

## Models

**AgentLM** models are produced by mixed training on AgentInstruct dataset and ShareGPT dataset from Llama-2-chat models. 

The models follow the conversation format of [Llama-2-chat](https://huggingface.co/blog/llama2#how-to-prompt-llama-2), with system prompt fixed as 

```
You are a helpful, respectful and honest assistant.
```

7B, 13B, and 70B models are available on Huggingface model hub.

|Model|Huggingface Repo|
|---|---|
|AgentLM-7B| [🤗Huggingface Repo](https://huggingface.co/THUDM/agentlm-7b) |
|AgentLM-13B| [🤗Huggingface Repo](https://huggingface.co/THUDM/agentlm-13b) |
|AgentLM-70B| [🤗Huggingface Repo](https://huggingface.co/THUDM/agentlm-70b) |

Check our [[Github Repo]](https://github.com/THUDM/AgentTuning) for details about **AgentTuning**.

## Citation

If you find our work useful, please consider citing AgentTuning:

```
@misc{zeng2023agenttuning,
      title={AgentTuning: Enabling Generalized Agent Abilities for LLMs}, 
      author={Aohan Zeng and Mingdao Liu and Rui Lu and Bowen Wang and Xiao Liu and Yuxiao Dong and Jie Tang},
      year={2023},
      eprint={2310.12823},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```