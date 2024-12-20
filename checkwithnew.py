from dataclasses import dataclass
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage
import pandas as pd
import os
import json
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import ceil
import numpy as np
from copy import deepcopy
import random
data_dir = "./data/"
train_file = "result4.json"
output_data = pd.read_json(os.path.join(data_dir, train_file))
def check_prompt(x):
	# 提示词，我们交代清楚大模型的角色、目标、注意事项，然后提供背景信息，输出格式就可以了
    prompt = f"""Instruction:
你是一个信息要素提取工作的信息核查人员，你需要核验从给定的`ChatText`中提取出**客户**的`Infos`中相关信息，核验'Infos'中的信息是否正确，
下面是对话相关信息ChatText:
{x["chat_text"]}
########
如下是你需要核验的Infos信息:
{x['infos_5']}
对于你存在疑问的信息，在相应的Infos信息后面给出你的疑问点，如果信息存在错误，你需要给出错误依据。
注意事项：
1. 核验一定要严格,逐条进行核验检查！！！
2. 仅输出你的意见信息即可！！！
4. 一定要认真分析每一条对话，仔细核对每一个结果！！！！
5. 会有相关人员检查你的核验正确率，请认真对待！！！
"""
    total=prompt
    return total
output_data['check'] = output_data.apply(check_prompt, axis=1)
spark = ChatSparkLLM(
    spark_api_url="",#spark pro微调的url
    spark_app_id="",
    spark_api_key="",
    spark_api_secret="",
    spark_llm_domain="", #spark pro微调的版本
    streaming=False,
    temperature=0.0001,
    max_tokens=8192,
    request_timeout=60,
)
def save_result(data):
    with open("./data/check_result4.json", "w",encoding='utf-8') as f:
        file = data.to_json(orient='records', index=False, force_ascii=False)
        f.write(file)
    f.close()
multi_res=[]
for j in range(0, 1):
    res = []
    for i in tqdm(range(len(output_data)), desc=f"正在询问第{j}轮"):
        messages = [ChatMessage(
            role="user",
            content=output_data.iloc[i]["check"]
        )]
        while True:
            try:
                handler = ChunkPrintHandler()
                a = spark.generate([messages], callbacks=[handler])
               # a = json.loads(a.generations[0][0].text.replace("'", "\""))
            except:
                print("出错了")
                continue
            res.append(a)
            break
    multi_res.append(res)
    output_data[f"check_infos_{j}"] = res
    save_result(output_data)