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
data_dir = "./LLM_with_dataprocess/data"
train_file = "myresult1.json"
output_data = pd.read_json(os.path.join(data_dir, train_file))
def check_prompt(x):
	# 提示词，我们交代清楚大模型的角色、目标、注意事项，然后提供背景信息，输出格式就可以了
    prompt = f"""Instruction:
你是一个信息要素提取工作的信息核查人员，你需要核验给定的`ChatText`中提取出**客户**的`Infos`中相关信息，核验'Infos'中的信息是否正确，
下面是对话相关信息ChatText:
{x["input"]}
如下是你需要核验的Infos信息:
{x['infos_3']}
对于错误的信息，你需要进行更正，并保证Infos的数据格式不变
注意事项：
1. 核验一定要严格,逐条进行核验检查！！！
2. 保持`Infos`的JSON格式不变，没有的信息项也要保留！！！
3. 对于有错误的数据项,进行更正并严格保证数据格式不发生改变!!!
"""
	# 要求的输出格式
    infos = """"要求输出的数据格式仍然按如下格式
Infos:
infos": [{
    "基本信息-姓名": "",
    "基本信息-手机号码": "",
    "基本信息-邮箱": "",
    "基本信息-地区": "",
    "基本信息-详细地址": "",
    "基本信息-性别": "",
    "基本信息-年龄": "",
    "基本信息-生日": "",
    "咨询类型": [],
    "意向产品": [],
    "购买异议点": [],
    "客户预算-预算是否充足": "",
    "客户预算-总体预算金额": "",
    "客户预算-预算明细": "",
    "竞品信息": "",
    "客户是否有意向": "",
    "客户是否有卡点": "",
    "客户购买阶段": "",
    "下一步跟进计划-参与人": [],
    "下一步跟进计划-时间点": "",
    "下一步跟进计划-具体事项": ""
}]
"""
    total=prompt+infos
    return total
output_data['check'] = output_data.apply(check_prompt, axis=1)
spark = ChatSparkLLM(
    spark_api_url="",#spark pro微调的url
    spark_app_id="",
    spark_api_key="",
    spark_api_secret="",
    spark_llm_domain="patchv3", #spark pro微调的版本
    streaming=False,
)
def save_result(data):
    with open("./data/check_result3.json", "w",encoding='utf-8') as f:
        file = data.to_json(orient='records', index=False, force_ascii=False)
        f.write(file)
    f.close()
multi_res=[]
for j in range(0, 2):
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
                a = json.loads(a.generations[0][0].text.replace("'", "\""))
            except:
                print("出错了")
                continue
            res.append(a)
            break
    multi_res.append(res)
    output_data[f"final_infos_{j}"] = res
    save_result(output_data)