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
data = pd.read_json('../data/my_test.jsonl', lines=True)
spark = ChatSparkLLM(
    spark_api_url="wss://spark-api-n.xf-yun.com/v3.1/chat",#spark pro微调的url
    spark_app_id="403f6768",
    spark_api_key="bb55d4b872e8a0fd5ab8dab00c676598",
    spark_api_secret="ZjEyNTIwZWFkYWQ5M2U4MjdmOTk2MzMx",
    spark_llm_domain="patchv3", #spark pro微调的版本
    streaming=False,
)
def save_result(data):
    with open("./data/myresult1.json", "w",encoding='utf-8') as f:
        file = data.to_json(orient='records', index=False, force_ascii=False)
        f.write(file)
    f.close()
multi_res=[]
for j in range(0, 6):
    res = []
    for i in tqdm(range(len(data)), desc=f"正在询问第{j}轮"):
        messages = [ChatMessage(
            role="user",
            content=data.iloc[i]["input"]
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
    data[f"infos_{j}"] = res
    save_result(data)