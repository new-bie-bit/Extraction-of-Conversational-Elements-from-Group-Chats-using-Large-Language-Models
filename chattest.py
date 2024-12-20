from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage
import pandas as pd
import os
from tqdm import tqdm
import json


spark = ChatSparkLLM(
    spark_api_url="",#spark pro微调的url
    spark_app_id="",
    spark_api_key="",
    spark_api_secret="",
    spark_llm_domain="patchv3", #spark pro微调的版本
    streaming=False,
)
def save_result(data):
    with open("./data/result1.json", "w") as f:
        file = data.to_json(orient='records', index=False, force_ascii=False)
        f.write(file)
    f.close()
for j in range(0, 10):
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
    test_data[f"infos_{j}"] = res
    save_result(test_data)
