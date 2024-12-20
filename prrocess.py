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

tqdm.pandas()
plt.rcParams['font.family'] = ['STFangsong']
plt.rcParams['axes.unicode_minus'] = False


data_dir = "./dataset"
train_file = "train.json"
test_file = "test_data.json"

train_data = pd.read_json(os.path.join(data_dir, train_file))

test_data =  pd.read_json(os.path.join(data_dir, test_file))

# 删除表情图片、超链接
train_data['chat_text'] = train_data['chat_text'].str.replace(r"\[[^\[\]]{2,10}\]", "", regex=True)
train_data['chat_text'] = train_data['chat_text'].str.replace("https?://\S+", "", regex=True)
test_data['chat_text'] = test_data['chat_text'].str.replace(r"\[[^\[\]]{2,10}\]", "", regex=True)
test_data['chat_text'] = test_data['chat_text'].str.replace("https?://\S+", "", regex=True)


def get_names_phones_and_emails(example):
    names = re.findall(r"(?:\n)?([\u4e00-\u9fa5]+\d+)：", example["chat_text"])
    names += re.findall(r"@([\u4e00-\u9fa5]+)\s", example["chat_text"])
    emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}", example["chat_text"])
    # phones = re.findall(r"1[356789]\d{9}", example["chat_text"]) # 文本中的手机号并不是标准手机号
    phones = re.findall(r"\d{3}\s*\d{4}\s*\d{4}", example["chat_text"])
    return pd.Series([set(names), set(phones), set(emails)], index=['names', 'phones', 'emails'])


def merge_chat(example):
    for name in example['names']:
        example["chat_text"] = example["chat_text"].replace(f"\n{name}：", f"<|sep|>{name}：")
    chats = example["chat_text"].split("<|sep|>")

    last_name = "UNKNOWN"
    new_chats = []
    for chat in chats:
        if chat.startswith(last_name):
            chat = chat.strip("\n")
            chat = "".join(chat.split("：")[1:])
            new_chats[-1] += " " + chat
        else:
            new_chats.append(chat)
            last_name = chat.split("：")[0]
    return pd.Series(["\n".join(new_chats), new_chats], index=["chats", "chat_list"])


# 使用正则表达式获得'names', 'phones', 'emails'
train_data[['names', 'phones', 'emails']] = train_data.apply(get_names_phones_and_emails, axis=1)
test_data[['names', 'phones', 'emails']] = test_data.apply(get_names_phones_and_emails, axis=1)
# 分割聊天记录， 合并连续相同人的聊天
train_data[["chats", "chat_list"]] = train_data.apply(merge_chat, axis=1)
test_data[["chats", "chat_list"]] = test_data.apply(merge_chat, axis=1)


def process(excemple):
    chat_list = excemple["chat_text"].split("\n")

    res = []
    s = 0
    while s < len(chat_list):

        i, j = s, s + 1
        start_j = j
        while i < len(chat_list) and j < len(chat_list):
            if chat_list[i] == chat_list[j]:
                i += 1
            else:
                if i != s:
                    if j - start_j > 10:
                        res += list(range(start_j, j))
                    i = s
                start_j = j
            j += 1
        s += 1
    texts = []
    for i in range(len(chat_list)):
        if i not in res:
            texts.append(chat_list[i])
    return "\n".join(texts)


train_data["chat_text"] = train_data.apply(process, axis=1)
print(len(train_data))
test_data["chat_text"] = test_data.apply(process, axis=1)

def process(x):
	# 提示词，我们交代清楚大模型的角色、目标、注意事项，然后提供背景信息，输出格式就可以了
    prompt = f"""Instruction:
你是一个信息要素提取工作人员助手，你需要从给定的`ChatText`中提取出**客户**的`Infos`中相关信息，将提取的信息填到`Infos`中，
########
注意事项：
1. 没有的信息无需填写
2. 保持`Infos`的JSON格式不变，没有的信息项也要保留！！！
4. 姓名可以是聊天昵称
5. 注意是客户的信息，不是客服的信息
6. 可以有多个客户信息
7. 确保信息正确无误,会有专门的核验人员去检查你的结果，不要偷懒否则会惩罚你！！！
#######
ChatText:
{x["chat_text"]}
"""
	# 要求的输出格式
    infos = """"
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
	# prompt+infos是文件中的input，answer是文件中的target
    answer = f"""{x["infos"]}""" #target
    total= len(prompt + infos + answer)
    if total > 8000:
        prompt = prompt[:8000-len(infos + answer)]
    return pd.Series([prompt, answer], index=["input", "target"])

data1 = train_data.apply(process, axis=1)
# 测试集中的target并没有用可以忽略
data2 = test_data.apply(process, axis=1)

# #保存数据
# with open(os.path.join(data_dir, "my_train.jsonl"), "w", encoding="utf-8") as f:
#     f.write("\n".join([json.dumps(i, ensure_ascii=False) for i in list(data1.transpose().to_dict().values())]))
# f.close()
# with open(os.path.join(data_dir, "my_test.jsonl"), "w", encoding="utf-8") as f:
#     f.write("\n".join([json.dumps(i, ensure_ascii=False) for i in list(data2.transpose().to_dict().values())]))
# f.close()

spark = ChatSparkLLM(
    spark_api_url="",#spark pro微调的url
    spark_app_id="",
    spark_api_key="",
    spark_api_secret="",
    spark_llm_domain="patchv3", #spark pro微调的版本
    streaming=False,
    temperature = 0.00001,
    max_tokens = 8192,
    request_timeout = 60,
)
def save_result(data):
    with open("./data/result4.json", "w",encoding='utf-8') as f:
        file = data.to_json(orient='records', index=False, force_ascii=False)
        f.write(file)
    f.close()
multi_res=[]
for j in range(0, 6):
    res = []
    for i in tqdm(range(len(data2)), desc=f"正在询问第{j}轮"):
        messages = [ChatMessage(
            role="user",
            content=data2.iloc[i]["input"]
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