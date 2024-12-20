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
def chatbot(prompt):
    #星火认知大模型Spark3.5 Max的URL值，其他版本大模型URL值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
    SPARKAI_URL = ''
    #星火认知大模型调用秘钥信息，请前往讯飞开放平台控制台（https://console.xfyun.cn/services/bm35）查看
    SPARKAI_APP_ID = ''
    SPARKAI_API_SECRET = ''
    SPARKAI_API_KEY = ''
    #星火认知大模型Spark3.5 Max的domain值，其他版本大模型domain值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
    SPARKAI_DOMAIN = 'generalv3.5'
    spark = ChatSparkLLM(
        spark_api_url=SPARKAI_URL,
        spark_app_id=SPARKAI_APP_ID,
        spark_api_key=SPARKAI_API_KEY,
        spark_api_secret=SPARKAI_API_SECRET,
        spark_llm_domain=SPARKAI_DOMAIN,
        streaming=False,
        temperature=0.05,
        max_tokens=8192,
        request_timeout=60

    )
    messages = [ChatMessage(
        role="user",
        content=prompt
    )]
    handler = ChunkPrintHandler()
    a = spark.generate([messages], callbacks=[handler])
    return a.generations[0][0].message.content

jsonl_data = {"instruction":"假设你是一个智能交互助手，基于用户的输入文本，解析其中语义，抽取关键信息，以json格式生成结构化的语义内容。","input":"请调小空气净化器的湿度到1","output":"{\"intent\":\"CONTROL\",\"slots\":[{\"name\":\"device\",\"normValue\":\"airCleaner\",\"value\":\"空气净化器\"},{\"name\":\"insType\",\"normValue\":\"set\",\"value\":\"调小\"},{\"name\":\"attr\",\"normValue\":\"humidity\",\"value\":\"湿度\"},{\"name\":\"attrValue\",\"normValue\":\"1\",\"value\":\"1\"}],\"sample\":\"请调小空气净化器的湿度到1\"}"}
#打开一个文件用于写入，如果文件已存在则会被覆盖
with open('./dataset/traindata.jsonl', 'w', encoding='utf-8') as file:
    # 训练集行数(130)不符合要求，范围：1500~90000000
    # 遍历数据列表，并将每一行写入文件
    # 这里为了满足微调需求我们重复12次数据集 130*12=1560

    for index in tqdm(train_data.index):
        line_input = train_data.loc[index, "chat_text"]
        line_output = train_data.loc[index, "infos"]
        content = line_input

        prompt = f'''
                你是一个数据分析大师，你需要从群聊对话中进行分析，里面对话的角色中大部分是客服角色，你需要从中区分出有需求的客户，并得到以下四类数据。

                ****群聊对话****
                {content}

                ****分析数据****
                客户基本信息：需要从中区分出客户角色，并得到客户基本信息，其中包括姓名、手机号码、邮箱、地区、详细地址、性别、年龄和生日
                客户意向与预算信息： 客户意向与预算信息包括咨询类型、意向产品、购买异议点、预算是否充足、总体预算金额以及预算明细
                客户购买准备情况：户购买准备情况包括竞品信息、客户是否有意向、客户是否有卡点以及客户购买阶段
                跟进计划信息： 跟进计划信息包括参与人、时间点和具体事项，这些信息用于指导销售团队在未来的跟进工作中与客户互动

                ****注意****
                1.只输出客户基本信息、客户意向与预算信息、客户购买准备情况、跟进计划信息对应的信息，不要输出无关内容
                2.不要输出分析内容
                3.输出内容格式为md格式
                '''
        res = chatbot(prompt=prompt)
        # print(res)
        line_write = {
            "instruction": jsonl_data["instruction"],
            "input": json.dumps(res, ensure_ascii=False),
            "output": json.dumps(line_output, ensure_ascii=False)
        }
        # 因为数据共有130行，为了能满足训练需要的1500条及以上，我们将正常训练数据扩充12倍。
        file.write(json.dumps(line_write, ensure_ascii=False) + '\n')  # '\n' 用于在每行末尾添加换行符
# with open('./dataset/testdata.jsonl', 'w', encoding='utf-8') as file:
#     # 训练集行数(130)不符合要求，范围：1500~90000000
#     # 遍历数据列表，并将每一行写入文件
#     # 这里为了满足微调需求我们重复12次数据集 130*12=1560
#
#     for index in tqdm(test_data.index):
#         line_input = test_data.loc[index, "chat_text"]
#         line_output = test_data.loc[index, "infos"]
#         content = line_input
#
#         prompt = f'''
#                 你是一个数据分析大师，你需要从群聊对话中进行分析，里面对话的角色中大部分是客服角色，你需要从中区分出有需求的客户，并得到以下四类数据。
#
#                 ****群聊对话****
#                 {content}
#
#                 ****分析数据****
#                 客户基本信息：需要从中区分出客户角色，并得到客户基本信息，其中包括姓名、手机号码、邮箱、地区、详细地址、性别、年龄和生日
#                 客户意向与预算信息： 客户意向与预算信息包括咨询类型、意向产品、购买异议点、预算是否充足、总体预算金额以及预算明细
#                 客户购买准备情况：户购买准备情况包括竞品信息、客户是否有意向、客户是否有卡点以及客户购买阶段
#                 跟进计划信息： 跟进计划信息包括参与人、时间点和具体事项，这些信息用于指导销售团队在未来的跟进工作中与客户互动
#
#                 ****注意****
#                 1.只输出客户基本信息、客户意向与预算信息、客户购买准备情况、跟进计划信息对应的信息，不要输出无关内容
#                 2.不要输出分析内容
#                 3.输出内容格式为md格式
#                 '''
#         res = chatbot(prompt=prompt)
#         # print(res)
#         line_write = {
#             "instruction": jsonl_data["instruction"],
#             "input": json.dumps(res, ensure_ascii=False),
#             "output": json.dumps(line_output, ensure_ascii=False)
#         }
#         # 因为数据共有130行，为了能满足训练需要的1500条及以上，我们将正常训练数据扩充12倍。
#         file.write(json.dumps(line_write, ensure_ascii=False) + '\n')  # '\n' 用于在每行末尾添加换行符