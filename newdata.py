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
print(train_data)
