from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage
import json


#星火认知大模型Spark3.5 Max的URL值，其他版本大模型URL值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
SPARKAI_URL = ''
#星火认知大模型调用秘钥信息，请前往讯飞开放平台控制台（https://console.xfyun.cn/services/bm35）查看
SPARKAI_APP_ID = ''
SPARKAI_API_SECRET = ''
SPARKAI_API_KEY = ''
#星火认知大模型Spark3.5 Max的domain值，其他版本大模型domain值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
SPARKAI_DOMAIN = 'generalv3.5'

def get_completions(text):
    messages = [ChatMessage(
        role="user",
        content=text
    )]
    spark = ChatSparkLLM(
        spark_api_url=SPARKAI_URL,
        spark_app_id=SPARKAI_APP_ID,
        spark_api_key=SPARKAI_API_KEY,
        spark_api_secret=SPARKAI_API_SECRET,
        spark_llm_domain=SPARKAI_DOMAIN,
        streaming=False,
    )
    handler = ChunkPrintHandler()
    a = spark.generate([messages], callbacks=[handler])
    return a.generations[0][0].text

# 测试模型配置是否正确
text = "你好"
get_completions(text)

def read_json(json_file_path):
    """读取json文件"""
    with open(json_file_path, 'r',encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_json(json_file_path, data):
    """写入json文件"""
    with open(json_file_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 读取数据

train_data = read_json("dataset/train.json")
test_data = read_json("dataset/test_data.json")

# prompt 设计
PROMPT_EXTRACT = """
你将获得一段群聊对话记录。你的任务是根据给定的表单格式从对话记录中提取结构化信息。在提取信息时，请确保它与类型信息完全匹配，不要添加任何没有出现在下面模式中的属性。

表单格式如下：
info: Array<Dict(
    "基本信息-姓名": string | "",  // 客户的姓名。
    "基本信息-手机号码": string | "",  // 客户的手机号码。
    "基本信息-邮箱": string | "",  // 客户的电子邮箱地址。
    "基本信息-地区": string | "",  // 客户所在的地区或城市。
    "基本信息-详细地址": string | "",  // 客户的详细地址。
    "基本信息-性别": string | "",  // 客户的性别。
    "基本信息-年龄": string | "",  // 客户的年龄。
    "基本信息-生日": string | "",  // 客户的生日。
    "咨询类型": string[] | [],  // 客户的咨询类型，如询价、答疑等。
    "意向产品": string[] | [],  // 客户感兴趣的产品。
    "购买异议点": string[] | [],  // 客户在购买过程中提出的异议或问题。
    "客户预算-预算是否充足": string | "",  // 客户的预算是否充足。示例：充足, 不充足
    "客户预算-总体预算金额": string | "",  // 客户的总体预算金额。
    "客户预算-预算明细": string | "",  // 客户预算的具体明细。
    "竞品信息": string | "",  // 竞争对手的信息。
    "客户是否有意向": string | "",  // 客户是否有购买意向。示例：有意向, 无意向
    "客户是否有卡点": string | "",  // 客户在购买过程中是否遇到阻碍或卡点。示例：有卡点, 无卡点
    "客户购买阶段": string | "",  // 客户当前的购买阶段，如合同中、方案交流等。
    "下一步跟进计划-参与人": string[] | [],  // 下一步跟进计划中涉及的人员（客服人员）。
    "下一步跟进计划-时间点": string | "",  // 下一步跟进的时间点。
    "下一步跟进计划-具体事项": string | ""  // 下一步需要进行的具体事项。
)>

请分析以下群聊对话记录，并根据上述格式提取信息：

**对话记录：**
```
{content}
```

请将提取的信息以JSON格式输出。
不要添加任何澄清信息。
输出必须遵循上面的模式。
不要添加任何没有出现在模式中的附加字段。
不要随意删除字段。

**输出：**
```
[{{
    "基本信息-姓名": "姓名",
    "基本信息-手机号码": "手机号码",
    "基本信息-邮箱": "邮箱",
    "基本信息-地区": "地区",
    "基本信息-详细地址": "详细地址",
    "基本信息-性别": "性别",
    "基本信息-年龄": "年龄",
    "基本信息-生日": "生日",
    "咨询类型": ["咨询类型"],
    "意向产品": ["意向产品"],
    "购买异议点": ["购买异议点"],
    "客户预算-预算是否充足": "充足或不充足",
    "客户预算-总体预算金额": "总体预算金额",
    "客户预算-预算明细": "预算明细",
    "竞品信息": "竞品信息",
    "客户是否有意向": "有意向或无意向",
    "客户是否有卡点": "有卡点或无卡点",
    "客户购买阶段": "购买阶段",
    "下一步跟进计划-参与人": ["跟进计划参与人"],
    "下一步跟进计划-时间点": "跟进计划时间点",
    "下一步跟进计划-具体事项": "跟进计划具体事项"
}}, ...]
```
"""

import json

class JsonFormatError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def convert_all_json_in_text_to_dict(text):
    """提取LLM输出文本中的json字符串"""
    dicts, stack = [], []
    for i in range(len(text)):
        if text[i] == '{':
            stack.append(i)
        elif text[i] == '}':
            begin = stack.pop()
            if not stack:
                dicts.append(json.loads(text[begin:i+1]))
    return dicts

# 查看对话标签
def print_json_format(data):
    """格式化输出json格式"""
    print(json.dumps(data, indent=4, ensure_ascii=False))



def check_and_complete_json_format(data):
    required_keys = {
        "基本信息-姓名": str,
        "基本信息-手机号码": str,
        "基本信息-邮箱": str,
        "基本信息-地区": str,
        "基本信息-详细地址": str,
        "基本信息-性别": str,
        "基本信息-年龄": str,
        "基本信息-生日": str,
        "咨询类型": list,
        "意向产品": list,
        "购买异议点": list,
        "客户预算-预算是否充足": str,
        "客户预算-总体预算金额": str,
        "客户预算-预算明细": str,
        "竞品信息": str,
        "客户是否有意向": str,
        "客户是否有卡点": str,
        "客户购买阶段": str,
        "下一步跟进计划-参与人": list,
        "下一步跟进计划-时间点": str,
        "下一步跟进计划-具体事项": str
    }

    if not isinstance(data, list):
        raise JsonFormatError("Data is not a list")

    for item in data:
        if not isinstance(item, dict):
            raise JsonFormatError("Item is not a dictionary")
        for key, value_type in required_keys.items():
            if key not in item:
                item[key] = [] if value_type == list else ""
            if not isinstance(item[key], value_type):
                raise JsonFormatError(f"Key '{key}' is not of type {value_type.__name__}")
            if value_type == list and not all(isinstance(i, str) for i in item[key]):
                raise JsonFormatError(f"Key '{key}' does not contain all strings in the list")

    return data
from tqdm import tqdm

retry_count = 5 # 重试次数
result = []
error_data = []

for index, data in tqdm(enumerate(test_data)):
    index += 1
    is_success = False
    for i in range(retry_count):
        try:
            res = get_completions(PROMPT_EXTRACT.format(content=data["chat_text"]))
            infos = convert_all_json_in_text_to_dict(res)
            infos = check_and_complete_json_format(infos)
            result.append({
                "infos": infos,
                "index": index
            })
            is_success = True
            break
        except Exception as e:
            print("index:", index, ", error:", e)
            continue
    if not is_success:
        data["index"] = index
        error_data.append(data)

write_json("output.json", result)