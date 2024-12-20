import json
def read_json(json_file_path):
    """读取json文件"""
    with open(json_file_path, 'r',encoding="utf-8") as f:
        data = json.load(f)
    return data
resultdata=read_json("./data/check_result4.json")
num=0

# 创建一个空列表，用于存储转换后的 JSON 对象
converted_infos = []

# 遍历原始的 infos_0 列表
for index, original_infos in enumerate(resultdata):
    # 创建新的 JSON 对象
    new_infos = {
        "infos": [],
        "index": index + 1  # 索引从1开始
    }
    new_infos['infos']=original_infos['final_infos_0']
    # 将新对象添加到转换后的列表中
    converted_infos.append(new_infos)

# 打印或使用转换后的数据
print(converted_infos)
# 定义一个函数来保存数据到 JSON 文件
def save_json(data, file_path):
    """将数据保存为 JSON 文件"""
    with open(file_path, 'w', encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 保存 converted_infos 到 JSON 文件
save_json(converted_infos, "compare/check_review2.json")