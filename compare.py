import json


# 读取JSON文件内容
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


# 递归比较两个字典，并记录不同的键和值
def compare_dicts(dict1, dict2, index, differences):
    for key in dict1:
        if key in dict2:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                compare_dicts(dict1[key], dict2[key], f"{index}.{key}", differences)
            elif dict1[key] != dict2[key]:
                differences.append((index, key, dict1[key], dict2[key]))


# 比较两个JSON文件
def compare_json_files(file1_path, file2_path):
    data1 = read_json(file1_path)
    data2 = read_json(file2_path)

    differences = []
    for index, (item1, item2) in enumerate(zip(data1, data2), start=1):
        compare_dicts(item1, item2, index, differences)

    return differences


# 指定JSON文件路径
json_file_path1 = './compare/check_review1.json'
json_file_path2 = './compare/check_review2.json'

# 执行比较
differences = compare_json_files(json_file_path1, json_file_path2)

# 打印结果
if differences:
    for diff in differences:
        print(f"索引 {diff[0]} 的 '{diff[1]}' 字段不同：({diff[2]} vs {diff[3]})")
else:
    print("两个JSON文件完全相同。")


