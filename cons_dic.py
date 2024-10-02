def load_dict(file_name, score):
    word_dict = {}
    with open(file_name, encoding='utf-8') as fin:
        for line in fin:
            word = line.strip()
            word_dict[word] = score
    return word_dict


def append_dict(word_dict, file_name, score):
    with open(file_name, encoding='utf-8') as fin:
        for line in fin:
            word = line.strip()
            word_dict[word] = score


def load_extent_dict(file_name):
    extent_dict = {}
    with open(file_name, encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            line = line.split(',')
            extent_dict[line[0]] = line[1]
        print(extent_dict)
    return extent_dict

post_dict = load_dict(u"dict/正面情绪词.txt", 1)  # 积极情感词典
neg_dict = load_dict(u"dict/负面情绪词.txt", -1)  # 消极情感词典
inverse_dict = load_dict(u"dict/否定词.txt", -1)  # 否定词词典
extent_dict = load_extent_dict(u"dict/程度副词.txt")

import json

# 保存到JSON文件
with open('dict/post_dict.json', 'w') as f:
    json.dump(post_dict, f)

with open('dict/neg_dict.json', 'w') as f:
    json.dump(neg_dict, f)

with open('dict/inverse_dict.json', 'w') as f:
    json.dump(inverse_dict, f)

with open('dict/extent_dict.json', 'w') as f:
    json.dump(extent_dict, f)
