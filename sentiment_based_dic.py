import thulac
import json

thu1 = thulac.thulac(seg_only=True, filt = True)  #默认模式
# text = thu1.cut("中华民族创造了悠久灿烂的中华文明，为人类作出了卓越贡献。", text=True)  #进行一句话分词
# words = text.split(' ')
# print(words)

with open('dict/extent_dict.json', 'r') as f:
    extent_dict = json.load(f)

with open('dict/inverse_dict.json', 'r') as f:
    inverse_dict = json.load(f)

with open('dict/neg_dict.json', 'r') as f:
    neg_dict = json.load(f)

with open('dict/post_dict.json', 'r') as f:
    post_dict = json.load(f)

with open('dict/punc.json', 'r', encoding="utf-8") as f:
    punc = json.load(f)

def text_split(text):
    # thu1 = thulac.thulac(seg_only=True, filt=True)  # 默认模式
    text = thu1.cut(text, text=True)  # 进行一句话分词
    words = text.split(' ')
    return words

def compute_score(text, like_count):
    total_score = 0
    last_word_pos = 0
    last_pun_pos = 0
    index = 0
    words = text_split(text)

    for word in words:
        # 分句
        if word in punc:
            last_pun_pos = index
        # 积极词处理规则
        # 与消极词相同，前面遇到否定词取相反数，遇到程度副词乘以系数。
        if word in post_dict:
            if last_word_pos > last_pun_pos:
                start = last_word_pos
            else:
                start = last_pun_pos
            score = 1
            for word_before in words[start: index]:
                if word_before in extent_dict:
                    score *= extent_dict[word_before]
                if word_before in inverse_dict:
                    score *= -1
            last_word_pos = index
            total_score += score
        elif word in neg_dict:
            if last_word_pos > last_pun_pos:
                start = last_word_pos
            else:
                start = last_pun_pos
            score = -1
            for word_before in words[start: index]:
                if word_before in extent_dict:
                    score *= extent_dict[word_before]
                if word_before in inverse_dict:
                    score *= -1
            last_word_pos = index
            total_score += score
        index += 1
    if total_score > 0: return total_score + 0.01 * like_count
    else: return  total_score - 0.01 * like_count

