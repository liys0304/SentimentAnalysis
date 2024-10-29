import thulac
import json
import tensorflow as tf
import numpy as np

class SentimentByDic:
    def __init__(self):
        self.model = tf.keras.models.load_model('classify.keras')
        self.split_model = thulac.thulac(seg_only=True, filt = True)  #默认模式

        with open('dict/extent_dict.json', 'r') as f:
            self.extent_dict = json.load(f)

        with open('dict/inverse_dict.json', 'r') as f:
            self.inverse_dict = json.load(f)

        with open('dict/neg_dict.json', 'r') as f:
            self.neg_dict = json.load(f)

        with open('dict/post_dict.json', 'r') as f:
            self.post_dict = json.load(f)

        with open('dict/punc.json', 'r', encoding="utf-8") as f:
            self.punc = json.load(f)

    def text_split(self, text):
        # thu1 = thulac.thulac(seg_only=True, filt=True)  # 默认模式
        text = self.split_model.cut(text, text=True)  # 进行一句话分词
        words = text.split(' ')
        return words

    def compute_score(self, text, like_count=0):
        total_score = 0
        last_word_pos = 0
        last_pun_pos = 0
        index = 0
        words = self.text_split(text)
        pos_word = 0
        neg_word = 0
        for word in words:
            # 分句
            if word in self.punc:
                last_pun_pos = index
            # 积极词处理规则
            # 与消极词相同，前面遇到否定词取相反数，遇到程度副词乘以系数。
            if word in self.post_dict:
                pos_word += 1
                if last_word_pos > last_pun_pos:
                    start = last_word_pos
                else:
                    start = last_pun_pos
                score = 1
                for word_before in words[start: index]:
                    if word_before in self.extent_dict:
                        score *= float(self.extent_dict[word_before])
                    if word_before in self.inverse_dict:
                        score *= -1
                last_word_pos = index
                total_score += score
            elif word in self.neg_dict:
                neg_word += 1
                if last_word_pos > last_pun_pos:
                    start = last_word_pos
                else:
                    start = last_pun_pos
                score = -1
                for word_before in words[start: index]:
                    if word_before in self.extent_dict:
                        score *= float(self.extent_dict[word_before])
                    if word_before in self.inverse_dict:
                        score *= -1
                last_word_pos = index
                total_score += score
            index += 1
        if total_score > 0:
            total_score += like_count * 0.1
        else:
            total_score -= like_count * 0.1
        if len(words):
            pos_freq = pos_word / len(words)
            neg_freq = neg_word / len(words)
        else:
            pos_freq = 0
            neg_freq = 0
        return total_score, pos_freq, neg_freq, len(words)

    def __call__(self, text, like_count=0):
        score, pos_freq, neg_freq, text_len = self.compute_score(text, like_count)
        input_features = np.array([score, pos_freq, neg_freq, text_len]).reshape(1, -1)
        probabilities = self.model.predict(input_features)
        predicted_class = np.argmax(probabilities, axis=1)
        if predicted_class == [0]:
            return "negative"
        elif predicted_class == [1]:
            return "neutral"
        else:
            return "positive"
