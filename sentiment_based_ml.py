import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json


class SentimentPipeline:
    def __init__(self, model_path, label_file_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()  # 设置模型为评估模式
        self.id2label = self.load_labels(label_file_path)

    def load_labels(self, file_path):
        '''加载标签映射'''
        id2label = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                label_data = json.loads(line.strip())
                id2label[int(label_data['label'])] = label_data['label_desc']
        return id2label

    def __call__(self, texts):
        # 确保输入是列表
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize 输入文本
        encoded_inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

        # 模型预测
        with torch.no_grad():
            outputs = self.model(**encoded_inputs)
        # 获取预测结果
        predictions = torch.argmax(outputs.logits, dim=1)

        # 解析标签
        results = []
        for text, pred in zip(texts, predictions):
            label = self.id2label[pred.item()]
            results.append({"text": text, "label": label})

        return results


# 使用
if __name__ == "__main__":
    model_path = "best_sentiment_classification/"
    label_file_path = "./data/labels.json"  # 替换为实际的标签文件路径
    pipeline = SentimentPipeline(model_path, label_file_path)

    texts = [
        "宿舍要民汉合宿了为毛都大三了还要折腾我",
        "早上的我，竟然也变成了一个无理取闹的人……",
        "多年来，周天先生率智多星的律师、策划师团队走出巴蜀，挺进海西，在厦门特区的发祥地安营扎寨。",
        "听故事的时候，你总喜欢眼巴巴的问，后来呢，可是当后来，自己成为讲故事的人，才发现，后来故事和话语就在嘴边，后来眼泪几度泪凝于睫，是真的说不下去了。。",
        "汉代前人口不多，而晋末、宋末、唐末及中国历史上三次移民潮，给皖南徽州送来了大量人口，人口众多，山多地少，怎么办？"
    ]

    results = pipeline(texts)

    for result in results:
        print(f"Text: {result['text'][:100]}...")  # 只打印前100个字符
        print(f"Predicted label: {result['label']}")
        print("-" * 50)