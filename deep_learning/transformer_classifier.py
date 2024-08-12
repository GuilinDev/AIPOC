from transformers import pipeline

# 使用预训练的文本分类模型
classifier = pipeline('sentiment-analysis')

# 输入文本进行分类
result = classifier("I love deep learning!")
print(result)  # 输出可能是： [{'label': 'POSITIVE', 'score': 0.9998}]