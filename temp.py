from transformers import BertTokenizer, BertForTokenClassification
from transformers import pipeline

# 加载分词器和模型
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertForTokenClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=13  # 根据实际需要的实体类型数量调整
)

# 创建NER pipeline
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

# 示例文本
text = "李雷住在北京市海淀区中关村南大街5号。他的电子邮件是lilei@example.com。"

# 应用NER
ner_results = nlp(text)

# 打印识别的实体
for ent in ner_results:
    print(f"{ent['word']} - {ent['entity']}")
