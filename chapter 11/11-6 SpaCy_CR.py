import spacy
import neuralcoref

# 加载Spacy模型
nlp = spacy.load("en_core_web_sm")

# 添加Neuralcoref扩展
neuralcoref.add_to_pipe(nlp)

# 处理文本
doc = nlp("John saw Mary. He waved to her.")

# 输出共指消解结果
for cluster in doc._.coref_clusters:
    print(cluster.mentions)
