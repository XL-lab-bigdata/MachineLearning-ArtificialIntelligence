import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag, ne_chunk
import os

# 设置代理（必要时调整）
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'https://127.0.0.1:7890'

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# 示例文本
text = (
    "Alice loves programming , She often spends her weekends writing code and learning new technologies. "
    "Recently, she started a project on machine learning. Bob, her friend, is also interested in this field. "
    "He joined her in the project. They have been working together for weeks now."
)


# 步骤 1：提及提取
def get_mentions(text):
    sentences = sent_tokenize(text)
    mentions = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        tagged = pos_tag(tokens)
        chunks = ne_chunk(tagged)
        mention_list = []
        for chunk in chunks:
            if hasattr(chunk, 'label'):
                mention_list.append(chunk)
        mentions.append(mention_list)
    return mentions


# 步骤 2：寻找前置词
def find_antecedent(mentions, pronoun):
    # 反向查找提及
    for i in range(len(mentions) - 1, -1, -1):
        for chunk in mentions[i]:
            name = ' '.join([leaf[0] for leaf in chunk.leaves()])
            if pronoun.lower() in name.lower():
                return chunk
    return None

# 步骤 3：应用启发式规则
def apply_rules(pronoun, candidates):
    if not candidates:
        return None

    # 选择最接近代词的先行词
    best_antecedent = candidates[0][0]
    best_distance = abs(candidates[0][1] - mentions.index(mentions))
    for candidate, distance in candidates[1:]:
        current_distance = abs(distance - mentions.index(mentions))
        if current_distance < best_distance:
            best_distance = current_distance
            best_antecedent = candidate

    return best_antecedent

# 步骤 4：核心ference选择
def resolve_coreference(text, pronoun):
    global mentions  # 使用全局变量mentions
    mentions = get_mentions(text)
    print("Identified Mentions:", mentions)  # 调试输出
    candidates = find_antecedent(mentions, pronoun)

    antecedent = apply_rules(pronoun, candidates)
    if antecedent:
        return antecedent
    return None

# 使用示例
pronoun = "He"
resolved_antecedent = resolve_coreference(text, pronoun)
if resolved_antecedent:
    print("The antecedent of the pronoun is:", ' '.join([leaf[0] for leaf in resolved_antecedent.leaves()]))
else:
    print("Antecedent not found")
