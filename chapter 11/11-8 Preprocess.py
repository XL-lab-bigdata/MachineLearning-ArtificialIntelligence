
# 分句函数
def cut_sent(infile, outfile):
    cutLineFlag = ["？", "！", "。", "…", "."]
    sentenceList = []

    # 打开输入文件，读取每行文本
    with open(infile, "r", encoding="UTF-8") as file:
        oneSentence = ""

        for line in file:
            # 替换掉全角和半角空格
            line = line.replace(u'\x20', ' ').replace(u'\u3000', ' ')

            # 去掉行首尾的空格并检查是否为空行
            words = line.strip()
            if not words:  # 如果当前行为空，跳过
                continue

            for word in words:
                if word not in cutLineFlag:  # 如果字符不是分句符号
                    oneSentence += word
                else:  # 如果字符是分句符号
                    oneSentence += word
                    if len(oneSentence) > 4:  # 如果句子长度超过4个字符，则保存
                        sentenceList.append(oneSentence.strip() + "\r")
                    oneSentence = ""  # 句子处理完成，重置

        # 处理最后一行可能没有分句符号结尾的情况
        if oneSentence:
            sentenceList.append(oneSentence.strip() + "\r")

    # 将结果写入输出文件
    with open(outfile, "w", encoding="UTF-8") as resultFile:
        print(f"共分割出 {len(sentenceList)} 句子")
        resultFile.writelines(sentenceList)

