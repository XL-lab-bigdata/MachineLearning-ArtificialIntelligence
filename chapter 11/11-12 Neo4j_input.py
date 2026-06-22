import csv  # 导入csv文件
import py2neo  # 导入py2neo库
from py2neo import Node, Relationship, Graph, NodeMatcher, RelationshipMatcher#导入我们需要的头文件
test_graph = Graph('http://localhost:7474',auth=('用户名','密码'))# 连接neo4j 数据库 # 连接neo4j，将'xxx'分别改为你的用户名和密码
test_graph.delete_all()  # 清除neo4j中原有的结点等所有信息

with open('.../12-8 data.csv', 'r', encoding='utf_8_sig') as f: #由kg_df存储而成
    reader = csv.reader(f)
    for item in reader:
        # if reader.line_num==1:
        #    continue
        print("当前行数：", reader.line_num, "当前内容：", item)
        start_node = Node("source", name=item[0], chapter = 1)
        end_node = Node("target", name=item[1], chapter = 2)
        relation = Relationship(start_node, item[2], end_node)

        test_graph.merge(start_node, "source", "name")
        test_graph.merge(end_node, "target", "name")
        test_graph.merge(relation, "edge", "属性")
