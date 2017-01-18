# Link Prediction

任务：给定论文标题、作者、期刊、摘要、发表时间，预测两篇论文之间是否具有引用关系。

任务链接：https://inclass.kaggle.com/c/link-prediction-tu

## 1. 运行说明

1. 下载语料放在data/raw文件夹下，结构如下：
    data/raw/training_set.txt
    data/raw/testing_set.txt
    data/raw/node_information.csv
2. 运行脚本run.sh，耗时大概4小时，预测结果将输出到data/raw/output.txt

## 2. run.sh脚本处理流程说明

1. 自动下载LINE工具，用于生成低维稠密的网络节点向量。
2. 自动运行process_network_data.py，根据原始语料生成LINE训练过程中使用的语料。
    输入文件：data/raw/training_set.txt, data/raw/node_information.csv
    输出文件：
        data/tmp/node_network.txt
        data/tmp/author_ids.pkl
        data/tmp/author_network.txt
3. 自动拷贝utils/train_LINE.sh脚本到LINE工具的可执行文件同级目录中，并运行。生成两个结果文件：
    data/features/node_network.bin      论文网络节点向量
    data/features/author_network.bin    作者网络节点向量
4. 自动运行test.py，完成特征工程，模型训练及预测的所有过程。

## 3. 任务中使用的特征列表

- year_dis 发表年份的差值
- year_source  源论文发表年份
- year_target  被引论文发表年份
- common_author  论文共同作者数
- overlap_title  论文标题的共现词数
- overlap_journal  期刊的共现词数
- abstract_tfidf_similar  摘要tfidf的相似度
```
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(min_df=2)
abstract_tfidf=vectorizer.fit_transform(node_df.abstract)
cosine(abstract_tfidf[row.sindex].toarray()[0],abstract_tfidf[row.tindex].toarray()[0])
```
- abstract_svd_similar  摘要svd降维后的相似度
```
from sklearn.decomposition import TruncatedSVD
abstract_svd=TruncatedSVD(n_components=100,random_state=100).fit_transform(abstract_tfidf)
cosine(abstract_svd[row.sindex],abstract_svd[row.tindex])
```
- in_degree_target  被引论文的入度
- out_degree_source  源论文的出度
- g_jaccard_index  无向图中jaccard index值
- g_neighbour_sqrt  无向图中公共邻居的比例
- g_neighbour_pearson  无向图邻居的pearson coefficient
- g_cluster_source  无向图中源论文的聚类系数
- g_cluster_target  无向图中被引论文的聚类系数
```
import networkx as nx
g_cluster=nx.algorithms.cluster.clustering(G)
g_cluster.get(row.sid,0)
```
- g_kcore_source  无向图中源论文的kcore
- g_kcore_target  无向图中被引论文的kcore
```
g_kcore=nx.core_number(G)
g_kcore.get(row.sid,0)
```
- g_pagerank_source  无向图中源论文的pagerank值
- g_pagerank_target  无向图中被引论文的pagerank值
```
g_pagerank=nx.pagerank(G)
g_pagerank.get(row.sid,0)
```
- g_aver_neighbour_source  无向图中源论文的平均邻居数
- g_aver_neighbour_target  无向图中被引论文的平均邻居数
```
g_aver_neighbor=nx.average_neighbor_degree(G)
g_aver_neighbor.get(row.tid,0)
```
## 4. Network Embedding
## 5. Doc2vec
## 6. 分类器
在本任务中尝试了Logistic Regression、SVM、GBDT、XGBoost及LightGBM几种分类器，其中LightGBM表现最好。
```
gbm = lgb.LGBMClassifier(objective='binary',
                        num_leaves=31,
                        learning_rate=0.05,
                        n_estimators=1000,subsample=0.8,)
```

## 7. 结果

仅使用特征列表中的21维特征，线上B榜可达到0.97422的成绩。