'''
论文引用关系预测
'''
import pandas as pd
import numpy as np
import sys,os,pickle,nltk
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.spatial import distance
from scipy.stats import pearsonr
import math
import networkx as nx
import lightgbm as lgb
from sklearn.metrics import log_loss,accuracy_score,f1_score
sys.path.append(os.path.abspath('../src'))

'''配置项
'''
folder_name='data/'
train_file='data/training_set.txt'
test_file='data/testing_set.txt'
output_file='data/output.txt'

'''全局变量
'''
train_df=pd.read_csv(train_file,sep=' ',names=['sid','tid','label'])
test_df=pd.read_csv(test_file,sep=' ',names=['sid','tid','label'])
node_df=pd.read_csv('data/node_information.csv',names=['id','year','title','authors','journal','abstract'])
features={}

'''工具函数
'''
def cosine(a,b):
    res=0
    if np.linalg.norm(a)!=0 and np.linalg.norm(b)!=0:
        res=distance.cosine(a,b)
    return res

def get_vectors(a_list,b_list):
    '''将两个list转换为两个one-hot向量
    '''
    id2index=dict([(id,i) for i,id in enumerate(set(a_list+b_list))])
    a=np.zeros((len(id2index),))
    b=np.zeros((len(id2index),))
    for key in a_list:
        a[id2index[key]]=1
    for key in b_list:
        b[id2index[key]]=1
    return a,b

def prepare_node_df():
    '''对node_df进行预置处理
    '''
    nltk.download('punkt') # for tokenization
    nltk.download('stopwords')
    stpwds = set(nltk.corpus.stopwords.words("english"))
    stemmer = nltk.stem.PorterStemmer()
    
    #分割作者字段
    node_df.authors.fillna('',inplace=True)
    node_df.loc[:,'author_set']=node_df.authors.apply(lambda x:[x.strip() for x in x.lower().split(',') if len(x.strip())>0])
    #分割文章标题
    node_df.title.fillna('',inplace=True)
    node_df.loc[:,'title_set']=node_df.title.apply(lambda x:set([stemmer.stem(w) for w in x.lower().split(' ') if w not in stpwds]))
    #处理摘要
    node_df.abstract.fillna('',inplace=True)
    node_df.loc[:,'simple_abstract']=node_df.abstract.apply(lambda x:' '.join([w.strip() for w in x.lower().split(' ') if w not in stpwds]))
    #处理journal
    node_df.journal.fillna('',inplace=True)
    node_df.loc[:,'journal_set']=node_df.journal.apply(lambda x:set([w.strip() for w in x.split('.') if len(x.strip())>0]))
    
def append_node_index(df):
    '''向df中添加source index及target index
    依赖全局变量：node_df
    '''
    id2index=dict([(id,i) for i,id in enumerate(node_df.id)])
    df.loc[:,'sindex']=df.sid.apply(lambda x:id2index[x])
    df.loc[:,'tindex']=df.tid.apply(lambda x:id2index[x])
    
def set_feature(name,val):
    '''设置特征值
    '''
    if name not in features:
        features[name]=[]
    features[name].append(val)
    
def get_features(df):
    '''特征工程
    '''
    features.clear()
    for i,row in enumerate(df.itertuples()):
        source=node_df.loc[row.sindex]
        target=node_df.loc[row.tindex]

        # 年份特征
        set_feature('year_dis',source.year-target.year)
        set_feature('year_source',source.year)
        set_feature('year_target',target.year)
        # 共同作者数量
        set_feature('common_author',len(set(source.author_set).intersection(set(target.author_set))))
        # 标题中共现词数
        set_feature('overlap_title',len(source.title_set.intersection(target.title_set)))
        # 期刊中共现词数
        set_feature('overlap_journal',len(source.journal_set.intersection(target.journal_set)))
        #摘要的cosine相似度
        set_feature('abstract_tfidf_similar',cosine(abstract_tfidf[row.sindex].toarray()[0],abstract_tfidf[row.tindex].toarray()[0]))
        set_feature('abstract_svd_similar',cosine(abstract_svd[row.sindex],abstract_svd[row.tindex]))
        # 节点的度
        set_feature('in_degree_target',in_degree.get(row.tid,0))
        set_feature('out_degree_target',out_degree.get(row.sid,0))
        # jaccard index
        sn=G.neighbors(row.sid)
        tn=G.neighbors(row.tid)
        common_ns=len(set(sn).intersection(set(tn)))
        set_feature('g_jaccard_index',common_ns/(1e-6+len(set(sn+tn))))
        sn_vec,tn_vec=get_vectors(sn,tn)
        set_feature('g_neighbour_sqrt',common_ns/(math.sqrt(len(sn)+len(tn))+1e-6))
        set_feature('g_neighbour_cosine',distance.cosine(sn_vec,tn_vec))
        set_feature('g_neighbour_pearson',pearsonr(sn_vec,tn_vec)[0])
        set_feature('g_cluster_source',g_cluster.get(row.sid,0))
        set_feature('g_cluster_target',g_cluster.get(row.tid,0))
        set_feature('g_kcore_source',g_kcore.get(row.sid,0))
        set_feature('g_kcore_target',g_kcore.get(row.tid,0))
        set_feature('g_pagerank_source',g_pagerank.get(row.sid,0))
        set_feature('g_pagerank_target',g_pagerank.get(row.tid,0))
        set_feature('g_aver_neighbour_source',g_aver_neighbor.get(row.sid,0))
        set_feature('g_aver_neighbour_target',g_aver_neighbor.get(row.tid,0))
        set_feature('g_aver_degree_source',g_aver_degree.get(row.sid,0))
        set_feature('g_aver_degree_target',g_aver_degree.get(row.tid,0))

        if i%10000==0:
            print(i,"training examples processsed")
    feature_df=pd.DataFrame(data=features)
    return feature_df

if __name__=='__main__':
    prepare_node_df()
    append_node_index(train_df)
    append_node_index(test_df)
    
    '''文本tfidf特征
    '''
    vectorizer=TfidfVectorizer(min_df=2)
    abstract_tfidf=vectorizer.fit_transform(node_df.simple_abstract)
    abstract_svd=TruncatedSVD(n_components=100,random_state=100).fit_transform(abstract_tfidf)
    
    '''构建有向图
    '''
    DiG=nx.DiGraph()
    G=nx.Graph()
    for row in train_df.itertuples():
        DiG.add_edge(row.sid,row.tid)
        G.add_edge(row.sid,row.tid)
    in_degree=DiG.in_degree()
    out_degree=DiG.out_degree()
    
    '''graph dict
    '''
    g_cluster=nx.algorithms.cluster.clustering(G)
    g_kcore=nx.core_number(G)
    g_pagerank=nx.pagerank(G)
    g_aver_neighbor=nx.average_neighbor_degree(G)
    g_aver_degree=nx.average_degree_connectivity(G)
        
    train_xs=get_features(train_df)
    test_xs=get_features(test_df)
    
    
    '''lightgbm
    '''
    gbm = lgb.LGBMClassifier(objective='binary',
                            num_leaves=31,
                            learning_rate=0.05,
                            n_estimators=1000,subsample=0.8,)
    gbm.fit(train_xs.values, train_df.label,verbose=200)
    test_ys=gbm.predict(test_xs.values)
    
    train_xs.to_csv(folder_name+'/train_xs.csv',index=False)
    test_xs.to_csv(folder_name+'/test_xs.csv',index=False)
    
    '''输出结果
    '''
    with open(output_file,'w') as f:
        f.write('Id,prediction\n')
        for i,val in enumerate(test_ys):
            f.write('%d,%d\n'%(i,val))
    print('预测结果输出到',output_file)
    print('验证集结果',f1_score(test_df.label,pd.read_csv(output_file).prediction))