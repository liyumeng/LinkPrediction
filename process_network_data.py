'''
训练论文及作者Network Embedding向量
'''

import pandas as pd
import pickle

#输入文件
train_file='data/raw/training_set.txt'
node_file='data/raw/node_information.csv'
#输出文件
node_network_file='data/tmp/node_network.txt'
author_ids_file='data/tmp/author_ids.pkl'
author_network_file='data/tmp/author_network.txt'

#生成论文网络训练文件
train_df=pd.read_csv(train_file,sep=' ',names=['sid','tid','label'])
train_df[train_df.label==1].to_csv(node_network_file,sep='\t',index=False,header=False,columns=['sid','tid'])

#生成作者网络训练文件
node_df=pd.read_csv(node_file,names=['id','year','title','authors','journal','abstract'])
node_df.authors.fillna('',inplace=True)
node_df.loc[:,'author_set']=node_df.authors.apply(lambda x:[x.strip() for x in x.lower().split(',') if len(x.strip())>0])

author_ids={}
author_net_set=set()

def get_author_id(name):
    global author_ids
    if name not in author_ids:
        author_ids[name]=len(author_ids)
    return author_ids[name]

def append_node_index(df):
    '''向df中添加source index及target index
    依赖全局变量：node_df
    '''
    id2index=dict([(id,i) for i,id in enumerate(node_df.id)])
    df.loc[:,'sindex']=df.sid.apply(lambda x:id2index[x])
    df.loc[:,'tindex']=df.tid.apply(lambda x:id2index[x])
    
def add_authors(a,b):
    aid=get_author_id(a)
    bid=get_author_id(b)
    token='%d\t%d'%(aid,bid)
    if token not in author_net_set:
        author_net_set.add(token)

append_node_index(train_df)

count=0
for row in train_df.itertuples():
    source=node_df.loc[row.sindex]
    target=node_df.loc[row.tindex]
    
    for a in source.author_set:
        for b in target.author_set:
            add_authors(a,b)
    
    for i in range(len(source.author_set)):
        for j in range(i+1,len(source.author_set)):
            add_authors(source.author_set[i],source.author_set[j])
            
    for i in range(len(target.author_set)):
        for j in range(i+1,len(target.author_set)):
            add_authors(target.author_set[i],target.author_set[j])
    
    if count % 10000 == 0:
        print(count, "items processsed")
    count+=1

pickle.dump(author_ids,open(author_ids_file,'wb'))
print('已保存作者id对应表',author_ids_file)

with open(author_network_file,'w') as f:
    for key in author_net_set:
        f.write(key)
        f.write('\n')
    print('已保存作者网络文件',author_network_file)
print('运行完毕')
