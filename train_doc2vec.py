from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from datetime import datetime
import sys

class LabeledLineSentence(object):
    def __init__(self, filename):
        self.filename = filename
    def __iter__(self):
        for uid, line in enumerate(open(self.filename)):
            words=line.strip().split()
            yield LabeledSentence(words[1:],[words[0]])

            
if __name__=='__main__':
    print(sys.argv[0])
    input_file=sys.argv[1]
    output_file=sys.argv[2]
    
    sentences=LabeledLineSentence(input_file)
    model = Doc2Vec(alpha=0.025, min_alpha=0.025)  # use fixed learning rate
    model.build_vocab(sentences)
    for epoch in range(10):
        print(datetime.now(),'epoch:',epoch)
        model.train(sentences)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay
    model.save(output_file)
    print('运行完毕，文档向量已输出到',output_file)
