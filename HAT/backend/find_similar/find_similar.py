try:
    # Python 3.x
    from urllib.parse import quote_plus
except ImportError:
    # Python 2.x
    from urllib import quote_plus
#import pymongo
import numpy as np
import json

from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer

def getSimilarClusterInWords(exist_triggers,threshold,cutoff,my_db):
    '''
    Attention!!!
    This function need a table called pairwise_m5m6m9 support at database
    :return:
    '''

    trigger_words = set([trigger.split("_")[-1] for trigger in exist_triggers])
    calculated_map = {}
    for trigger_word in trigger_words:
        for line in my_db.find(trigger_word):
            src = line['src']
            dst = line['dst']
            score = line['score']
            calculated_map[dst] = calculated_map.get(dst, 0) + score
    out = []
    for key,val in calculated_map.items():
        if val / len(trigger_words) >= threshold:
            wordphrase = [key]
            out.extend([(trigger,val / len(trigger_words)) for trigger in wordphrase if trigger not in exist_triggers])
    out = sorted(out,key=lambda i:i[1],reverse=True)
    #out = list(map(lambda x:{'key':x[0],'aux':{},'type':'trigger'},out[:min(cutoff,len(out))]))
    out = list(map(lambda x:x[0],out[:min(cutoff,len(out))]))
    return out

class FakeDB:
    def __init__(self, idx_map, sim_mtx):
        self._idx_map = idx_map
        ridx_map = dict()
        for key,value in idx_map.items():
            ridx_map[value] = key
        self._ridx_map = ridx_map
        self._sim_mtx = sim_mtx
    def find(self, trigger_word):
        if trigger_word in self._idx_map:
            entries = []
            idx1 = self._idx_map[trigger_word]
            for key, value in self._idx_map.items():
                idx2 = value
                entry = {
                    'src': trigger_word,
                    'dst': key,
                    'score':self._sim_mtx[idx1,idx2],
                    }
                entries.append(entry)
            return entries
        return []


if __name__ == "__main__":
    mtx = np.load('/nfs/raid87/u13/users/jfaschin/investigating_nn_events_vs_untyped_relations/matrix.npz.npy')
    idx_map = json.load(open('/nfs/raid87/u13/users/jfaschin/investigating_nn_events_vs_untyped_relations/index.json'))
    #keywords = json.load(open('/nfs/raid87/u13/users/jfaschin/event_keywords.ui.7.json'))
    keywords = json.load(open('/nfs/raid87/u13/users/jfaschin/recovered_keyword_newtypes.json'))
    my_db = FakeDB(idx_map,mtx)
    wn_l = WordNetLemmatizer()
    stem = SnowballStemmer('english')
    # for entry in keywords:
    #     print("Event Type: {}".format(entry['event_type']))
    #     exist_triggers = [x.split('.')[0] for x in entry['keywords']]
    #     exist_triggers = [wn_l.lemmatize(x) for x in exist_triggers]
    #     exist_triggers = [stem.stem(x) for x in exist_triggers]
    #     result = getSimilarClusterInWords(exist_triggers,0,30,my_db)
    #     for word in result:
    #         print(word)
    #     print()
        
    # Please lemmaized the below array.
    exist_triggers = ["rain","flood"]
    
    print('Done')
    print(getSimilarClusterInWords(exist_triggers,0,30,my_db))
