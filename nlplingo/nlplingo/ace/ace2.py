from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ace.ace_docs import AceCorpus

from ace.ace_utils import readHengjiPartitionLists

corpus = AceCorpus()
partitions = readHengjiPartitionLists()

timeWords = ['time', 'a.m.', 'am', 'p.m.', 'pm', 'A.M.', 'AM', 'P.M.', 'PM', 'day', 'days', 'week', 'weeks', 'month', 'months', 'year', 'years', 'morning', 'afternoon', 'evening', 'night', 'anniversary', 'birtday', 'second', 'seconds', 'minute', 'minutes', 'hour', 'hours', 'decade', 'decades', 'era', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'today', 'yesterday', 'tomorrow', 'past', 'future', 'present', 'Jan', 'Jan.', 'Feb', 'Feb.', 'Mar', 'Mar.', 'Apr', 'Apr.', 'Jun', 'Jun.', 'Jul', 'Jul.', 'Aug', 'Aug.', 'Sept', 'Sept.', 'Oct', 'Oct.', 'Nov', 'Nov.', 'Dec', 'Dec.', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december', 'jan.', 'feb', 'feb.', 'mar', 'mar.', 'apr', 'apr.', 'jun', 'jun.', 'jul', 'jul.', 'aug', 'aug.', 'sept', 'sept.', 'oct', 'oct.', 'nov', 'nov.', 'dec', 'dec.', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

timeWordSet = set(timeWords)


def get_time_head(tokens):
    tokenText = [tk.text for tk in tokens]
    tokenTextSet = set(tokenText)
    if len(tokens) == 1:
        return tokens[0]
    if timeWordSet.isdisjoint(tokenTextSet):
        return tokens[-1]
    for word in timeWords:
        if word in tokenTextSet:
            index = tokenText.index(word)
            return tokens[index]

ofToken = NLP('of')[0]
toToken = NLP('to')[0]
atToken = NLP('at')[0]
def get_crime_head(tokens):
    head = [tk for tk in tokens if tk.pos_=='VERB']
    if head:
        return(head[0], 'verb')
    for i, tk in enumerate(tokens):
        if i > 0 and (tk.text == ofToken.text or tk.text == toToken.text):
            return(tokens[i-1], 'of')
    return(tokens[-1], 'last')

def get_job_title_head(tokens):
    for i, tk in enumerate(tokens):
        if i > 0 and (tk.text == ofToken.text or tk.text == atToken.text):
            print('{} <- {} {}'.format(tokens[i-1], 'of', tokens))
            return(tokens[i-1], 'of')
    print('{} <- {} {}'.format(tokens[-1], 'last', tokens))
    return(tokens[-1], 'last')
    
        
for name, partition in partitions.items():
    print('====' + name)
    for docPath in partition:
        doc = corpus.getDocumentByPath(docPath)
        for time in doc.times:
            for mention in time.mentions:
                tokens = doc.annotation.lookupTokens(mention.id)
                head = get_time_head(tokens)
                if len(tokens) > 1:
                    print('{} <- {}'.format(head, tokens))
    break


for name, partition in partitions.items():
    print('====' + name)
    for docPath in partition:
        doc = corpus.getDocumentByPath(docPath)
        for value in doc.values:
            for mention in value.mentions:
                tokens = doc.annotation.lookupTokens(mention.id)
                head = None
                if len(tokens) > 1:
                    if value.type == 'Numeric' or value.type == 'Sentence':
                        head = tokens[-1]
                    if head:
                        print('{} <- {}'.format(head, tokens))
    break

for name, partition in partitions.items():
    print('====' + name)
    for docPath in partition:
        doc = corpus.getDocumentByPath(docPath)
        for value in doc.values:
            for mention in value.mentions:
                tokens = doc.annotation.lookupTokens(mention.id)
                head = None
                if len(tokens) > 1:
                    if value.type == 'Crime':
                        head, reason = get_crime_head(tokens)
                        # print('{} <- {} {}'.format(head, reason, tokens))
                    elif value.type == 'Job-Title':
                        head = get_job_title_head(tokens)
    break


for name, partition in partitions.items():
    print('====' + name)
    for docPath in partition:
        doc = corpus.getDocumentByPath(docPath)
        for entity in doc.entities:
            print([mention.text for mention in entity.mentions])
    break

