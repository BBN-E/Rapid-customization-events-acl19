import collections

EventMention = collections.namedtuple('EventMention',['docId','eventTypeName','sentStartCharOffset','sentEndCharOffset','anchorStartCharOffset','anchorEndCharOffset','positive'])
SentenceInfoKey = collections.namedtuple('SentenceInfoKey',['docId','sentenceId'])