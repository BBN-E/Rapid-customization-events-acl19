from __future__ import absolute_import

import spacy

import config
from ace.ace_utils import event_stats, sentence_stats, event_stats_by_partition, role_stats

if __name__ == u"__main__":
    nlp = spacy.load(u'en')

    print(u'\n== Event statistics')
    event_stats(config.ace_data_dir, display=1)

    print(u'\n== Event statistics by parition')
    event_stats_by_partition(config.ace_data_dir, display=1)

    print(u'\n== Sentence statistics')
    sentence_stats(config.ace_data_dir, display=0, nlp=nlp)

    print(u'\n== Role statistics')
    role_stats()
