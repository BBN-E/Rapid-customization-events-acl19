
import codecs
import json
import sys

from nlplingo.common.utils import ComplexEncoder
from wordnet import SynsetWords
from wordnet import WordNetManager
from wordnet import SynsetInfo


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Require an input param file with values: wordnet_dir, lemma_list, output_dir')
        sys.exit(1)

    with open(sys.argv[1], 'r') as f:
        params = json.load(f)

    wn_dir = params['wordnet_dir']
    lemma_list = params['lemma_list']
    output_dir = params['output_dir']

    # for each sense in the sense_list, we will produce synonyms, hypernyms, siblings
    wn = WordNetManager( wn_dir + '/data.noun', wn_dir + '/data.verb', wn_dir + '/index.sense')

    with open(lemma_list, 'r') as f:
        lemma_json = json.load(f)


    wn_info = []
    for lemma_data in lemma_json:
        lemma = lemma_data['lemma'].replace(' ', '_')
        pos_category = lemma_data['pos_category']

        # we are only interested in nouns and verbs, so the following might return None
        lemma_senses = wn.get_senses_for_lemma(lemma, pos_category)
        """:type: list[Sense]"""

        if lemma_senses is None:
            continue

        for sense in lemma_senses:
            lexsn = sense.id
            synset = sense.synset

            synset_info = SynsetInfo(lemma.replace('_', ' '), lexsn, pos_category, synset.gloss)
            synset_info.synonyms = [word.replace('_', ' ') for word in synset.words if word != lemma]

            hyper_synset = synset
            all_hypernym_paths = []
            """:type: list[list[Synset]]"""
            wn.get_all_hypernym_paths(hyper_synset, [], all_hypernym_paths)
            for path in all_hypernym_paths:
                # we only grab hypernym-synsets that are max X hops away, i.e. synset-parent, synset-parent-parent
                synset_info.add_hypernym_path([SynsetWords(hyper_s.words) for hyper_s in path[0:]])

            #print('Grabbing hyponyms for {} {}'.format(lexsn, synset.to_string()))
            hypo_synset = synset
            all_hyponym_paths = []
            """:type: list[list[Synset]]"""
            wn.get_all_hyponym_paths(hypo_synset, [], all_hyponym_paths)
            for path in all_hyponym_paths:
                # we only grab hyponym-synsets that are max X hops away, i.e. synset-parent, synset-parent-parent
                synset_info.add_hyponym_path([SynsetWords(hypo_s.words) for hypo_s in path[0:]])

            sibling_synsets = wn.get_sibling_synsets(synset)
            """:type: set(Synset)"""
            for sibling in sibling_synsets:
                synset_info.add_sibling(SynsetWords(sibling.words))

            wn_info.append(synset_info)

    with codecs.open(output_dir + '/wn_info.json', 'w', encoding='utf-8') as o:
        o.write(json.dumps(wn_info, indent=4, cls=ComplexEncoder, ensure_ascii=False))
