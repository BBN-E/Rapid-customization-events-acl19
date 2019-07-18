import sys
import os
import codecs
import argparse
import subprocess

import pycrfsuite

from nlplingo.common.parameters import Parameters
from ner_feature import NerFeature
import decoder
from decoder import Decoder


def read_conll(filepath):
    corpus = []

    f = codecs.open(filepath, 'r', encoding='utf8')
    sentence = []
    for line in f:
        line = line.strip()
        if len(line) == 0:
            corpus.append(sentence)
            sentence = []
        else:
            tokens = line.split()
            if len(tokens)==3:
                sentence.append((tokens[0], tokens[1], tokens[2]))
            elif len(tokens)==2:
                sentence.append((tokens[0], 'NN', tokens[1]))	# default all pos-tags to NN

    f.close()

    return corpus


def prepare_training(train_file, ner_fea, content_type, params):
    dec = Decoder(params)

    train_corpus = read_conll(train_file)

    print("There are %d sentences in our training corpus" % (len(train_corpus)))
    sent_read = 0
    examples = []
    for sent in train_corpus:
        (labels, feas, words) = get_labels_and_features(sent, ner_fea, content_type)
        word_seq = dec.instances(feas)  # word_seq is of type pycrfsuite.ItemSequence
        examples.append((word_seq, labels))
        sent_read += 1
        if (sent_read % 10)==0:
            print("Prepared features for %s sentences" % (sent_read))

    return examples


def do_training(train_examples, model_file, l1, l2, max_iterations):
    trainer = pycrfsuite.Trainer(verbose=True)

    # set training parameters
    trainer.set_params({
        'c1': l1,   # coefficient for L1 penalty ; 1.0
        'c2': l2,  # coefficient for L2 penalty ; 1e-3
        'max_iterations': max_iterations,  # stop earlier
        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })

    for (word_seq, labels) in train_examples:
        trainer.append(word_seq, labels)

    trainer.train(model_file)
    print len(trainer.logparser.iterations), trainer.logparser.iterations[-1]


def prepare_tagging(test_file, ner_fea, content_type, params):
    dec = Decoder(params)

    test_corpus = read_conll(test_file)

    print("There are %d sentences in our test corpus" % (len(test_corpus)))
    sent_read = 0
    examples = []
    for sent in test_corpus:
        (labels, feas, words) = get_labels_and_features(sent, ner_fea, content_type)
        word_seq = dec.instances(feas)  # word_seq is of type pycrfsuite.ItemSequence
        examples.append( (word_seq, labels, words) )
        sent_read += 1
        if (sent_read % 10)==0:
            print("Prepared features for %s sentences" % (sent_read))
 
    return examples


def do_tagging(test_examples, model_file, prediction_file):
    tagger = pycrfsuite.Tagger()
    tagger.open(model_file)

    f = codecs.open(prediction_file, 'w', encoding='utf8')

    for (word_seq, labels, words) in test_examples:
        tagger.set(word_seq)
        predictions = tagger.tag()
        
        assert len(labels) == len(predictions)
        i = 0
        for gold_label, pred_label in zip(labels, predictions):
            w = words[i]
            f.write('%s\t%s\t%s' % (w, gold_label, pred_label))    
            f.write('\n')
            i += 1
        f.write('\n')    

    f.close()


def do_scoring(prediction_file):
    score_conll_script = os.path.join(os.getcwd(), "..", "ner-common", "score_conll.pl")
    results = prediction_file + ".score"
    OUT = open(results, 'w')
    subprocess.call(["perl", score_conll_script, prediction_file], stdout=OUT)
    with open(results) as IN:
        for line in IN:
            if line.startswith("ALL"):
                print(prediction_file + " " + line.strip())

def get_labels_and_features(sent, ner_fea, content_type):
    words  = []
    d = ('', '', '')
    seq = [d, d]
    for word, postag, label in sent:
        seq.append((ner_fea.encode(word), postag, label))
        words.append(word)
    seq.append(d)
    seq.append(d)

    feas = ner_fea.extract_features(seq, content_type)
    labels = get_labels(feas)
   
    return (labels, feas, words)


def get_labels(word_feas):
    ret = []
    for line in word_feas:
        fields = line.split('\t')
        ret.append(fields[0])
    return ret


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('exptdir', help='experiment directory')
    parser.add_argument('params', help='parameter file')
    parser.add_argument('content_type', choices=['SocialMediaPosting','Blog','NewsArticle'])
    parser.add_argument('--train', nargs='+')
    parser.add_argument('--test', nargs='+', default=[])
    parser.add_argument('--L1', type=float, help='hyperparameter(s) for L1 regularization', nargs='*', default=[0.1])
    parser.add_argument('--L2', type=float, help='hyperparameter(s) for L2 regularization', nargs='*', default=[0.1])
    parser.add_argument('--max_iterations', type=int, help='maximum training iterations', default=100)
    parser.add_argument('--overwrite', action='store_true', help='Allow existing directory to be used (and overwritten where applicable)')

    args = parser.parse_args()

    try:
        os.mkdir(args.exptdir)
    except:
        if not args.overwrite:
            raise Exception("%s already exists (and --overwrite is not specified)" % args.exptdir)

    # Read parameter file
    overrides = {}
    overrides['base_dir'] = os.path.join(os.getcwd(), "..", "..")    
    overrides['content_type'] = args.content_type
    params = Parameters(args.params, overrides)

    # Cache parameters
    with open(os.path.join(args.exptdir, "params.txt"), 'w') as OUT:
        params.print_params(OUT)

    # Load up feature creation object
    ner_fea = NerFeature(params)

    # Create combined training file, if necessary
    training_file = None
    if args.train:
        training_file = args.train[0]
        if len(args.train) > 1:
            print("Combining training files into single file")
            training_file = os.path.join(args.exptdir, "combined_training.txt")
            with open(training_file, 'w') as TRAIN:
                for individual_training_file in args.train:
                    with open(individual_training_file) as infile:
                        for line in infile:
                            TRAIN.write(line)

    # Make sure that all test basenames are unique
    unique_test_names = set()
    for test_file in args.test:
        basename = os.path.basename(test_file)
        if basename in unique_test_names:
            raise Exception('To test multiple files, their names (ignoring path) must be unique')
        unique_test_names.add(basename)

    train_examples = prepare_training(training_file, ner_fea, args.content_type, params)
    # Grid search
    for L1 in args.L1:
        for L2 in args.L2:
            model_file = os.path.join(args.exptdir, "%0.2f_%0.2f.model" % (L1, L2))
            if training_file:
                print("Training with L1=%0.2f and L2=%0.2f" % (L1, L2))
                do_training(train_examples, model_file, L1, L2, args.max_iterations)

                
    for test_file in args.test:
        print ("Testing", test_file)
        test_examples = prepare_tagging(test_file, ner_fea, args.content_type, params)
        for L1 in args.L1:
            for L2 in args.L2:
                model_file = os.path.join(args.exptdir, "%0.2f_%0.2f.model" % (L1, L2))
                out_file = os.path.join(args.exptdir, "%s.%0.2f_%0.2f.out" % (os.path.basename(test_file), L1, L2))
                print("Doing tagging to %s" % (out_file))
                do_tagging(test_examples, model_file, out_file)
                do_scoring(out_file)

        
