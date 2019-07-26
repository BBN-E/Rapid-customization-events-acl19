import codecs
from collections import defaultdict

import numpy as np
from nlplingo.common.utils import F1Score
from nlplingo.event.argument.example import EventArgumentExample

def evaluate_f1(prediction, label, none_class_index, num_true=None):
    """We will input num_true if we are using predicted triggers to score arguments

    ('- prediction=', array([[ 0.00910971,  0.00806234,  0.03608446,  0.94674349],
       [ 0.02211222,  0.01518068,  0.17702729,  0.78567982],
       [ 0.01333893,  0.00946771,  0.03522802,  0.94196534],
       ...,
       [ 0.00706887,  0.01225629,  0.01827211,  0.9624027 ],
       [ 0.0132369 ,  0.03686138,  0.02967645,  0.92022526],
       [ 0.01413057,  0.03428967,  0.02316411,  0.92841566]], dtype=float32))
    ('- label=', array([[0, 0, 0, 1],
       [0, 0, 1, 0],
       [0, 0, 0, 1],
       ...,
       [0, 0, 0, 1],
       [0, 0, 0, 1],
       [0, 0, 0, 1]], dtype=int32))

    label: matrix of size (#instance, #label-types)
    So doing an argmax along 2nd dimension amounts to
    extracting the index of the true/predicted label, for each instance

    ('- label_arg_max=', array([3, 2, 3, ..., 3, 3, 3])

    :type prediction: numpy.matrix
    :type label: numpy.array
    :type none_class_index: int
    :type num_true: int

    Returns:
        nlplingo.common.utils.F1Score, dict
    """

    num_instances = label.shape[0]
    label_arg_max = np.argmax(label, axis=1)
    pred_arg_max = np.argmax(prediction, axis=1)
    return evaluate_f1_lists(pred_arg_max, label_arg_max, none_class_index, num_true)


def evaluate_f1_lists(pred_arg_max, label_arg_max, none_class_index, num_true=None):
    """We will input num_true if we are using predicted triggers to score arguments

    :type pred_arg_max: list
    :type label_arg_max: list
    :type none_class_index: int
    :type num_true: int

    Returns:
        nlplingo.common.utils.F1Score, dict
    """

    num_instances = label_arg_max.shape[0]

    # check whether each element in label_arg_max != none_class_index
    # So the result is a 1-dim vector of size #instances, where each element is True or False
    # And then we sum up the number of True elements to obtain the num# of true events
    if num_true is None:
        num_true = np.sum(label_arg_max != none_class_index)
    num_predict = np.sum(pred_arg_max != none_class_index)

    c = 0
    for i, j in zip(label_arg_max, pred_arg_max):
        if i == j and i != none_class_index:
            c += 1

    # calculate F1 for each label type
    recall_counts = defaultdict(int)
    precision_counts = defaultdict(int)
    correct_counts = defaultdict(int)
    for i in range(len(label_arg_max)):
        if label_arg_max[i] != none_class_index:
            recall_counts[label_arg_max[i]] += 1
            if pred_arg_max[i] == label_arg_max[i]:
                correct_counts[label_arg_max[i]] += 1
    for i in range(len(pred_arg_max)):
        if pred_arg_max[i] != none_class_index:
            precision_counts[pred_arg_max[i]] += 1

    score_breakdown = dict()
    for label_class in recall_counts.keys():
        label_r = recall_counts[label_class]
        if label_class in precision_counts:
            label_p = precision_counts[label_class]
        else:
            label_p = 0
        if label_class in correct_counts:
            label_c = correct_counts[label_class]
        else:
            label_c = 0
        score_breakdown[label_class] = F1Score(label_c, label_r, label_p)

    return F1Score(c, num_true, num_predict), score_breakdown


def evaluate_arg_f1_lists(event_domain, gold_data, predict_data):

    # tabulate for score_breakdown by each label type
    recall_counts = defaultdict(int)
    precision_counts = defaultdict(int)
    correct_counts = defaultdict(int)

    c = len(gold_data.intersection(predict_data))
    num_true = len(gold_data)
    num_predict = len(predict_data)

    for d in gold_data:
        label = d[1][1]
        recall_counts[event_domain.get_event_role_index(label)] += 1

    for d in predict_data:
        label = d[1][1]
        index = event_domain.get_event_role_index(label)
        precision_counts[index] += 1
        if d in gold_data:
            correct_counts[index] += 1

    score_breakdown = dict()
    for label_class in recall_counts.keys():
        label_r = recall_counts[label_class]
        if label_class in precision_counts:
            label_p = precision_counts[label_class]
        else:
            label_p = 0
        if label_class in correct_counts:
            label_c = correct_counts[label_class]
        else:
            label_c = 0
        score_breakdown[label_class] = F1Score(label_c, label_r, label_p)

    return F1Score(c, num_true, num_predict), score_breakdown, gold_data


def evaluate_arg_f1(event_domain, test_label, test_examples, predictions, scoring_domain=None,
                    gold_labels=None):
    """
    :type event_domain: nlplingo.event.event_domain.EventDomain
    :type test_label: np.array
    :type test_examples: list[nlplingo.event.event_argument.EventArgumentExample]
    :type scoring_domain: nlplingo.event.event_domain.EventDomain

    Returns:
        common.utils.F1Score
    """
    assert len(test_label) == len(test_examples)
    assert len(predictions) == len(test_examples)

    none_class_index = event_domain.get_event_role_index('None')

    if gold_labels is not None:
        gold_data = gold_labels
    else:
        gold_data = set()
        test_arg_max = np.argmax(test_label, axis=1)
        for i, index in enumerate(test_arg_max):
            if index != none_class_index:
                eg = test_examples[i]
                """:type: nlplingo.event.event_argument.EventArgumentExample"""
                if (
                    (scoring_domain is not None and scoring_domain.event_type_in_domain(eg.anchor.label)) or
                    scoring_domain is None
                ):
                    id_ = (
                        eg.sentence.docid,
                        eg.argument.head().start_char_offset(),
                        eg.argument.head().end_char_offset()
                    )
                    if isinstance(eg, EventArgumentExample):
                        label = (
                            eg.anchor.label,
                            event_domain.get_event_role_from_index(index)
                        )
                    elif isinstance(eg, ArgumentExample):
                        label = (
                            eg.event_type,
                            event_domain.get_event_role_from_index(index)
                        )
                    else:
                        raise RuntimeError('test_example not an instance of an implemented type.')
                    gold_data.add((id_, label))
                    # tabulating for score_breakdown
                    #recall_counts[index] += 1

    predict_data = set()
    pred_arg_max = np.argmax(predictions, axis=1)
    for i, index in enumerate(pred_arg_max):
        if index != none_class_index:
            # pred_scores = predictions[i]
            # score_strings = []
            # for j, score in enumerate(pred_scores):
            #     if score >= 0.5:
            #         score_strings.append('{}:{}'.format(str(j), '%.2f' % score))
            #print('{}: {}'.format(index, ', '.join(score_strings)))

            eg = test_examples[i]
            """:type: nlplingo.event.event_argument.EventArgumentExample"""
            if (
                (scoring_domain is not None and scoring_domain.event_type_in_domain(eg.anchor.label)) or
                scoring_domain is None
            ):
                id_ = (
                    eg.sentence.docid,
                    eg.argument.head().start_char_offset(),
                    eg.argument.head().end_char_offset()
                )
                if isinstance(eg, EventArgumentExample):
                    label = (
                        eg.anchor.label,
                        event_domain.get_event_role_from_index(index)
                    )
                elif isinstance(eg, ArgumentExample):
                    label = (
                        eg.event_type,
                        event_domain.get_event_role_from_index(index)
                    )
                else:
                    raise RuntimeError('test_example not an instance of an implemented type.')
                predict_data.add((id_, label))

    # predict_data = set()
    # for i in range(len(predictions)):
    #     scores = predictions[i]
    #     for index in range(len(scores)):
    #         score = scores[index]
    #         if score >= 0.5 and index != none_class_index:
    #             eg = test_examples[i]
    #             """:type: nlplingo.event.event_argument.EventArgumentExample"""
    #             if (scoring_domain is not None and scoring_domain.event_type_in_domain(
    #                     eg.anchor.label)) or scoring_domain is None:
    #                 id = '{}_{}_{}'.format(eg.sentence.docid,
    #                                        eg.argument.head().start_char_offset(),
    #                                        eg.argument.head().end_char_offset())
    #                 if isinstance(eg, EventArgumentExample):
    #                     label = '{}_{}'.format(eg.anchor.label,
    #                                            event_domain.get_event_role_from_index(index))
    #                 elif isinstance(eg, ArgumentExample):
    #                     label = '{}_{}'.format(eg.event_type,
    #                                            event_domain.get_event_role_from_index(index))
    #                 predict_data.add('{}__{}'.format(id, label))
    return evaluate_arg_f1_lists(event_domain, gold_data, predict_data)


def print_score_breakdown(extractor, score_breakdown):
    """
    :type extractor: nlplingo.nn.extractor.Extractor
    :type score_breakdown: dict
    """
    for index, f1_score in score_breakdown.items():
        et = extractor.domain.get_event_type_from_index(index)
        print('{}\t{}'.format(et, f1_score.to_string()))


def write_score_to_file(extractor, score, score_breakdown, filepath):
    """
    :type extractor: nlplingo.nn.extractor.Extractor
    :type score: nlplingo.common.utils.F1Score
    :type score_breakdown: dict
    :type filepath: str
    """
    with codecs.open(filepath, 'w', encoding='utf-8') as f:
        f.write(score.to_string() + '\n')
        for index, f1_score in score_breakdown.items():
            et = extractor.domain.get_event_type_from_index(index)
            f.write('{}\t{}\n'.format(et, f1_score.to_string()))

