import numpy as np
from collections import defaultdict
from nlplingo.common.utils import F1Score
from nlplingo.event.event_mention import EventMentionGenerator
from nlplingo.event.event_argument import EventArgumentExample
from nlplingo.event.argument import ArgumentExample


def evaluate_accuracy(prediction, label):
    #label_arg_max = np.argmax(label, axis=1)
    #pred_arg_max = np.argmax(prediction, axis=1)

    #label_arg_max = np.asarray(label)
    #pred_arg_max = np.asarray(prediction)

    # number of instances
    num_instances = label.shape[0]

    c = 0
    for i in range(len(prediction)):
        if (prediction[i] >= 0.5 and label[i] == 1) or (prediction[i] < 0.5 and label[i] == 0):
            c += 1

    accuracy = float(c) / num_instances
    return accuracy


def print_pair_example(example, class_label, prediction):
    eg1 = example.eg1
    eg2 = example.eg2
    eg1_text = EventMentionGenerator.sentence_text_with_markup(eg1.sentence, eg1.anchor, eg1.argument)
    eg2_text = EventMentionGenerator.sentence_text_with_markup(eg2.sentence, eg2.anchor, eg2.argument)

    anchor_sim = None
    if eg1.anchor.head().has_vector and eg2.anchor.head().has_vector:
        v1 = eg1.anchor.head().word_vector
        v2 = eg2.anchor.head().word_vector
        anchor_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    obj_sim = None
    if eg1.anchor_obj is not None and eg2.anchor_obj is not None:
        if eg1.anchor_obj.has_vector and eg2.anchor_obj.has_vector:
            v1 = eg1.anchor_obj.word_vector
            v2 = eg2.anchor_obj.word_vector
            obj_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    if anchor_sim is not None and obj_sim is not None:
        avg_sim = (anchor_sim + obj_sim) / 2.0
    elif anchor_sim is not None:
        avg_sim = anchor_sim
    else:
        avg_sim = 0

    eg1_obj_string = None
    if eg1.anchor_obj is not None:
        eg1_obj_string = eg1.anchor_obj.text
    eg2_obj_string = None
    if eg2.anchor_obj is not None:
        eg2_obj_string = eg2.anchor_obj.text

    anchor_strings = 'anchors {} {}\n'.format(eg1.anchor.head().text, eg2.anchor.head().text)
    obj_strings = 'objs {} {}\n'.format(eg1_obj_string, eg2_obj_string)
    print('event_cnn.print_pair_example: {} {}\n- {}\n- {}\nanchor_sim={} obj_sim={} avg_sim={}\n'.format(class_label, prediction, eg1_text, eg2_text, anchor_sim, obj_sim, avg_sim))
    print(anchor_strings)
    print(obj_strings)


def evaluate_f1_binary(prediction, label, examples, target_types, num_true_positive, class_label='OVERALL'):
    """We will input num_true if we are using predicted triggers to score arguments
    Given a binary task, we will always assume 0 class index is negative, 1 is positive

    :type event_domain: nlplingo.event.event_domain.EventDomain
    :type none_label_index: int
    :type class_labels: list[str]
    :type examples: list[nlplingo.event.event_pair.EventPairExample]
    :type target_types: set[str]

    target_types: a set of event types that we want to evaluate on
    Returns:
        nlplingo.common.utils.F1Score
    """
    ret = []

    #R_dict = defaultdict(int)
    #P_dict = defaultdict(int)
    #C_dict = defaultdict(int)

    print('In evaluate_f1_binary: #prediction={} #label={} #examples={}'.format(len(prediction), len(label), len(examples)))

    num_correct = 0
    num_true = 0
    num_predict = 0
    for i in range(len(prediction)):
        if examples[i].eg1.event_type in target_types or examples[i].eg2.event_type in target_types:
            if prediction[i] >= 0.5:
                #P_dict[class_labels[i]] += 1
                num_predict += 1
                #print_pair_example(examples[i], class_labels[i], prediction[i])
                if label[i] == 1:
                    num_correct += 1
                    #C_dict[class_labels[i]] += 1

    for i in range(len(label)):
        #if examples[i].eg1.event_type.startswith('Business'):
        #    print('eg1.event_type={}, eg2.event_type={}'.format(examples[i].eg1.event_type, examples[i].eg2.event_type))
        #    print('label[i]={}'.format(label[i]))
        if label[i] == 1 and examples[i].eg1.event_type in target_types:
            num_true += 1
            #R_dict[class_labels[i]] += 1
            #print_pair_example(examples[i], class_labels[i], prediction[i])

    if num_true_positive is not None:
        print('num_true={} num_true_positive={}'.format(num_true, num_true_positive))
        ret.append(F1Score(num_correct, num_true_positive, num_predict, class_label))
    else:
        print('num_true={} num_true_positive={}'.format(num_true, num_true))
        ret.append(F1Score(num_correct, num_true, num_predict, class_label))
    return ret


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
    :type label: numpy.matrix
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
