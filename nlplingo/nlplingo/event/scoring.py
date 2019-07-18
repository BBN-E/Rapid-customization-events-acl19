import argparse
import codecs
from itertools import chain
from collections import defaultdict
import numpy as np

from nlplingo.event.event_domain import EventDomain
from nlplingo.common.scoring import evaluate_f1_lists
from nlplingo.common.scoring import evaluate_arg_f1_lists


class AnnotationEntry(object):
    def __init__(
        self,
        docid,
        label,
        start_offset,
        end_offset
    ):
        self.docid = docid
        self.label = label
        self.start_offset = start_offset
        self.end_offset = end_offset

    def is_overlap(self, threshold, other_ann):
        overlap = self.iou(other_ann)
        return self.docid == other_ann.docid and overlap > threshold

    def iou(self, other_ann):
        return float(self.intersection(other_ann))/self.union(other_ann)

    def intersection(self, other_ann):
        min_end = min(self.end_offset, other_ann.end_offset)
        max_start = max(self.start_offset, other_ann.start_offset)
        return max(0.0, min_end - max_start)

    def union(self, other_ann):
        _start = min(self.start_offset, other_ann.start_offset)
        _end = max(self.end_offset, other_ann.end_offset)
        return abs(_start - _end)


def parse_event_span_file(input_file):
    annotations = []
    with codecs.open(input_file, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            entries = line.split('\t')
            docid = entries[0]
            label = entries[1]
            start_offset = entries[2]
            end_offset = entries[3]
            annotations.append(
                AnnotationEntry(
                    docid,
                    label,
                    float(start_offset),
                    float(end_offset)
                )
            )
    return annotations


def parse_event_arg_span_file(input_file):
    annotations = []
    with codecs.open(input_file, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            entries = line.split('\t')
            docid = entries[0]
            event_label = entries[1]
            arg_label = entries[2]
            start_offset = entries[3]
            end_offset = entries[4]
            annotations.append(
                AnnotationEntry(
                    docid,
                    (event_label, arg_label),
                    float(start_offset),
                    float(end_offset)
                )
            )
    return annotations


def match_labels_to_predicted(ground_truth, predicted, threshold=1.0):
    event_types = set()
    for ann in chain(ground_truth, predicted):
        event_types.add(ann.label)
    domain = EventDomain(
        list(event_types),
        [],
        [],
        []
    )

    ground_truth_indices = set()
    indices_to_labels = defaultdict(
        lambda: {
            'ground_truth': domain.get_event_type_index('None'),
            'predicted': domain.get_event_type_index('None')
        }
    )

    for ann in ground_truth:
        key = (ann.docid, ann.start_offset, ann.end_offset)
        ground_truth_indices.add(key)

    assert (len(ground_truth_indices) == len(ground_truth))

    for ann in ground_truth:
        key = (ann.docid, ann.start_offset, ann.end_offset)
        indices_to_labels[key]['ground_truth'] = domain.get_event_type_index(ann.label)

    ###
    if threshold < 1.0:
        for ann_p in predicted:
            count = 0
            for ann_gt in ground_truth:
                if ann_p.is_overlap(threshold, ann_gt):
                    count += 1
                    key = (ann_gt.docid, ann_gt.start_offset, ann_gt.end_offset)
                    indices_to_labels[key]['predicted'] = domain.get_event_type_index(
                        ann_p.label
                    )

            if count > 1:
                raise RuntimeError(
                    'Predicted trigger matched to two or more ground truth labels'
                )
            elif count == 0:
                key = (ann_p.docid, ann_p.start_offset, ann_p.end_offset)
                indices_to_labels[key]['predicted'] = domain.get_event_type_index(ann_p.label)
    elif threshold == 1.0:
        for ann in predicted:
            key = (ann.docid, ann.start_offset, ann.end_offset)
            indices_to_labels[key]['predicted'] = domain.get_event_type_index(ann.label)
    else:
        raise RuntimeError('Threshold should not be set above 1.')

    return (
        domain,
        [x['ground_truth'] for x in indices_to_labels.values()],
        [x['predicted'] for x in indices_to_labels.values()]
    )


def match_arg_labels_to_predicted(
        ground_truth,
        predicted,
        threshold
):
    event_types = set()
    arg_roles = set()
    for ann in chain(ground_truth, predicted):
        event_types.add(ann.label[0])
        arg_roles.add(ann.label[1])
    domain = EventDomain(
        list(event_types),
        list(arg_roles),
        [],
        []
    )
    ground_truth_keys = set()
    predicted_keys = set()

    for ann in ground_truth:
        ground_truth_keys.add(
            (
                (
                    ann.docid,
                    ann.start_offset,
                    ann.end_offset,
                ),
                ann.label
            )
        )

    if threshold < 1.0:
        for ann_p in predicted:
            count = 0
            for ann_gt in ground_truth:
                if ann_p.label == ann_gt.label and ann_p.is_overlap(threshold, ann_gt):
                    count += 1
                    predicted_keys.add(
                        (
                            (
                                ann_gt.docid,
                                ann_gt.start_offset,
                                ann_gt.end_offset,
                            ),
                            ann_gt.label
                        )
                    )
            if count > 1:
                raise RuntimeError(
                    'Predicted argument matched to two or more ground truth labels'
                )
            elif count == 0:
                predicted_keys.add(
                    (
                        (
                            ann_p.docid,
                            ann_p.start_offset,
                            ann_p.end_offset,
                        ),
                        ann_p.label
                    )
                )

    elif threshold == 1:
        for ann in predicted:
            predicted_keys.add(
                (
                    (
                        ann.docid,
                        ann.start_offset,
                        ann.end_offset,
                    ),
                    ann.label
                )
            )
    else:
        raise RuntimeError('Threshold should not be set above 1.')

    return domain, ground_truth_keys, predicted_keys


def main(args):

    if args.mode == 'event':
        ground_truth = parse_event_span_file(args.ground_truth_file)
        predicted = parse_event_span_file(args.predicted_file)

        domain, labels, predicted_labels = match_labels_to_predicted(
            ground_truth,
            predicted,
            args.overlap_threshold
        )

        score, score_breakdown = evaluate_f1_lists(
            np.array(predicted_labels),
            np.array(labels),
            domain.get_event_type_index('None')
        )
        print(score.to_string())

        for index, f1_score in score_breakdown.items():
            et = domain.get_event_type_from_index(index)
            print('{}\t{}'.format(et, f1_score.to_string()))

    elif args.mode == 'event_arg':
        ground_truth = parse_event_arg_span_file(args.ground_truth_file)
        predicted = parse_event_arg_span_file(args.predicted_file)

        domain, labels, predicted_labels = match_arg_labels_to_predicted(
            ground_truth,
            predicted,
            args.overlap_threshold
        )
        score, score_breakdown, predicted_labels = evaluate_arg_f1_lists(
            domain,
            labels,
            predicted_labels
        )

        print(score.to_string())

        for index, f1_score in score_breakdown.items():
            er = domain.get_event_role_from_index(index)
            print('Arg-scores with predicted-triggers: {}\t{}'.format(er, f1_score.to_string()))

    else:
        print('{} mode not implemented. Exiting.'.format(args.mode))


def parse_setup():
    parser = argparse.ArgumentParser(
        description='Score text span detection problems.')
    parser.add_argument(
        '--ground_truth_file',
        required=True,
        help='Ground truth span file'
    )
    parser.add_argument(
        '--predicted_file',
        required=True,
        help='Predicted span file'
    )
    parser.add_argument(
        '--overlap_threshold',
        type=float,
        default=1.0,
        help='Minimum overlap for text string match'
    )

    parser.add_argument(
        '--mode',
        required=True,
        type=str,
        help='Either event or event_arg'
    )
    return parser


if __name__ == '__main__':
    main(parse_setup().parse_args())
