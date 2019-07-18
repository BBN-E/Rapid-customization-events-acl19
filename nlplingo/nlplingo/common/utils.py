from __future__ import absolute_import
from __future__ import division


class IntPair(object):
    """A utility class to store a pair of integers
   
    Attributes:
        first: first integer
        second: second integer
    """

    def __init__(self, first, second):
        self.first = first
        self.second = second

    def to_string(self):
        return '({},{})'.format(self.first, self.second)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.first == other.first and self.second == other.second
        return False

    def __ne__(self, other):
        return self.first != other.first or self.second != other.second


class Struct:
    """A structure that can have any fields defined

    Example usage:
    options = Struct(answer=42, lineline=80, font='courier')
    options.answer (prints out 42)
    # adding more
    options.cat = 'dog'
    options.cat (prints out 'dog')
    """
    def __init__(self, **entries):
        self.__dict__.update(entries)


class F1Score(object):
    def __init__(self, c, num_true, num_predict, class_label='class_label'):
        self.c = c
        self.num_true = num_true
        self.num_predict = num_predict
        self.class_label = class_label
        self.calculate_score()

    def calculate_score(self):
        if self.c > 0 and self.num_true > 0:
            self.recall = float(self.c) / self.num_true
        else:
            self.recall = 0

        if self.c > 0 and self.num_predict > 0:
            self.precision = float(self.c) / self.num_predict
        else:
            self.precision = 0

        if self.recall > 0 and self.precision > 0:
            self.f1 = (2 * self.recall * self.precision) / (self.recall + self.precision)
        else:
            self.f1 = 0

    def to_string(self):
        return '%s #C=%d,#R=%d,#P=%d R,P,F=%.2f,%.2f,%.2f' % (self.class_label, self.c, self.num_true, self.num_predict, self.recall, self.precision, self.f1)


