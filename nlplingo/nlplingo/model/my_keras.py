# from dmpooling import DynamicMultiPooling
from __future__ import absolute_import
#import theano
from keras.engine.topology import Layer

# Assuming theano backend
from keras import backend as K

"""
class DynamicMultiPooling(Layer):
    def __init__(self, layers=None, **kwargs):
        super(DynamicMultiPooling, self).__init__(**kwargs)
        # self.layers = layers

    def build(self, input_shape):
        super(DynamicMultiPooling, self).build(input_shape)

    def call(self, inputs, mask=None):
        if not isinstance(inputs, list) or not len(inputs) == 2:
            raise TypeError(u'DynamicMultiPooling must be called on a list of two tensors.'
                            u' Got: ' + unicode(inputs))
        x = inputs[0]
        i = inputs[1]
        # print((type(x), x.type))
        # print((type(i), i.type))
        result, _ = theano.scan(fn=lambda v, pos: K.concatenate([K.max(v[:pos, :], axis=0),
                                                                 K.max(v[pos:, :], axis=0)]),
                                     sequences=[x, i[:,0]])
        return result

    def get_output_shape_for(self, input_shapes):
        # print(input_shapes)
        return (input_shapes[0][0], 2 * input_shapes[0][2])

    def compute_output_shape(self, input_shape):
        # Keras 2 method
        return self.get_output_shape_for(input_shape)


class DynamicMultiPooling3(Layer):
    def __init__(self, layers=None, **kwargs):
        super(DynamicMultiPooling3, self).__init__(**kwargs)
        # self.layers = layers

    def build(self, input_shape):
        super(DynamicMultiPooling3, self).build(input_shape)

    def call(self, inputs, mask=None):
        if not isinstance(inputs, list) or not len(inputs) == 2:
            raise TypeError(u'DynamicMultiPooling3 must be called on a list of two tensors.'
                            u' Got: ' + unicode(inputs))
        x = inputs[0]
        i = inputs[1]
        # print((type(x), x.type))
        # print((type(i), i.type))
        result, _ = theano.scan(fn=lambda v, pos1, pos2: K.concatenate([K.max(v[:pos1, :], axis=0),
                                                                        K.max(v[pos1:pos2, :], axis=0),
                                                                        K.max(v[pos2:, :], axis=0)]),
                                     sequences=[x, i[:,0], i[:,1]])
        return result

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], 3 * input_shapes[0][2])

    def compute_output_shape(self, input_shape):
        # Keras 2 method
        return self.get_output_shape_for(input_shape)
"""

class MyRange(Layer):
    def __init__(self, start, end, layers=None, **kwargs):
        super(MyRange, self).__init__(**kwargs)
        # self.layers = layers
        self.start = start
        self.end = end

    def get_config(self):
        config = {u'start' : self.start,
                  u'end' : self.end}
        base_config = super(MyRange, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        super(MyRange, self).build(input_shape)

    def call(self, word_vec, mask=None):
        result = word_vec[:,self.start:self.end:,:]
        return result

    def get_output_shape_for(self, input_shape):
        # Keras 1 method
        if self.end is None:
            end = input_shape[1]
        elif self.end < 0:
            end = input_shape[1] + self.end
        else:
            end = self.end
        if self.start < 0:
            start = input_shape[1] + self.start
        else:
            start = self.start
        delta = end - start
        print(u'MyRange delta = {0}'.format(delta))
        return (input_shape[0], delta, input_shape[2])

    def compute_output_shape(self, input_shape):
        # Keras 2 method
        return self.get_output_shape_for(input_shape)


class MySelect(Layer):
    def __init__(self, index, layers=None, **kwargs):
        super(MySelect, self).__init__(**kwargs)
        # self.layers = layers
        self.index = index

    def get_config(self):
        config = {u'index' : self.index}
        base_config = super(MySelect, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        super(MySelect, self).build(input_shape)

    def call(self, x, mask=None):
        return x[:, self.index, :]

    def get_output_shape_for(self, input_shape):
        return ((input_shape[0], input_shape[2]))

    def compute_output_shape(self, input_shape):
        return self.get_output_shape_for(input_shape)


class ShiftPosition(Layer):
    def __init__(self, shift, layers=None, **kwargs):
        super(ShiftPosition, self).__init__(**kwargs)
        self.shift = shift

    def get_config(self):
        config = {u'shift' : self.shift}
        base_config = super(ShiftPosition, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        super(ShiftPosition, self).build(input_shape)

    def call(self, x, mask=None):
        return K.maximum(x-self.shift, 1)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def compute_output_shape(self, input_shape):
        # Keras 2 method
        return self.get_output_shape_for(input_shape)

