# Copyright 2015 by Raytheon BBN Technologies Corp.
# All Rights Reserved.

"""
Python API for Accessing SerifXML Files.

    >>> import serifxml3
    >>> document_text = '''
    ...     John talked to his sister Mary.
    ...     The president of Iran, Joe Smith, said he wanted to resume talks.
    ...     I saw peacemaker Bob at the mall.'''
    >>> serif_doc = serifxml3.send_serifxml_document(document_text)
    >>> print serif_doc
    Document:
      docid = u'anonymous'
      language = u'English'
      source_type = u'UNKNOWN'
      ...

or, to load a document from a file:

    >>> serif_doc = serifxml3.Document(filename)

to save a document to a file:

    >>> serif_doc.save(output_filename)

For a list of attributes that any serif theory object takes, use the
'help' method on the theory object (or its class).  E.g.:

    >>> serif_doc.sentences[0].help()
    >>> MentionSet.help()
"""

import os, re, select, socket, sys, textwrap, time, weakref
from optparse import OptionParser
from xml.etree import ElementTree as ET

def escape_cdata_carriage_return(text, encoding='us-ascii'):
    """
    Source copied from ElementTree.py and modified to add
    '\r' -> '&#xD;' replacement. Monkey patch!
    """
    # escape character data
    try:
        # it's worth avoiding do-nothing calls for strings that are
        # shorter than 500 character, or so.  assume that's, by far,
        # the most common case in most applications.
        if "&" in text:
            text = text.replace("&", "&amp;")
        if "<" in text:
            text = text.replace("<", "&lt;")
        if ">" in text:
            text = text.replace(">", "&gt;")
        if "\r" in text:
            text = text.replace("\r", "&#xD;")
        # Need to return a string, so after patching up the XML,
        # we need to decode it again...  I'm not convinced this
        # actually does anything.  I haven't found a counterexample
        # yet. -DJE
        return text.encode(encoding, "xmlcharrefreplace").decode(encoding)
    except (TypeError, AttributeError):
        ET._raise_serialization_error(text)


ET._escape_cdata = escape_cdata_carriage_return

SERIFXML_VERSION = 18

"""If true, then SerifTheory objects will keep a pointer to the
   ElementTree.Element that they were constructed from.  This
   makes it possible for the save() method to preserve any extra
   attributes or elements that were present in the original
   document."""
KEEP_ORIGINAL_ETREE = False


######################################################################
# { Theory Attribute Specifications
######################################################################

class _AutoPopulatedXMLAttributeSpec(object):
    """
    This is the abstract base class for \"Auto-populated XML attribute
    specifications\" (or AttributeSpec's for short).  Each
    AttributeSpec is used to define a single attribute for a Serif
    theory class.  Some examples of AttributeSpecs are::

        is_downcased = _SimpleAttribute(bool, default=False)
        sentences    = _ChildTheoryElement('Sentences')
        start_token  = _ReferenceAttribute('start_token', is_required=True)

    Each AttributeSpec defines a `set_value()` method, which is used
    to read the attribute's value from the XML input for a given
    theory object.  The default implementation of `set_value()` calls
    the abstract method `get_value()`, which should read the
    appropriate value from a given XML node, and stores it in the
    theory object (using `setattr`).

    The name of the instance variable that is used to store an
    attribute's value is always identical to the name of the class
    variable that holds the AttributeSpec.  For example, the Document
    class contains an AttributeSpec named 'docid'; and each instance
    of the Document class will have an instance variable with the same
    name ('docid') that is initialized by that AttributeSpec.  Note
    that this instance variable (containing the attribute value)
    shadows the class variable containing the AttributeSpec.
    """

    # We assign a unique attribute number to each AttributeSpec that
    # gets created.  This allows us to display attributes in the
    # correct order when pretty-printing.  (In particular, attributes
    # are displayed in the order in which they were defined.)
    attribute_counter = 0

    def __init__(self):
        self._attribute_number = self.attribute_counter
        _AutoPopulatedXMLAttributeSpec.attribute_counter += 1

    def set_value(self, etree, theory):
        """
        Set the value of this attribute.

        @param name: The name that should be used to store the attribute.
        @param etree: The (input) XML tree corresponding to `theory`.
        @param theory: The Serif theory object, to which the attribute
            should be added.
        """
        setattr(theory, self.__name__, self.get_value(etree, theory))

    def get_value(self, etree, theory):
        """
        Extract and return the value of this attribute from an input
        XML tree.

        @param name: The name that should be used to store the attribute.
        @param etree: The (input) XML tree corresponding to `theory`.
        @param theory: The Serif theory object, to which the attribute
            should be added.
        """
        raise AssertionError('get_value() is an abstract method.')

    def serialize(self, etree, theory, **options):
        raise AssertionError('serialize() is an abstract method.')

    def default_value(self):
        return None

    def help(self):
        """
        Return a single-line string describing this attribute
        """
        raise AssertionError('help() is an abstract method.')


class _SimpleAttribute(_AutoPopulatedXMLAttributeSpec):
    """
    A basic serif theory attribute, whose value is copied directly
    from a corresonding XML attribute.  The value should have a simple
    type (such as string, boolean, or integer).
    """

    def __init__(self, value_type=str, default=None, attr_name=None,
                 is_required=False):
        """
        @param value_type: The type of value expected for this attribute.
            This should be a Python type (such as int or bool), and is
            used directly to cast the string value to an appropriate value.
        @param default: The default value for this attribute.  I.e., if
            no value is provided, then the attribute will take this value.
            The default value is *not* required to be of the type specified
            by value_type -- in particular, the default may be None.
        @param attr_name: The name of the XML attribute used to store this
            value.  If not specified, then the name will default to the
            name of the serif theory attribute.
        @param is_required: If true, then raise an exception if this
            attribute is not defined on the XML input element.
        """
        _AutoPopulatedXMLAttributeSpec.__init__(self)
        self._value_type = value_type
        self._default = default
        self._attr_name = attr_name
        self._is_required = is_required

    def get_value(self, etree, theory):
        name = self._attr_name or self.__name__
        if name in etree.attrib:
            return self._parse_value(name, etree.attrib[name])
        elif self._is_required:
            raise ValueError('Attribute %s is required for %s' %
                             (name, etree))
        else:
            return self._default

    def _parse_value(self, name, value):
        if self._value_type == bool:
            if value.lower() == 'true': return True
            if value.lower() == 'false': return False
            raise ValueError('Attribute %s must have a boolean value '
                             '(either TRUE or FALSE)' % name)
        else:
            return self._value_type(value)

    def _encode_value(self, value):
        if value is True:
            return 'TRUE'
        elif value is False:
            return 'FALSE'
        elif isinstance(value, bytes):
            return value.decode('utf-8')
        elif isinstance(value, EnumeratedType._BaseClass):
            return value.value
        elif not isinstance(value, str):
            return str(value)
        else:
            return value

    def serialize(self, etree, theory, **options):
        value = getattr(theory, self.__name__, None)
        explicit_defaults = options.get('explicit_defaults', False)
        if value is not None:
            if ((not explicit_defaults) and
                    (self._default is not None) and
                    (value == self._default)):
                return
            attr_name = self._attr_name or self.__name__
            value = self._encode_value(value)
            etree.attrib[attr_name] = value

    _HELP_TEMPLATE = 'a %s value extracted from the XML attribute %r'

    def help(self):
        name = self._attr_name or self.__name__
        s = self._HELP_TEMPLATE % (
            self._value_type.__name__, name)
        if self._is_required:
            s += ' (required)'
        else:
            s += ' (default=%r)' % self._default
        return s


class _SimpleListAttribute(_SimpleAttribute):
    def _parse_value(self, name, value):
        return tuple(_SimpleAttribute._parse_value(self, name, v)
                     for v in value.split())

    def _encode_value(self, value):
        return ' '.join(_SimpleAttribute._encode_value(self, v)
                        for v in value)

    _HELP_TEMPLATE = 'a list of %s values extracted from the XML attribute %r'


class _IdAttribute(_AutoPopulatedXMLAttributeSpec):
    """
    An identifier attribute (copied from the XML attribute \"id\").
    In addtion to initializing theory.id, this attribute also
    registers the id in the identifier map that is owned by the
    theory's document.
    """

    def set_value(self, etree, theory):
        theory.id = etree.attrib.get('id')
        document = theory.document
        if document is None:
            raise ValueError('Containing document not found!')
        document.register_id(theory)

    def serialize(self, etree, theory, **options):
        xml_id = getattr(theory, 'id', None)
        if xml_id is not None:
            etree.attrib['id'] = xml_id

    def help(self):
        return "The XML id for this theory object (default=None)"


class _ReferenceAttribute(_SimpleAttribute):
    """
    An attribute that is used to point to another Serif theory object,
    using its identifier.  When this attribute is initialized, the
    target id is copied from the XML attribute with a specified name
    (`attr_name`), and stored as a private variable.  This id is *not*
    looked up during initialization, since its target may not have
    been created yet.

    Instead, this attribute uses a Python feature called
    \"descriptors\" to resolve the target id to a value when the
    attribute is accessed.

    In particular, each _ReferencedAttribute is a (non-data)
    descriptor on the Serif theory class, which means that its
    `__get__()` method is called whenever the corresponding Serif
    theory attribute is read.  The `__get__()` method looks up the
    target id in the identifier map that is owned by the theory's
    document.  If the identifier is found, then the corresponding
    theory object is returned; otherwise, a special `DanglingPointer`
    object is returned.
    """

    def __init__(self, attr_name, is_required=False, cls=None):
        """
        @param attr_name: The name of the XML idref attribute used to
            hold the pointer to a theory object.  Typically, these
            attribute names will end in '_id'.
        @param is_required: If true, then raise an exception if this
            attribute is not defined on the XML input element.  If
            is_required is false and the attribute is not defined on
            the XML input element, then the Serif theory attribute's
            value will be None.
        @param cls: The Serif theory class (or name of the class)
            that the target value should belong to.
        """
        self._attr_name = attr_name
        self._private_attr_name = '_' + attr_name
        self._cls = cls
        _SimpleAttribute.__init__(self, is_required=is_required,
                                  attr_name=attr_name)

    def set_value(self, etree, theory):
        # This stores the id, but does *not* look it up -- the target
        # for the pointer might not have been deserialized from xml yet.
        setattr(theory, self._private_attr_name,
                self.get_value(etree, theory))

    def serialize(self, etree, theory, **options):
        child = getattr(theory, self.__name__, None)
        if child is not None:
            etree.attrib[self._attr_name] = self._get_child_id(child)

    def _get_child_id(self, child):
        child_id = getattr(child, 'id', None)
        if child_id is None:
            raise ValueError('Serialization Error: attempt to serialize '
                             'a pointer to an object that has no id (%r)'
                             % child)
        return child_id

    def __get__(self, instance, owner=None):
        # We look up the id only when the attribute is accessed.
        if instance is None: return self
        theory_id = getattr(instance, self._private_attr_name)
        if theory_id is None: return None
        document = instance.document
        if document is None:
            return DanglingPointer(theory_id)
        target = document.lookup_id(theory_id)
        if target is None:
            return DanglingPointer(theory_id)
        if self._cls is not None:
            if isinstance(self._cls, str):
                self._cls = SerifTheory._theory_classes[self._cls]
            if not isinstance(target, self._cls):
                raise ValueError('Expected %s to point to a %s' % (
                    self._attr_name, self._cls.__name__))
        return target

    def _cls_name(self):
        if self._cls is None:
            return 'theory object'
        elif isinstance(self._cls, str):
            return self._cls
        else:
            return self._cls.__name__

    def help(self):
        name = self._attr_name or self.__name__
        s = 'a pointer to a %s extracted from the XML attribute %r' % (
            self._cls_name(), name)
        if self._is_required: s += ' (required)'
        return s


class _ReferenceListAttribute(_ReferenceAttribute):
    """
    An attribute that is used to point to a sequence of Serif theory
    objects, using their identifiers.  This AttributeSpec is similar
    to `_ReferenceAttribute`, except that its value is a list of
    theory objects, rather than a single theory object.
    """

    def __get__(self, instance, owner):
        theory_ids = getattr(instance, self._private_attr_name)
        theory_ids = (theory_ids or '').split()
        document = instance.document
        if document is None:
            return [DanglingPointer(tid) for tid in theory_ids]
        targets = [(document.lookup_id(tid) or DanglingPointer(tid))
                   for tid in theory_ids]
        if self._cls is not None:
            if isinstance(self._cls, str):
                self._cls = SerifTheory._theory_classes[self._cls]
            for t in targets:
                if not isinstance(t, (self._cls, DanglingPointer)):
                    raise ValueError('Expected %s to point to a %s; got a %s' % (
                        self._attr_name, self._cls.__name__, t.__class__.__name__))
        return targets

    def serialize(self, etree, theory, **options):
        child_ids = [self._get_child_id(child)
                     for child in getattr(theory, self.__name__, ())]
        if child_ids:
            etree.attrib[self._attr_name] = ' '.join(child_ids)

    def default_value(self):
        return []

    def help(self):
        name = self._attr_name or self.__name__
        s = ('a list of pointers to %ss extracted from '
             'the XML attribute %r' % (self._cls_name(), name))
        return s


class DanglingPointer(object):
    """
    A class used by `_ReferenceAttribute` to indicate that the target
    id has not yet been read.  In particular, a DanglingPointer will
    be returned by `ReferenceAttribute.__get__()` if a target pointer
    id is not found in the identifier map.
    """

    def __init__(self, id):
        self.id = id

    def __repr__(self):
        return "<Dangling Pointer: id=%r>" % self.id

    def _get_summary(self):
        return "<Dangling Pointer: id=%r>" % self.id


class _OffsetAttribute(_AutoPopulatedXMLAttributeSpec):
    """
    An attribute used to store a start or end offset.  These
    attributes may be stored in the XML in two different ways: either
    using separate XML attributes for the begin and end offsets; or
    using a single XML attribute for both.  This AttributeSpec
    subclass is responsible for reading both formats.
    """

    def __init__(self, offset_side, offset_name, value_type=int):
        _AutoPopulatedXMLAttributeSpec.__init__(self)
        assert offset_side in ('start', 'end')
        self.is_start = (offset_side == 'start')
        self.offset_name = offset_name
        self.offset_attr = '%s_%s' % (offset_side, offset_name)
        self.condensed_offsets_attr = '%s_offsets' % offset_name
        self._value_type = value_type

    def get_value(self, etree, theory):
        if self.offset_attr in etree.attrib:
            return self._value_type(etree.attrib[self.offset_attr])
        elif self.condensed_offsets_attr in etree.attrib:
            s, e = etree.attrib[self.condensed_offsets_attr].split(':')
            if self.is_start:
                return self._value_type(s)
            else:
                return self._value_type(e)
        else:
            return None

    def serialize(self, etree, theory, **options):
        value = getattr(theory, self.__name__, None)
        if value is not None:
            if options.get('condensed_offsets', True):
                etree.attrib[self.condensed_offsets_attr] = '%s:%s' % (
                    getattr(theory, 'start_%s' % self.offset_name),
                    getattr(theory, 'end_%s' % self.offset_name))
            else:
                etree.attrib[self.offset_attr] = '%s' % value

    def help(self):
        return 'an offset extracted from XML attribute %r or %r' % (
            (self.offset_attr, self.condensed_offsets_attr))


class _ChildTheoryElement(_AutoPopulatedXMLAttributeSpec):
    """
    An attribute used to hold a child theory that is described in
    a child XML element.
    """

    def __init__(self, cls_name, is_required=False):
        """
        @param cls_name: The name of the Serif theory class for the
            child value.
        """
        _AutoPopulatedXMLAttributeSpec.__init__(self)
        self._is_required = is_required
        self._cls_name = cls_name

    def _get_child_elt(self, name, etree):
        if isinstance(name, tuple):
            elts = [elt for elt in etree if elt.tag in name]
            name = ' or '.join(name)  # for error messages.
        else:
            elts = [elt for elt in etree if elt.tag == name]
        if len(elts) == 1:
            return elts[0]
        elif len(elts) > 1:
            raise ValueError('Expected at most one %s' % name)
        elif self._is_required:
            raise ValueError('Expected exactly one %s' % name)
        else:
            return None

    def serialize(self, etree, theory, **options):
        child = getattr(theory, self.__name__, None)
        if child is not None:
            if (hasattr(child, '_etree') and child._etree in etree):
                child_etree = child.toxml(child._etree, **options)
            else:
                child_etree = child.toxml(**options)
                etree.append(child_etree)
            if isinstance(self._cls_name, tuple):
                assert child_etree.tag in self._cls_name
            else:
                assert child_etree.tag == self._cls_name

    def get_value(self, etree, theory):
        name = self._cls_name or self.__name__
        child_elt = self._get_child_elt(name, etree)
        if child_elt is None:
            return None
        cls = SerifTheory._theory_classes.get(child_elt.tag)
        if cls is None:
            raise AssertionError('Theory class %s not defined!' % name)
        return cls(child_elt, theory)

    def help(self):
        s = 'a child %s theory' % self._cls_name
        if self._is_required:
            s += ' (required)'
        else:
            s += ' (optional)'
        return s


class _ChildTextElement(_ChildTheoryElement):
    """
    An attribute whose value should be extracted from the string text
    of a child XML element.  (c.f. _TextOfElement)
    """

    def get_value(self, etree, theory):
        child_elt = self._get_child_elt(self._cls_name, etree)
        if KEEP_ORIGINAL_ETREE:
            self._child_elt = child_elt
        if child_elt is None:
            return None
        else:
            return child_elt.text

    def serialize(self, etree, theory, **options):
        text = getattr(theory, self.__name__, None)
        if text is not None:
            if hasattr(self, '_child_elt') and self._child_elt in etree:
                child_etree = self._child_elt
            else:
                del etree[:]
                child_etree = ET.Element(self._cls_name or self.__name__)
                etree.append(child_etree)
            child_etree.text = text
            child_etree.tail = '\n' + options.get('indent', '')

    def help(self):
        return 'a text string extracted from the XML element %r' % (
            self._cls_name)


class _TextOfElement(_AutoPopulatedXMLAttributeSpec):
    """
    An attribute whose value should be extracted from the string text
    of *this* XML element.  (c.f. _ChildTextElement)
    """

    def __init__(self, is_required=False, strip=False):
        _AutoPopulatedXMLAttributeSpec.__init__(self)
        self._strip = strip
        self._is_required = is_required

    def get_value(self, etree, theory):
        text = etree.text or ''
        if self._strip: text = text.strip()
        if self._is_required and not text:
            raise ValueError('Text content is required for %s' %
                             self.__name__)
        return text

    def serialize(self, etree, theory, **options):
        text = getattr(theory, self.__name__, None)
        if text is not None:
            # assert etree.text is None # only one text string!
            etree.text = text

    def help(self):
        return ("a text string extracted from this "
                "theory's XML element text")


class _ChildTheoryElementList(_AutoPopulatedXMLAttributeSpec):
    """
    An attribute whose value is a list of child theories.  Each child
    theory is deserialized from a single child XML element.
    """

    def __init__(self, cls_name, index_attrib=None):
        _AutoPopulatedXMLAttributeSpec.__init__(self)
        self._cls_name = cls_name
        self._index_attrib = index_attrib

    def get_value(self, etree, theory):
        name = self._cls_name or self.__name__
        elts = [elt for elt in etree if elt.tag == name]
        cls = SerifTheory._theory_classes.get(name)
        if cls is None:
            raise AssertionError('Theory class %s not defined!' % name)
        result = [cls(elt, theory) for elt in elts]
        if self._index_attrib:
            for i, child in enumerate(result):
                child.__dict__[self._index_attrib] = i
        return result

    def serialize(self, etree, theory, **options):
        children = getattr(theory, self.__name__, ())
        if KEEP_ORIGINAL_ETREE:
            child_etrees = set(etree)
        else:
            child_etrees = set()
        for child in children:
            if (hasattr(child, '_etree') and child._etree in child_etrees):
                child_etree = child.toxml(child._etree, **options)
            else:
                child_etree = child.toxml(**options)
                etree.append(child_etree)
            assert child_etree.tag == self._cls_name

    def default_value(self):
        return []

    def help(self):
        s = 'a list of child %s theory objects' % self._cls_name
        return s


######################################################################
# { Enumerated Type metaclass
######################################################################

class EnumeratedType(type):
    """
    >>> colors = EnumeratedType('colors', 'red green blue')
    >>> assert colors.red != colors.green
    >>> assert colors.red == colors.red
    """

    class _BaseClass(object):
        def __init__(self, value):
            self.__value = value
            self.__hash = hash(value)

        def __repr__(self):
            return '%s.%s' % (self.__class__.__name__, self.__value)

        def __hash__(self):
            return self.__hash

        def __eq__(self, other):
            if isinstance(other, self.__class__):
                return self.value == other.value
            return NotImplemented

        @property
        def value(self):
            return self.__value

    def __new__(cls, name, values):
        return type.__new__(cls, name, (cls._BaseClass,), {})

    def __init__(cls, name, values):
        if isinstance(values, str):
            values = values.split()
        cls.values = [cls(value) for value in values]
        for enum_name, enum_value in zip(values, cls.values):
            setattr(cls, enum_name, enum_value)

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, i):
        return self.values[i]

    def __repr__(self):
        return '<%s enumeration: %r>' % (self.__name__, tuple(self.values),)


######################################################################
# { Theory Objects Base Classes
######################################################################

# Define some enumerated types.
ParseType = EnumeratedType(
    'ParseType', 'full_parse np_chunk')
MentionType = EnumeratedType(
    'MentionType', 'none name pron desc part appo list nest')
PredType = EnumeratedType(
    'PredType', 'verb copula modifier noun poss loc set name pronoun comp')
Genericity = EnumeratedType(
    'Genericity', 'Specific Generic')
Polarity = EnumeratedType(
    'Polarity', 'Positive Negative')
DirectionOfChange = EnumeratedType(
    'DirectionOfChange', 'Unspecified Increase Decrease')
Tense = EnumeratedType(
    'Tense', 'Unspecified Past Present Future')
Modality = EnumeratedType(
    'Modality', 'Asserted Other')
PropStatus = EnumeratedType(
    'PropStatus', 'Default If Future Negative Alleged Modal Unreliable')


class metaSerifTheory(type):
    def __init__(cls, name, bases, dct):
        type.__init__(cls, name, bases, dct)
        # Register the class in a registry.
        cls.__theory_name__ = name
        if hasattr(cls, '__overrides__'):
            cls.__theory_name__ = cls.__overrides__
        # elif name in cls._theory_classes:
        #    print "Warning: overriding %s!" % name
        cls._theory_classes[cls.__theory_name__] = cls

        # Add an _auto_attribs attribute
        cls._auto_attribs = [
            (k, v) for (k, v) in list(dct.items())
            if isinstance(v, _AutoPopulatedXMLAttributeSpec)]
        for attr_name, attr_spec in cls._auto_attribs:
            attr_spec.__name__ = attr_name
        for base in bases:
            cls._auto_attribs.extend(getattr(base, '_auto_attribs', []))

        def sort_key(attrib):
            return (attrib[1]._attribute_number, attrib[0].lower())

        cls._auto_attribs.sort(key=sort_key)


class SerifTheory(object, metaclass=metaSerifTheory):
    """
    The base class for serif theory types.
    """
    _theory_classes = {}

    # Every theory object may take an id.
    id = _IdAttribute()

    _OWNER_IS_REQUIRED = True

    def __init__(self, etree=None, owner=None, **attribs):
        # Set our owner pointer.
        if owner is not None:
            self._owner = weakref.ref(owner)
        elif self._OWNER_IS_REQUIRED:
            raise ValueError('%s constructor requires an owner' %
                             self.__class__.__name__)
        else:
            self._owner = None
        # Intialize, either from etree or from attributes.
        if etree is not None:
            if attribs:
                raise ValueError('Specify etree or attribs, not both!')
            self._init_from_etree(etree, owner)
        else:
            for attr_name, attr_spec in self._auto_attribs:
                value = attribs.pop(attr_name, None)
                if value is not None:
                    setattr(self, attr_name, value)
                else:
                    setattr(self, attr_name, attr_spec.default_value())

    def _init_from_etree(self, etree, owner):
        assert etree is not None
        if etree.tag != self.__class__.__theory_name__:
            raise ValueError('Expected a %s, got a %s!' %
                             (self.__class__.__theory_name__, etree.tag))
        if KEEP_ORIGINAL_ETREE:
            self._etree = etree
        # Fill in any attribute values
        for name, attr in self._auto_attribs:
            attr.set_value(etree, self)

    def toxml(self, etree=None, **options):
        """
        If `etree` is specified, then this theory object will be
        serialized into that element tree Element; otherwise, a new
        Element will be created.
        """
        # print 'serializing %s' % self.__class__.__theory_name__
        indent = options.get('indent')
        if indent is not None: options['indent'] += '  '

        if etree is None:
            etree = ET.Element(self.__class__.__theory_name__)
        else:
            assert etree.tag == self.__class__.__theory_name__, (
                etree.tag, self.__class__.__theory_name__)

        for name, attr in self._auto_attribs:
            attr.serialize(etree, self, **options)

        # Indentation...
        if len(etree) > 0 and indent is not None:
            etree.text = '\n' + indent + '  '
            for child in etree[:-1]:
                child.tail = '\n' + indent + '  '
            etree[-1].tail = '\n' + indent
        if indent is not None: options['indent'] = indent
        etree.tail = '\n'
        return etree

    def pprint(self, depth=-1, hide=(), follow_pointers=False,
               indent='  ', memo=None):
        """
        Return a pretty-printed string representation of this SERIF
        theory object.  The first line identifies this theory object,
        and the subsequent lines describe its contents (including
        nested or referenced theory objects).

        @param depth: The maximum depth to which nested theory objects
            should be displayed.
        @param hide: A set of names of attributes that should not
            be displayed.  (By default, the XML id and the EDT and
            byte offsets are not displayed by __str__).
        @param follow_pointers: If true, then attributes that contain
            pointers have their contents displayed just like nested
            elements.  If false, then the pointer targets are not
            expanded.
        """
        if memo is None: memo = set()
        if id(self) in memo:
            return '<%s...>' % self.__class__.__theory_name__
        memo.add(id(self))
        s = self._pprint_firstline(indent)
        for attr_name, attr_spec in self.__class__._auto_attribs:
            if attr_name in hide: continue
            val = getattr(self, attr_name)
            if attr_name == '_children':
                attr_name = ''
            elif attr_name.startswith('_'):
                continue
            attr_depth = depth
            if (not follow_pointers and val is not None and
                    isinstance(attr_spec, _ReferenceAttribute)
                    and not isinstance(val, DanglingPointer)):
                s += '\n%s%s = <%s...>' % (
                    indent, attr_name, getattr(val.__class__, '__theory_name__',
                                               val.__class__.__name__))
            else:
                s += '\n' + self._pprint_value(attr_name, val, attr_depth, hide,
                                               follow_pointers, indent, memo)
        return s

    def _get_summary(self):
        return None

    def _pprint_firstline(self, indent):
        s = self.__class__.__theory_name__ + ':'
        text = self._get_summary()
        if text:
            maxlen = max(9, 65 - len(indent) -
                         len(self.__class__.__theory_name__) * 2)
            s += ' %s' % _truncate(text, maxlen)
        return s

    def _pprint_value(self, attr, val, depth, hide,
                      follow_pointers, indent, memo):
        s = indent
        if attr: s += attr + ' = '
        if isinstance(val, SerifTheory):
            if depth is not None and depth == 0:
                return s + '<%s...>' % getattr(val.__class__, '__theory_name__',
                                               val.__class__.__name__)
            return s + val.pprint(depth - 1, hide, follow_pointers,
                                  indent + '  ', memo)
        elif isinstance(val, list):
            if len(val) == 0: return s + '[]'
            if depth is not None and depth == 0: return s + '[...]'
            items = [self._pprint_value('', item, depth - 1, hide,
                                        follow_pointers, indent + '  ', memo)
                     for item in val]
            if depth == 1 and len(items) > 12:
                items = items[:10] + ['%s  ...and %d more...' %
                                      (indent, len(items) - 10)]
            s += '[\n%s\n%s]' % ('\n'.join(items), indent)
            return s
        elif isinstance(val, str):
            text = repr(val)
            maxlen = max(9, 75 - len(s))
            if len(text) > maxlen:
                text = text[:maxlen - 9] + '...' + text[-6:]
            return s + text
        else:
            return s + repr(val)

    _default_hidden_attrs = set(['id', 'start_byte', 'end_byte',
                                 'start_edt', 'end_edt'])

    def __repr__(self):
        text = self._get_summary()
        if text:
            return '<%s %s>' % (self.__class__.__theory_name__, text)
        else:
            return '<%s>' % self.__class__.__theory_name__

    def __str__(self):
        return self.pprint(depth=2, hide=self._default_hidden_attrs,
                           follow_pointers=False)

    @property
    def owner(self):
        """The theory object that owns this SerifTheory"""
        if self._owner is None:
            return None
        else:
            return self._owner()

    def owner_with_type(self, theory_class):
        """
        Find and return the closest owning theory with the given
        class.  If none is found, return None.  E.g., use
        tok.owner(Sentence) to find the sentence containing a token.
        """
        if isinstance(theory_class, str):
            theory_class = SerifTheory._theory_classes[theory_class]
        theory = self
        while theory is not None and not isinstance(theory, theory_class):
            if theory._owner is None: return None
            theory = theory._owner()
        return theory

    @property
    def document(self):
        """The document that contains this SerifTheory"""
        return self.owner_with_type(Document)

    def get_original_text_substring(self, start_char, end_char):
        """
        Return the original text substring spanning from start_char to
        end_char.  If this theory is contained in a sentence, then the
        string is taken from that sentence's contents string (if defined);
        otherwise, it is taken from the document's original text string.
        """
        theory = self
        while theory is not None:
            if isinstance(theory, OriginalText):
                s = theory.start_char
                return theory.contents[start_char - s:end_char - s + 1]
            elif isinstance(theory, Document):
                theory = theory.original_text
            else:
                theory = theory.owner
        return None

    def resolve_pointers(self, fail_on_dangling_pointer=True):
        """
        Replace reference attributes with their actual values for this
        theory and any theory owned by this theory (directly or
        indirectly).  Prior to calling this, every time you access a
        reference attribute, its value will be looked up in the
        document's identifier map.

        @param fail_on_dangling_pointer: If true, then raise an exception
        if we find a dangling pointer.
        """
        for attr_name, attr_spec in self._auto_attribs:
            attr_val = getattr(self, attr_name)
            # Replace any reference attribute w/ its actual value (unless
            # it's a dangling pointer)
            if isinstance(attr_spec, _ReferenceAttribute):
                if attr_name not in self.__dict__:
                    if not isinstance(attr_val, DanglingPointer):
                        setattr(self, attr_name, attr_val)
                    elif fail_on_dangling_pointer:
                        raise ValueError('Dangling pointer: %r' % attr_val)

            # Recurse to any owned objects.
            elif isinstance(attr_val, SerifTheory):
                attr_val.resolve_pointers(fail_on_dangling_pointer)

    @classmethod
    def _help_header(cls):
        return 'The %r class defines the following attributes:' % (
            cls.__theory_name__)

    @classmethod
    def help(cls):
        props = [(k, v) for base in cls.mro()
                 for (k, v) in list(base.__dict__.items())
                 if isinstance(v, property)]

        s = cls._help_header() + '\n'
        w = max([8] + [len(n) for (n, c) in cls._auto_attribs] +
                [len(n) for (n, p) in props]) + 2
        for attr_name, attr_spec in cls._auto_attribs:
            if attr_name == '_children': continue
            help_line = textwrap.fill(attr_spec.help(),
                                      initial_indent=' ' * (w + 3),
                                      subsequent_indent=' ' * (w + 3)).strip()
            s += '  %s %s\n' % (attr_name.ljust(w, '.'), help_line)
        if props:
            s += ('The following derived properties are also '
                  'available as attributes:\n')
            for (k, v) in props:
                help_text = v.__doc__ or '(undocumented)'
                help_text = help_text.replace(
                    'this SerifTheory', 'this ' + cls.__theory_name__)
                help_text = ' '.join(help_text.split())
                help_line = textwrap.fill(
                    help_text,
                    initial_indent=' ' * (w + 3),
                    subsequent_indent=' ' * (w + 3)).strip()
                s += '  %s %s\n' % (k.ljust(w, '.'), help_line)
        # s += '  %s %s\n' % ('owner'.ljust(w, '.'),
        #                     'The theory object that owns this %s' %
        #                     cls.__theory_name__)
        # s += '  %s %s\n' % ('document'.ljust(w, '.'),
        #                     'The Document that contains this %s' %
        #                     cls.__theory_name__)
        print(s.rstrip())


def _truncate(text, maxlen):
    if text is None:
        return None
    elif len(text) <= maxlen:
        return text
    else:
        return text[:maxlen - 9] + '...' + text[-6:]


class SerifDocumentTheory(SerifTheory):
    def __init__(self, etree=None, owner=None, **attribs):
        self._idmap = weakref.WeakValueDictionary()
        SerifTheory.__init__(self, etree, owner, **attribs)

    _OWNER_IS_REQUIRED = False

    def _init_from_etree(self, etree, owner):
        # If the argument isn't an etree, then create one.
        if hasattr(etree, 'makeelement'):
            pass  # ok.
        elif hasattr(etree, 'getroot'):
            etree = etree.getroot()  # ElementTree object
        elif isinstance(etree, str):
            if re.match('^\s*<', etree):
                etree = ET.fromstring(etree)  # xml string
            elif '\n' not in etree:
                etree = ET.parse(etree).getroot()  # filename
            else:
                raise ValueError('Expected a filename, xml string, stream, '
                                 'or ElementTree.  Got a %s' %
                                 etree.__class__.__name__)
        elif hasattr(etree, 'read'):
            etree = ET.fromstring(etree.read())  # file object
        else:
            raise ValueError('Expected a filename, xml string, stream, '
                             'or ElementTree.  Got a %s' %
                             etree.__class__.__name__)
        # If we got a SerifXML element, then take its document.
        if (etree.tag == 'SerifXML' and len(etree) == 1 and
                etree[0].tag == 'Document'):
            etree = etree[0]
        SerifTheory._init_from_etree(self, etree, owner)
        # Resolve pointers.
        self.resolve_pointers()

    def save(self, file_or_filename):
        serifxml_etree = ET.Element('SerifXML')
        serifxml_etree.text = '\n  '
        serifxml_etree.attrib['version'] = str(SERIFXML_VERSION)
        etree = getattr(self, '_etree', None)
        serifxml_etree.append(self.toxml(etree, indent='  '))
        ET.ElementTree(serifxml_etree).write(file_or_filename)

    def register_id(self, theory):
        if theory.id is not None:
            if theory.id in self._idmap:
                raise ValueError('Duplicate id %s' % theory.id)
            self._idmap[theory.id] = theory

    def lookup_id(self, theory_id):
        return self._idmap.get(theory_id)

    _default_hidden_attrs = set(['lexicon'])


class SerifSequenceTheory(SerifTheory):
    _children = "This class attr must be defined by subclasses."

    def __len__(self):
        return len(self._children)

    def __iter__(self):
        return self._children.__iter__()

    def __contains__(self, item):
        return self._children.__contains__(item)

    def __getitem__(self, n):
        return self._children.__getitem__(n)

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__theory_name__, self._children)

    def resolve_pointers(self, fail_on_dangling_pointer=True):
        SerifTheory.resolve_pointers(self, fail_on_dangling_pointer)
        for child in self._children:
            child.resolve_pointers(fail_on_dangling_pointer)

    @classmethod
    def _help_header(cls):
        child_class_name = cls._children._cls_name
        return textwrap.fill(
            'The %r class acts as a sequence of %r elements.  '
            'Additionally, it defines the following attributes:'
            % (cls.__theory_name__, child_class_name))


class SerifOffsetTheory(SerifTheory):
    """Base class for theory objects that have attributes"""
    start_byte = _OffsetAttribute('start', 'byte')
    end_byte = _OffsetAttribute('end', 'byte')
    start_char = _OffsetAttribute('start', 'char')
    end_char = _OffsetAttribute('end', 'char')
    start_edt = _OffsetAttribute('start', 'edt')
    end_edt = _OffsetAttribute('end', 'edt')

    def _get_summary(self):
        text = self.text
        if text is None:
            return None
        else:
            return repr(text)

    @property
    def text(self):
        """The original text substring covered by this theory"""
        return self.get_original_text_substring(self.start_char, self.end_char)


class SerifSentenceTheory(SerifOffsetTheory):
    def save_text(self):
        if self.contents is None:
            self.contents = self.text

    @property
    def sent_no(self):
        """
        The index of this sentence in the 'Sentences' that owns it, or
        None if this sentence is not owned by a 'Sentences'.
        """
        if isinstance(self.owner, Sentences):
            return self.owner._children.index(self)

    def _get_parse(self):
        if len(self.parses) == 0:
            return None
        elif len(self.parses) == 1:
            return self.parses[0]
        else:
            _raise_expected_exactly_one_error(
                'Sentence', 'parse', 'parses',
                len(self.parses))

    def _set_parse(self, value):
        if value is None:
            self.parses = []
        else:
            self.parses = [value, ]

    parse = property(_get_parse, _set_parse, doc="""
        The unique parse for this sentence, or None if the sentence has
        no parse.  If the sentence has multiple candidate parses (because
        it has multiple sentence theories), then this will raise an
        exception.""")

    def _get_sentence_theory(self):
        if len(self.sentence_theories) == 0:
            return None
        elif len(self.sentence_theories) == 1:
            return self.sentence_theories[0]
        else:
            _raise_expected_exactly_one_error(
                'Sentence', 'sentence_theory', 'sentence_theories',
                len(self.sentence_theories))

    def _set_sentence_theory(self, value):
        if value is None:
            self.sentence_theories = []
        else:
            self.sentence_theories = [value, ]

    sentence_theory = property(_get_sentence_theory,
                               _set_sentence_theory, doc="""
        The unique sentence_theory for this sentence, or None if the
        sentence has no sentence_theory.  If the sentence has multiple
        candidate sentence_theories, then this will raise an
        exception.""")


class SerifMentionTheory(SerifTheory):
    @property
    def child_mention_list(self):
        """A list of the children of this mention.  This list contains
        child_mention (if defined), plus the chain of next_mentions
        starting at child_mention.
        """
        if self.child_mention is None: return []
        child_mentions = [self.child_mention]
        while child_mentions[-1].next_mention is not None:
            child_mentions.append(child_mentions[-1].next_mention)
        return child_mentions

    @property
    def text(self):
        """The text content of this mention."""
        if self.syn_node is None:
            return None
        else:
            return self.syn_node.text

    @property
    def tokens(self):
        """The tokens contained in this mention's SynNode."""
        if self.syn_node is None:
            return None
        else:
            return self.syn_node.tokens

    @property
    def start_char(self):
        """The start character index of this Mention's SynNode"""
        return self.syn_node.start_token.start_char

    @property
    def end_char(self):
        """The start character index of this Mention's SynNode"""
        return self.syn_node.end_token.end_char

    def _get_summary(self):
        if self.syn_node is None:
            return None
        else:
            return self.syn_node._get_summary()

    @property
    def sent_no(self):
        return self.syn_node.sent_no

    _head = None

    @property
    def head(self):
        """ Implementation borrowed from APF4GenericResultCollector::_getEDTHead() """
        if self._head is None:
            self._head = self._find_head()
        return self._head

    def _find_head(self):
        mention = self
        if mention.mention_type == MentionType.name:
            while mention.child_mention is not None:
                mention = mention.child_mention
            return mention.syn_node
        if mention.mention_type == MentionType.appo:
            return mention.child_mention.head
        node = mention.syn_node
        if not node.is_preterminal:
            node = node.head
            while (not node.is_preterminal) and (node.mention is None):
                node = node.head
        if node.is_preterminal:
            return node
        return node.mention.head

    @property
    def atomic_head(self):
        """ Implementation borrowed from Mention::getAtomicHead """
        if self.mention_type == MentionType.name:
            return self.syn_node.head_preterminal.parent
        else:
            return self.syn_node.head_preterminal

    @property
    def premod_text(self):
        e_min = self.syn_node.start_token.start_char
        h_min = self.head.start_token.start_char
        if h_min > e_min:
            return self.get_original_text_substring(e_min, h_min - 1)
        else:
            return ''

    @property
    def postmod_text(self):
        h_max = self.head.end_token.end_char
        e_max = self.syn_node.end_token.end_char
        if e_max > h_max:
            return self.get_original_text_substring(h_max + 1, e_max)
        else:
            return ''


class SerifValueMentionTheory(SerifOffsetTheory):
    @property
    def tokens(self):
        """The list of tokens covered by this ValueMention"""
        tok_seq = list(self.owner_with_type(Sentence).token_sequence)
        s = tok_seq.index(self.start_token)
        e = tok_seq.index(self.end_token)
        return tok_seq[s:e + 1]

    @property
    def sentence(self):
        if self.sent_no:
            return self.sent_no
        else:
            return self.owner.owner.sent_no


class SerifTokenTheory(SerifOffsetTheory):
    def _init_from_etree(self, etree, owner):
        SerifOffsetTheory._init_from_etree(self, etree, owner)
        # If the text is not specified, it defaults to the original
        # text substring.
        if self.text is None:
            self.text = self.original_text_substring

    @property
    def syn_node(self):
        """The terminal SynNode corresponding to this Token, or None
           if the parse is not available."""
        # Cache the value in self._syn_node, since it's nontrivial to
        # look it up.
        if not hasattr(self, '_syn_node'):
            if self.owner is None: return None
            sent = self.owner_with_type(Sentence)
            if sent is None: return None
            parse = sent.parse
            if parse is None: return None
            self._syn_node = self._find_syn_node(parse.root)
        return self._syn_node

    def _find_syn_node(self, syn_node):
        if len(syn_node) == 0:
            assert syn_node.start_token == syn_node.end_token == self
            return syn_node
        elif len(syn_node) == 1:
            return self._find_syn_node(syn_node[0])
        else:
            for child in syn_node:
                if ((child.start_char <= self.start_char) and
                        (self.end_char <= child.end_char)):
                    return self._find_syn_node(child)
            return None  # should never happen

    def __repr__(self):
        return '<%s: %r>' % (self.__class__.__theory_name__, self.text)


class SerifParseTheory(SerifTheory):
    def _init_from_etree(self, etree, owner):
        SerifTheory._init_from_etree(self, etree, owner)
        if self.root is None:
            if self._treebank_string is None:
                raise ValueError('Parse requires SynNode or TreebankString')
            else:
                self._parse_treebank_string()
        assert self.root is not None or len(self.token_sequence) == 0

    indent = 0

    def toxml(self, etree=None, **options):
        indent = options.get('indent')
        if indent is not None: options['indent'] += '  '

        if etree is None:
            etree = ET.Element(self.__class__.__theory_name__)
        else:
            assert etree.tag == self.__class__.__theory_name__

        parse_format = options.get('parse_format', 'treebank').lower()
        # Add all attributes; but skip _treebank_string, and also skip
        # _root if we're serializing the tree as a treebank string.
        for name, attr in self._auto_attribs:
            if name == '_treebank_string': continue
            if name == 'root' and parse_format == 'treebank': continue
            attr.serialize(etree, self, **options)
        # Serialize the tree as a treebank string (if requested)
        if parse_format == 'treebank':
            treebank_str = ET.Element('TreebankString')
            treebank_str.attrib['node_id_method'] = 'DFS'
            if self.root is None:
                treebank_str.text = '(X^ -empty-)'
            else:
                treebank_str.text = self.root._treebank_str()
            del etree[:]  # discard old treebank string.
            etree.append(treebank_str)
        # [xx] should there be an else clause here?

        # Indentation...
        if len(etree) > 0 and indent is not None:
            etree.text = '\n' + indent + '  '
            for child in etree[:-1]:
                child.tail = '\n' + indent + '  '
            etree[-1].tail = '\n' + indent
        if indent is not None: options['indent'] = indent
        etree.tail = '\n'
        return etree

    # to do -- handle is_head!
    _TB_TOKEN = re.compile('|'.join([
        '\((?P<start>[^\s\^]+)(?P<headmark>\^?)', '(?P<end>\))',
        '\s+', '(?P<token>[^\(\)\s]+)']))

    def _parse_treebank_string(self):
        token_sequence = self.token_sequence
        # Special case: if there's no token sequence, then we can't
        # define a parse (since the parse points into the token seq)
        if len(token_sequence) == 0:
            if (self._treebank_string != '(X^ -empty-)' and
                    self._treebank_string != '(FRAG^)'):
                print(('Warning: discarding treebank string %r because '
                       'no token sequence is defined' % self._treebank_string))
            self.root = None
            del self._treebank_string
            return

        token_index = 0
        stack = [self]
        dfs_count = 0
        for piece in self._TB_TOKEN.finditer(self._treebank_string):
            if piece.group('start') or piece.group('token'):
                if token_index >= len(token_sequence):
                    # print `self._treebank_string`
                    # print `token_sequence`
                    raise ValueError(
                        "Number of terminals in parse string is "
                        "greater than the number of tokens in the "
                        "token sequence (%d)" % len(token_sequence))
                tag = piece.group('start') or piece.group('token')
                syn_node = SynNode(tag=tag, owner=stack[-1])
                syn_node.start_token = token_sequence[token_index]
                syn_node.end_token = None  # filled in later.
                syn_node.is_head = piece.group('headmark') == '^'
                syn_node.id = '%s.%s' % (self.id, dfs_count)
                dfs_count += 1
                self.document.register_id(syn_node)
                if piece.group('start'):
                    stack.append(syn_node)
                else:
                    syn_node.end_token = self.token_sequence[token_index]
                    syn_node.is_head = True
                    token_index += 1
                    stack[-1]._children.append(syn_node)
            elif piece.group('end'):
                assert len(stack) > 1
                assert token_index > 0
                completed = stack.pop()
                completed.end_token = self.token_sequence[token_index - 1]
                if stack[-1] is self:
                    assert self.root is None
                    self.root = completed
                else:
                    stack[-1]._children.append(completed)
        assert token_index == len(self.token_sequence), (
            self.token_sequence, token_index, self._treebank_string)
        assert len(stack) == 1
        self._treebank_string = None

    @property
    def sent_no(self):
        if self.owner:
            return self.owner.sent_no
        else:
            return None


class SerifSynNodeTheory(SerifSequenceTheory):
    @property
    def parent(self):
        """The parent syntactic node (or None for the root node)"""
        owner = self.owner
        if isinstance(owner, SynNode):
            return owner
        else:
            return None

    @property
    def parse(self):
        """The Parse object that contains this SynNode"""
        owner = self.owner
        if isinstance(owner, SynNode):
            return owner.parse
        else:
            return owner

    @property
    def sent_no(self):
        return self.parse.sent_no

    @property
    def right_siblings(self):
        """ A list of siblings to the right of this node, if any exist"""
        if self.parent:
            siblings = [x for x in self.parent]
            index = siblings.index(self)
            return siblings[index + 1:]
        else:
            return []

    @property
    def tokens(self):
        """The list of tokens covered by this SynNode"""
        tok_seq = list(self.parse.token_sequence)
        s = tok_seq.index(self.start_token)
        e = tok_seq.index(self.end_token)
        return tok_seq[s:e + 1]

    @property
    def is_terminal(self):
        """Is this SynNode a terminal?"""
        return len(self) == 0

    @property
    def is_preterminal(self):
        """Is this SynNode a preterminal?"""
        return len(self) == 1 and len(self[0]) == 0

    @property
    def preterminals(self):
        """The list of preterminal SynNode descendents of this SynNode"""
        if len(self) == 0:
            raise ValueError('Can not get preterminals of a terminal')
        if len(self) == 1 and len(self[0]) == 0:
            return [self]
        else:
            return sum((c.preterminals for c in self), [])

    @property
    def terminals(self):
        """The list of terminal SynNode descendents of this SynNode"""
        if len(self) == 0:
            return [self]
        else:
            return sum((c.terminals for c in self), [])

    @property
    def text(self):
        """The original text substring covered by this SynNode"""
        return self.get_original_text_substring(
            self.start_token.start_char, self.end_token.end_char)

    def _get_summary(self):
        return repr(self.text)

    @property
    def head(self):
        """The head child of this SynNode"""
        for child in self:
            if child.is_head: return child
        # Report an error if we didn't find a head?
        return None

    @property
    def headword(self):
        """The text of the head terminal of this SynNode"""
        head_terminal = self.head_terminal
        if head_terminal is None: return None
        return head_terminal.text

    @property
    def head_terminal(self):
        """The head terminal of this SynNode"""
        if len(self) == 0:
            return self
        else:
            return self.head.head_terminal

    @property
    def head_preterminal(self):
        """The head pre-terminal of this SynNode"""
        if len(self) == 0:
            raise ValueError('Can not get preterminal of a terminal')
        elif len(self) == 1 and len(self[0]) == 0:
            return self
        else:
            return self.head.head_preterminal

    @property
    def head_tag(self):
        """The tag of the head terminal of this SynNode"""
        if len(self) == 0:
            return self.tag
        else:
            return self.head.head_tag

    @property
    def start_char(self):
        """The start character index of this SynNode's start token"""
        return self.start_token.start_char

    @property
    def end_char(self):
        """The end character index of this SynNode's end token"""
        return self.end_token.end_char

    @property
    def mention(self):
        """The mention corresponding to this SynNode, or None if there
           is no such mention."""
        # Cache the value in self._mention, since it's nontrivial to
        # look it up.
        if not hasattr(self, '_mention'):
            parse = self.parse
            if parse is None: return None
            sent = parse.owner
            if sent is None: return None
            mention_set = sent.mention_set
            if mention_set is None: return None
            for mention in mention_set:
                if mention.syn_node == self:
                    self._mention = mention
                    break
            else:
                self._mention = None
        return self._mention

    @property
    def value_mention(self):
        """The value mention corresponding to this SynNode, or None if
           there is no such value mention."""
        # Cache the value in self._value_mention, since it's nontrivial to
        # look it up.
        if not hasattr(self, '_value_mention'):
            parse = self.parse
            if parse is None: return None
            sent = parse.owner
            if sent is None: return None
            value_mention_set = sent.value_mention_set
            if value_mention_set is None: return None
            for value_mention in value_mention_set:
                if (value_mention.start_token == self.start_token and
                        value_mention.end_token == self.end_token):
                    self._value_mention = value_mention
                    break
            else:
                self._value_mention = None
        return self._value_mention

    def _treebank_str(self, depth=-1):
        if len(self) == 0: return self.tag
        s = '(%s' % self.tag
        if self.is_head: s += '^'
        if depth == 0 and len(self) > 0:
            s += ' ...'
        else:
            s += ''.join(' %s' % child._treebank_str(depth - 1)
                         for child in self)
        return s + ')'

    def _pprint_treebank_str(self, indent='', width=75):
        # Try putting this tree on one line.
        self_repr = self.__repr__()
        if len(self_repr) + len(indent) < width:
            return "%s%s" % (indent, self_repr)
        # Otherwise, put each child on a separate line.
        s = '%s(%s' % (indent, self.tag)
        if self.is_head: s += '^'
        for child in self:
            s += '\n' + child._pprint_treebank_str(indent + '  ', width)
        return s + ')'

    def pprint(self, depth=-1, hide=(), follow_pointers=True,
               indent='  ', memo=None):
        return self._treebank_str(depth)

    def __repr__(self):
        s = self._treebank_str()
        return s

    def __str__(self):
        return self._pprint_treebank_str()

    def remove_child(self, child):
        """
        Remove a child from this SynNode.  The child's owner will be
        set to None.
        """
        if child not in self._children:
            raise ValueError('remove_child(c): c not a child')
        self._children.remove(child)
        child.owner = None
        return child

    def gorn_address(self):
        """
        Returns a string containing the Gorn address of this node.
        http://en.wikipedia.org/wiki/Gorn_address
        """
        indices = []
        self._gorn_address(indices)
        # _gorn_address will put the node indices in the list in
        # backwards order
        indices.reverse()
        return ".".join(str(idx) for idx in indices)

    def _gorn_address(self, indexList):
        if self.parent is None:
            indexList.append(0)
        else:
            indexList.append(self.parent._children.index(self))
            self.parent._gorn_address(indexList)


class ICEWSEventParticipantTheory(SerifTheory):
    @property
    def text(self):
        """The original text substring covered by this event participant"""
        return self.actor.text

    @property
    def mention(self):
        """Shortcut for self.actor.mention"""
        return self.actor.mention

    @property
    def sentence_theory(self):
        """Shortcut for self.actor.sentence_theory"""
        return self.actor.sentence_theory

    def _get_summary(self):
        return repr(self.text)


class ActorMentionTheory(SerifTheory):
    @property
    def text(self):
        """The original text substring covered by this event participant"""
        return self.mention.text

    def _get_summary(self):
        return repr(self.text)


def _raise_expected_exactly_one_error(cls, single, multiple, n):
    raise ValueError(
        'The %(cls)s.%(single)s property can only be used if '
        'len(%(cls)s.%(multiple)s)==1.  For this instance, '
        'len(%(cls)s.%(multiple)s)==%(n)s.' % dict(
            cls=cls, single=single, multiple=multiple, n=n))


######################################################################
# { Theory Classes
######################################################################

class Document(SerifDocumentTheory):
    docid = _SimpleAttribute(is_required=True)
    language = _SimpleAttribute(is_required=True)
    source_type = _SimpleAttribute(default='UNKNOWN')
    is_downcased = _SimpleAttribute(bool, default=False)
    document_time_start = _SimpleAttribute()
    document_time_end = _SimpleAttribute()
    original_text = _ChildTheoryElement('OriginalText')
    date_time = _ChildTheoryElement('DateTime')
    regions = _ChildTheoryElement('Regions')
    segments = _ChildTheoryElement('Segments')
    metadata = _ChildTheoryElement('Metadata')
    sentences = _ChildTheoryElement('Sentences')
    entity_set = _ChildTheoryElement('EntitySet')
    value_set = _ChildTheoryElement('ValueSet')
    relation_set = _ChildTheoryElement('RelationSet')
    event_set = _ChildTheoryElement('EventSet')
    value_mention_set = _ChildTheoryElement('ValueMentionSet')
    utcoref = _ChildTheoryElement('UTCoref')
    rel_mention_set = _ChildTheoryElement('RelMentionSet')
    lexicon = _ChildTheoryElement('Lexicon')
    actor_entity_set = _ChildTheoryElement('ActorEntitySet')
    actor_mention_set = _ChildTheoryElement('ActorMentionSet')
    document_actor_info = _ChildTheoryElement('DocumentActorInfo')
    fact_set = _ChildTheoryElement('FactSet')
    icews_actor_mention_set = _ChildTheoryElement('ICEWSActorMentionSet')
    icews_event_mention_set = _ChildTheoryElement('ICEWSEventMentionSet')
    flexible_event_mention_set = _ChildTheoryElement('FlexibleEventMentionSet')
    event_event_relation_mention_set = _ChildTheoryElement('EventEventRelationMentionSet')

class OriginalText(SerifOffsetTheory):
    contents = _ChildTextElement('Contents')
    href = _SimpleAttribute()


class Regions(SerifSequenceTheory):
    _children = _ChildTheoryElementList('Region')


class Region(SerifOffsetTheory):
    contents = _ChildTextElement('Contents')
    tag = _SimpleAttribute()  # is_required=True)
    is_speaker = _SimpleAttribute(bool, default=False)
    is_receiver = _SimpleAttribute(bool, default=False)


class DateTime(SerifOffsetTheory):
    contents = _ChildTextElement('Contents')


class Metadata(SerifSequenceTheory):
    _children = _ChildTheoryElementList('Span')


class Span(SerifOffsetTheory):
    span_type = _SimpleAttribute(is_required=True)
    region_type = _SimpleAttribute()
    idf_type = _SimpleAttribute()
    idf_role = _SimpleAttribute()
    original_sentence_index = _SimpleAttribute(int)  # For ICEWS_Sentence


class Sentences(SerifSequenceTheory):
    _children = _ChildTheoryElementList('Sentence', index_attrib='sent_no')


class Sentence(SerifSentenceTheory):
    region = _ReferenceAttribute('region_id', cls=Region,
                                 is_required=True)
    is_annotated = _SimpleAttribute(bool, default=True)
    primary_parse = _SimpleAttribute(ParseType,
                                     default=ParseType.full_parse)
    contents = _ChildTextElement('Contents')
    token_sequence = _ChildTheoryElement('TokenSequence')
    pos_sequence = _ChildTheoryElement('PartOfSpeechSequence')
    name_theory = _ChildTheoryElement('NameTheory')
    nested_name_theory = _ChildTheoryElement('NestedNameTheory')
    value_mention_set = _ChildTheoryElement('ValueMentionSet')
    np_chunk_theory = _ChildTheoryElement('NPChunkTheory')
    parses = _ChildTheoryElementList('Parse')
    mention_set = _ChildTheoryElement('MentionSet')
    proposition_set = _ChildTheoryElement('PropositionSet')
    rel_mention_set = _ChildTheoryElement('RelMentionSet')
    event_mention_set = _ChildTheoryElement('EventMentionSet')
    actor_mention_set = _ChildTheoryElement('ActorMentionSet')
    icews_actor_mention_set = _ChildTheoryElement('ICEWSActorMentionSet')
    sentence_theories = _ChildTheoryElementList('SentenceTheory')
    # n.b.: "parse" and "sentence_theory" are defined as properties.


class SentenceTheory(SerifTheory):
    actor_mention_set = _ReferenceAttribute('actor_mention_set_id',
                                            cls='ActorMentionSet')
    token_sequence = _ReferenceAttribute('token_sequence_id',
                                         cls='TokenSequence')
    pos_sequence = _ReferenceAttribute('part_of_speech_sequence_id',
                                       cls='PartOfSpeechSequence')
    name_theory = _ReferenceAttribute('name_theory_id',
                                      cls='NameTheory')
    value_mention_set = _ReferenceAttribute('value_mention_set_id',
                                            cls='ValueMentionSet')
    np_chunk_theory = _ReferenceAttribute('np_chunk_theory_id',
                                          cls='NPChunkTheory')
    parse = _ReferenceAttribute('parse_id',
                                cls='Parse')
    mention_set = _ReferenceAttribute('mention_set_id',
                                      cls='MentionSet')
    proposition_set = _ReferenceAttribute('proposition_set_id',
                                          cls='PropositionSet')
    rel_mention_set = _ReferenceAttribute('rel_mention_set_id',
                                          cls='RelMentionSet')
    event_mention_set = _ReferenceAttribute('event_mention_set_id',
                                            cls='EventMentionSet')
    # If the following lines are uncommented,
    # then the 'primary_parse' field is added 
    # to the output. -DJE
    primary_parse = _SimpleAttribute(ParseType,
                                     default=ParseType.full_parse)


class TokenSequence(SerifSequenceTheory):
    score = _SimpleAttribute(float)
    _children = _ChildTheoryElementList('Token')


class Token(SerifTokenTheory):
    # note: default value for text is extracted from the original string.
    text = _TextOfElement(strip=True)
    lexical_entries = _ReferenceListAttribute('lexical_entries',
                                              cls='LexicalEntry')
    original_token_index = _SimpleAttribute(int)


class PartOfSpeechSequence(SerifSequenceTheory):
    score = _SimpleAttribute(float)
    token_sequence = _ReferenceAttribute('token_sequence_id',
                                         cls=TokenSequence)
    _children = _ChildTheoryElementList('POS')


class POS(SerifTheory):
    tag = _SimpleAttribute(is_required=False)
    prob = _SimpleAttribute(float, default=1.0)
    token = _ReferenceAttribute('token_id', cls=Token)
    alternate_pos_tags = _ChildTheoryElementList('AlternatePOS')


class AlternatePOS(SerifTheory):
    tag = _SimpleAttribute(is_required=True)
    prob = _SimpleAttribute(float)


class NameTheory(SerifSequenceTheory):
    score = _SimpleAttribute(float)
    token_sequence = _ReferenceAttribute('token_sequence_id',
                                         cls=TokenSequence)
    _children = _ChildTheoryElementList('Name')


class Name(SerifOffsetTheory):
    entity_type = _SimpleAttribute(is_required=True)
    start_token = _ReferenceAttribute('start_token', cls=Token,
                                      is_required=True)
    end_token = _ReferenceAttribute('end_token', cls=Token,
                                    is_required=True)
    transliteration = _SimpleAttribute(is_required=False)
    score = _SimpleAttribute(float, is_required=False)


class NestedNameTheory(NameTheory):
    name_theory = _ReferenceAttribute('name_theory_id',
                                      cls=NameTheory)
    _children = _ChildTheoryElementList('NestedName')


class NestedName(Name):
    parent = _ReferenceAttribute('parent', cls=Name,
                                 is_required=True)


class ValueMentionSet(SerifSequenceTheory):
    score = _SimpleAttribute(float)
    token_sequence = _ReferenceAttribute('token_sequence_id',
                                         cls=TokenSequence)
    _children = _ChildTheoryElementList('ValueMention')


class ValueMention(SerifValueMentionTheory):
    value_type = _SimpleAttribute(is_required=True)
    start_token = _ReferenceAttribute('start_token', cls=Token,
                                      is_required=True)
    end_token = _ReferenceAttribute('end_token', cls=Token,
                                    is_required=True)
    sent_no = _SimpleAttribute(int, default=None)


class NPChunkTheory(SerifSequenceTheory):
    score = _SimpleAttribute(float)
    token_sequence = _ReferenceAttribute('token_sequence_id',
                                         cls=TokenSequence)
    _children = _ChildTheoryElementList('NPChunk')
    _parse = _ChildTheoryElement('Parse')


class NPChunk(SerifOffsetTheory):
    start_token = _ReferenceAttribute('start_token', cls=Token,
                                      is_required=True)
    end_token = _ReferenceAttribute('end_token', cls=Token,
                                    is_required=True)


class Parse(SerifParseTheory):
    score = _SimpleAttribute(float)
    token_sequence = _ReferenceAttribute('token_sequence_id',
                                         cls=TokenSequence)
    root = _ChildTheoryElement('SynNode')
    _treebank_string = _ChildTextElement('TreebankString')


class SynNode(SerifSynNodeTheory):
    tag = _SimpleAttribute(is_required=True)
    start_token = _ReferenceAttribute('start_token', cls=Token,
                                      is_required=True)
    end_token = _ReferenceAttribute('end_token', cls=Token,
                                    is_required=True)
    is_head = _SimpleAttribute(bool, default=False)
    _children = _ChildTheoryElementList('SynNode')


class MentionSet(SerifSequenceTheory):
    name_score = _SimpleAttribute(float)
    desc_score = _SimpleAttribute(float)
    parse = _ReferenceAttribute('parse_id', cls=Parse)
    _children = _ChildTheoryElementList('Mention')


class Mention(SerifMentionTheory):
    syn_node = _ReferenceAttribute('syn_node_id', cls=SynNode,
                                   is_required=True)
    mention_type = _SimpleAttribute(MentionType, is_required=True)
    entity_type = _SimpleAttribute(is_required=True)
    entity_subtype = _SimpleAttribute(default='UNDET')
    is_metonymy = _SimpleAttribute(bool, default=False)
    intended_type = _SimpleAttribute(default='UNDET')
    role_type = _SimpleAttribute(default='UNDET')
    link_confidence = _SimpleAttribute(float, default=1.0)
    confidence = _SimpleAttribute(float, default=1.0)
    parent_mention = _ReferenceAttribute('parent', cls='Mention')
    child_mention = _ReferenceAttribute('child', cls='Mention')
    next_mention = _ReferenceAttribute('next', cls='Mention')


class PropositionSet(SerifSequenceTheory):
    mention_set = _ReferenceAttribute('mention_set_id', cls=MentionSet)
    _children = _ChildTheoryElementList('Proposition')


class Proposition(SerifTheory):
    arguments = _ChildTheoryElementList('Argument')
    pred_type = _SimpleAttribute(PredType, attr_name='type',
                                 is_required=True)
    head = _ReferenceAttribute('head_id', cls=SynNode)
    particle = _ReferenceAttribute('particle_id', cls=SynNode)
    adverb = _ReferenceAttribute('adverb_id', cls=SynNode)
    negation = _ReferenceAttribute('negation_id', cls=SynNode)
    modal = _ReferenceAttribute('modal_id', cls=SynNode)
    statuses = _SimpleListAttribute(PropStatus, attr_name='status')

    def _get_summary(self):
        if self.head:
            pred = '%s:%s' % (self.pred_type.value,
                              self.head._get_summary())
        else:
            pred = self.pred_type.value
        return '%s(%s)' % (pred,
                           ', '.join(arg._get_summary()
                                     for arg in self.arguments))


class Argument(SerifTheory):
    role = _SimpleAttribute(default='')
    mention = _ReferenceAttribute('mention_id', cls=Mention)
    syn_node = _ReferenceAttribute('syn_node_id', cls=SynNode)
    proposition = _ReferenceAttribute('proposition_id', cls=Proposition)

    value = property(
        lambda self: self.mention or self.syn_node or self.proposition)

    def _get_summary(self):
        return '%s=%r' % (self.role or '<val>', self.value)


class RelMentionSet(SerifSequenceTheory):
    score = _SimpleAttribute(float)
    _children = _ChildTheoryElementList('RelMention')


class RelMention(SerifTheory):
    score = _SimpleAttribute(float, default=1.0)
    confidence = _SimpleAttribute(float, default=1.0)
    type = _SimpleAttribute(is_required=True)
    raw_type = _SimpleAttribute()
    tense = _SimpleAttribute(Tense, is_required=True)
    modality = _SimpleAttribute(Modality, is_required=True)
    left_mention = _ReferenceAttribute('left_mention_id', cls=Mention)
    right_mention = _ReferenceAttribute('right_mention_id', cls=Mention)
    time_arg = _ReferenceAttribute('time_arg_id', cls=ValueMention)
    time_arg_role = _SimpleAttribute()
    time_arg_score = _SimpleAttribute(float, default=0.0)


class EventMentionSet(SerifSequenceTheory):
    score = _SimpleAttribute(float)
    _children = _ChildTheoryElementList('EventMention')


class EventMention(SerifTheory):
    arguments = _ChildTheoryElementList('EventMentionArg')
    score = _SimpleAttribute(float, default=1.0)
    event_type = _SimpleAttribute(is_required=True)
    pattern_id = _SimpleAttribute(is_required=False)
    genericity = _SimpleAttribute(Genericity, is_required=True)
    polarity = _SimpleAttribute(Polarity, is_required=True)
    direction_of_change = _SimpleAttribute(DirectionOfChange, is_required=False)
    tense = _SimpleAttribute(Tense, is_required=True)
    modality = _SimpleAttribute(Modality, is_required=True)
    anchor_prop = _ReferenceAttribute('anchor_prop_id',
                                      cls=Proposition)
    anchor_node = _ReferenceAttribute('anchor_node_id',
                                      cls=SynNode)


class EventMentionArg(SerifOffsetTheory):
    role = _SimpleAttribute(default='')
    mention = _ReferenceAttribute('mention_id',
                                  cls=Mention)
    value_mention = _ReferenceAttribute('value_mention_id',
                                        cls=ValueMention)
    score = _SimpleAttribute(float, default=0.0)
    value = property(
        lambda self: self.mention or self.value_mention)


class EntitySet(SerifSequenceTheory):
    score = _SimpleAttribute(float)
    _children = _ChildTheoryElementList('Entity')


class Entity(SerifTheory):
    mentions = _ReferenceListAttribute('mention_ids', cls=Mention)
    entity_type = _SimpleAttribute(is_required=True)
    entity_subtype = _SimpleAttribute(default='UNDET')
    is_generic = _SimpleAttribute(bool, is_required=True)
    canonical_name = _SimpleAttribute()
    entity_guid = _SimpleAttribute()
    confidence = _SimpleAttribute(float, default=1.0)
    mention_confidences = _SimpleAttribute()


class ValueSet(SerifSequenceTheory):
    _children = _ChildTheoryElementList('Value')


class Value(SerifTheory):
    # todo: should value_mention_ref be renamed to value_mention_id???
    value_mention = _ReferenceAttribute('value_mention_ref',
                                        cls=ValueMention)
    value_type = _SimpleAttribute(attr_name='type', is_required=True)
    timex_val = _SimpleAttribute()
    timex_anchor_val = _SimpleAttribute()
    timex_anchor_dir = _SimpleAttribute()
    timex_set = _SimpleAttribute()
    timex_mod = _SimpleAttribute()
    timex_non_specific = _SimpleAttribute()

    specific_year_re = re.compile('^([12][0-9][0-9][0-9])$')
    specific_sub_year_re = re.compile('^([12][0-9][0-9][0-9])-.*')

    def is_specific_date(self):
        if self.timex_val:
            return (Value.specific_year_re.match(self.timex_val)
                    or Value.specific_sub_year_re.match(self.timex_val))
        else:
            return False


class RelationSet(SerifSequenceTheory):
    _children = _ChildTheoryElementList('Relation')


class Relation(SerifTheory):
    rel_mentions = _ReferenceListAttribute('rel_mention_ids',
                                           cls=RelMention)
    relation_type = _SimpleAttribute(is_required=True, attr_name='type')
    left_entity = _ReferenceAttribute('left_entity_id', cls=Entity,
                                      is_required=True)
    right_entity = _ReferenceAttribute('right_entity_id', cls=Entity,
                                       is_required=True)
    tense = _SimpleAttribute(Tense, is_required=True)
    modality = _SimpleAttribute(Modality, is_required=True)
    confidence = _SimpleAttribute(float, default=1.0)


class EventSet(SerifSequenceTheory):
    _children = _ChildTheoryElementList('Event')


class Event(SerifTheory):
    arguments = _ChildTheoryElementList('EventArg')
    event_type = _SimpleAttribute(is_required=True)
    event_mentions = _ReferenceListAttribute('event_mention_ids',
                                             cls=EventMention)
    genericity = _SimpleAttribute(Genericity, is_required=True)
    polarity = _SimpleAttribute(Polarity, is_required=True)
    tense = _SimpleAttribute(Tense, is_required=True)
    modality = _SimpleAttribute(Modality, is_required=True)
    annotation_id = _SimpleAttribute()


class EventArg(SerifTheory):
    role = _SimpleAttribute(default='')
    entity = _ReferenceAttribute('entity_id',
                                 cls=Entity)
    value_entity = _ReferenceAttribute('value_id',
                                       cls=Value)
    score = _SimpleAttribute(float, default=0.0)
    value = property(
        lambda self: self.entity or self.value_entity)


class UTCoref(SerifTheory):
    """currently not documented"""


class Segments(SerifSequenceTheory):
    _children = _ChildTheoryElementList('Segment')


class Segment(SerifTheory):
    attributes = _ChildTheoryElementList('Attribute')
    fields = _ChildTheoryElementList('Field')


class Attribute(SerifTheory):
    key = _SimpleAttribute(is_required=True)
    val = _SimpleAttribute(is_required=True)


class Field(SerifTheory):
    name = _SimpleAttribute(is_required=True)
    entries = _ChildTheoryElementList('Entry')


class Entry(SerifSequenceTheory):
    _children = _ChildTheoryElementList('Attribute')
    contents = _ChildTextElement('Contents')


class Lexicon(SerifSequenceTheory):
    _children = _ChildTheoryElementList('LexicalEntry')


class LexicalEntry(SerifTheory):
    key = _SimpleAttribute()
    category = _SimpleAttribute()
    voweled_string = _SimpleAttribute()
    pos = _SimpleAttribute()
    gloss = _SimpleAttribute()
    analysis = _ReferenceListAttribute('analysis',
                                       cls='LexicalEntry')


class ActorMentionSet(SerifSequenceTheory):
    _children = _ChildTheoryElementList('ActorMention')


class ICEWSActorMentionSet(SerifSequenceTheory):
    _children = _ChildTheoryElementList('ActorMention')


class ActorMention(ActorMentionTheory):
    mention = _ReferenceAttribute('mention_id', cls=Mention)
    sentence_theory = _ReferenceAttribute('sentence_theory_id',
                                          cls=SentenceTheory)
    source_note = _SimpleAttribute()
    # For Proper Name Actor Mentions:
    actor_db_name = _SimpleAttribute()
    actor_uid = _SimpleAttribute(int)
    actor_code = _SimpleAttribute()
    actor_pattern_uid = _SimpleAttribute(int)
    is_acronym = _SimpleAttribute(bool)
    requires_context = _SimpleAttribute(bool)
    actor_name = _SimpleAttribute()

    # For Composite Actor Mentions:
    paired_actor_uid = _SimpleAttribute(int)
    paired_actor_code = _SimpleAttribute()
    paired_actor_pattern_uid = _SimpleAttribute(int)
    paired_actor_name = _SimpleAttribute()
    paired_agent_uid = _SimpleAttribute(int)
    paired_agent_code = _SimpleAttribute()
    paired_agent_pattern_uid = _SimpleAttribute(int)
    paired_agent_name = _SimpleAttribute()
    actor_agent_pattern = _SimpleAttribute()

    # For locations resolved to the icews DB
    geo_country = _SimpleAttribute()
    geo_latitude = _SimpleAttribute()
    geo_longitude = _SimpleAttribute()
    geo_uid = _SimpleAttribute()
    geo_text = _SimpleAttribute()

    # Country info
    country_id = _SimpleAttribute(int)
    iso_code = _SimpleAttribute()
    country_info_actor_id = _SimpleAttribute(int)
    country_info_actor_code = _SimpleAttribute()

    # Scores - don't necessarily exist
    pattern_match_score = _SimpleAttribute(float)
    association_score = _SimpleAttribute(float)
    edit_distance_score = _SimpleAttribute(float)
    georesolution_score = _SimpleAttribute(float)
    confidence = _SimpleAttribute(float)

    # Note: this attribute will be empty unless the
    # "icews_include_actor_names_in_serifxml" parameter is enabled.
    name = _TextOfElement(strip=True)


class ActorEntitySet(SerifSequenceTheory):
    _children = _ChildTheoryElementList('ActorEntity')


class ActorEntity(SerifTheory):
    entity = _ReferenceAttribute('entity_id', cls=Entity)
    actor_uid = _SimpleAttribute(int)
    actor_mentions = _ReferenceListAttribute('actor_mention_ids', cls=ActorMention)
    confidence = _SimpleAttribute(float)
    name = _TextOfElement(strip=True)
    actor_name = _SimpleAttribute()


class DocumentActorInfo(SerifTheory):
    default_country_actor = _ChildTheoryElement('DefaultCountryActor')


class DefaultCountryActor(SerifTheory):
    actor_db_name = _SimpleAttribute()
    actor_uid = _SimpleAttribute(int)


class FactSet(SerifSequenceTheory):
    _children = _ChildTheoryElementList('Fact')


class Fact(SerifTheory):
    start_sentence = _SimpleAttribute(int)
    end_sentence = _SimpleAttribute(int)
    start_token = _SimpleAttribute(int)
    end_token = _SimpleAttribute(int)
    fact_type = _SimpleAttribute()
    pattern_id = _SimpleAttribute()
    score_group = _SimpleAttribute(int)
    score = _SimpleAttribute(float)

    mention_fact_arguments = _ChildTheoryElementList('MentionFactArgument')
    value_mention_fact_arguments = _ChildTheoryElementList('ValueMentionFactArgument')
    text_span_fact_arguments = _ChildTheoryElementList('TextSpanFactArgument')
    string_fact_arguments = _ChildTheoryElementList('StringFactArgument')


class FactArgument(SerifTheory):
    role = _SimpleAttribute()


class MentionFactArgument(FactArgument):
    mention = _ReferenceAttribute('mention_id',
                                  cls=Mention)


class ValueMentionFactArgument(FactArgument):
    is_doc_date = _SimpleAttribute(bool, default=False)
    value_mention = _ReferenceAttribute('value_mention_id',
                                        cls=ValueMention)


class TextSpanFactArgument(FactArgument):
    start_sentence = _SimpleAttribute(int)
    end_sentence = _SimpleAttribute(int)
    start_token = _SimpleAttribute(int)
    end_token = _SimpleAttribute(int)


class StringFactArgument(FactArgument):
    string = _SimpleAttribute()


class ICEWSEventMentionSet(SerifSequenceTheory):
    _children = _ChildTheoryElementList('ICEWSEventMention')


class ICEWSEventMention(SerifTheory):
    participants = _ChildTheoryElementList('ICEWSEventParticipant')
    event_code = _SimpleAttribute(is_required=True)
    event_tense = _SimpleAttribute(is_required=True)
    pattern_id = _SimpleAttribute(is_required=True)
    time_value_mention = _ReferenceAttribute('time_value_mention_id',
                                             cls=ValueMention,
                                             is_required=False)
    propositions = _ReferenceListAttribute('proposition_ids', cls=Proposition)
    original_event_id = _SimpleAttribute(is_required=False)
    is_reciprocal = _SimpleAttribute(bool, is_required=False)


class ICEWSEventParticipant(ICEWSEventParticipantTheory):
    role = _SimpleAttribute(is_required=True)
    actor = _ReferenceAttribute('actor_id', cls=ActorMention)

class EventEventRelationMentionSet(SerifSequenceTheory):
    _children           = _ChildTheoryElementList('EventEventRelationMention')

class EventEventRelationMention(SerifTheory):
    pattern             = _SimpleAttribute()
    relation_type       = _SimpleAttribute()
    confidence          = _SimpleAttribute(float)
    model               = _SimpleAttribute()

    event_mention_relation_arguments        = _ChildTheoryElementList('EventMentionRelationArgument')
    icews_event_mention_relation_arguments  = _ChildTheoryElementList('ICEWSEventMentionRelationArgument')

class RelationArgument(SerifTheory):
    role                   = _SimpleAttribute()

class EventMentionRelationArgument(RelationArgument):
    event_mention          = _ReferenceAttribute('event_mention_id', 
                                                 cls=EventMention)
 
class ICEWSEventMentionRelationArgument(RelationArgument):
    icews_event_mention     = _ReferenceAttribute('icews_event_mention_id', 
                                                  cls=ICEWSEventMention)

class FlexibleEventMentionSet(SerifSequenceTheory):
    _children = _ChildTheoryElementList('FlexibleEventMention')

class FlexibleEventMention(SerifTheory):
    args = _ChildTheoryElementList('FlexibleEventMentionArg')
    event_type = _SimpleAttribute(is_required=True)
    modality = _SimpleAttribute(is_required=False, attr_name='Modality')
    number = _SimpleAttribute(is_required=False, attr_name='Number')
    population = _SimpleAttribute(is_required=False, attr_name='Population')
    population1 = _SimpleAttribute(is_required=False, attr_name='Population1')
    population2 = _SimpleAttribute(is_required=False, attr_name='Population2')
    reason = _SimpleAttribute(is_required=False, attr_name='Reason')
    violence = _SimpleAttribute(is_required=False, attr_name='Violence')


class FlexibleEventMentionArg(SerifTheory):
    role = _SimpleAttribute(is_required=True)
    start_sentence = _SimpleAttribute(int, is_required=False)
    end_sentence = _SimpleAttribute(int, is_required=False)
    start_token = _SimpleAttribute(int, is_required=False)
    end_token = _SimpleAttribute(int, is_required=False)
    mention = _ReferenceAttribute('mention_id', cls=Mention, is_required=False)
    syn_node = _ReferenceAttribute('syn_node_id', cls=SynNode, is_required=False)
    value_mention = _ReferenceAttribute('value_mention_id', cls=ValueMention, is_required=False)
    geo_uid = _SimpleAttribute()
    geo_country = _SimpleAttribute()
    temporal_resolution = _ChildTheoryElement('Timex2')


class Timex2(SerifTheory):
    val = _SimpleAttribute(is_required=True)
    mod = _SimpleAttribute()
    set = _SimpleAttribute(bool)
    granularity = _SimpleAttribute()
    periodicity = _SimpleAttribute()
    anchor_val = _SimpleAttribute()
    anchor_dir = _SimpleAttribute()
    non_specific = _SimpleAttribute(bool)


######################################################################
# { Serif HTTP Server
######################################################################

HOSTNAME = 'localhost'
PORT = 8000

PROCESS_DOCUMENT_TEMPLATE = r'''
<SerifXMLRequest>
  <ProcessDocument end_stage="%(end_stage)s" output_format="serifxml" input_type="%(input_type)s" %(date_string)s>
    %(document)s
  </ProcessDocument>
</SerifXMLRequest>
'''

PATTERN_MATCH_DOCUMENT_TEMPLATE = r'''
<SerifXMLRequest>
  <PatternMatch pattern_set_name="%(pattern_set_name)s" slot1_start_offset="%(slot1_start_offset)d" slot2_start_offset="%(slot2_start_offset)d" slot3_start_offset="%(slot3_start_offset)d">
    %(document)s
    %(slot1)s
    %(slot1_weights)s
    %(slot2)s
    %(slot2_weights)s
    %(slot3)s
    %(slot3_weights)s
    %(equiv_names)s
  </PatternMatch>
</SerifXMLRequest>
'''

DOCUMENT_TEMPLATE = r'''
<Document language="%(language)s" docid="%(docid)s">
  <OriginalText><Contents>%(content)s</Contents></OriginalText>
</Document>
'''


def send_serifxml_pd_request(document, hostname=HOSTNAME, port=PORT,
                             end_stage='output', input_type='auto',
                             verbose=False, timeout=0, num_tries=1,
                             document_date=None):
    """
    Send a SerifXML request to process the given document to the
    specified server.  If successful, then return a `Document` object
    containing the processed document.  If unsuccessful, then raise an
    exception with the response message from the server.

    @param document: A string containing an XML <Document> element.
    @param hostname: The hostname of the Serif HTTP server.
    @param port: The port on which the Serif HTTP server is listening.
    @param end_stage: The end stage for processing.
    """
    date_string = ""
    if document_date:
        date_string = "document_date=\"" + document_date + "\""
    request = PROCESS_DOCUMENT_TEMPLATE % dict(
        document=document, end_stage=end_stage, input_type=input_type,
        date_string=date_string)
    response = send_serif_request(
        'POST SerifXMLRequest', request,
        verbose=verbose, hostname=hostname, port=port, timeout=timeout, num_tries=num_tries)
    if re.match('HTTP/.* 200 OK', response):
        body = response.split('\r\n\r\n', 1)[1]
        return Document(ET.fromstring(body))
    else:
        raise ValueError(response)


def send_serifxml_pm_request(document, hostname=HOSTNAME, port=PORT,
                             pattern_set_name='test_patterns', slot1='', slot2='', slot3='',
                             slot1_weights='', slot2_weights='', slot3_weights='',
                             slot1_start_offset=-1, slot2_start_offset=-1, slot3_start_offset=-1, equiv_names='',
                             verbose=False, timeout=0, num_tries=1):
    """
    Send a SerifXML request to pattern match against the given document
    to the specified server.  If successful, return the output of the
    pattern match.  If unsuccessful, then raise an exception with the
    response message from the server.

    @param document: A string containing an XML <Document> element.
    @param hostname: The hostname of the Serif HTTP server.
    @param port: The port on which the Serif HTTP server is listening.
    """
    request = PATTERN_MATCH_DOCUMENT_TEMPLATE % dict(document=document, pattern_set_name=pattern_set_name, slot1=slot1,
                                                     slot2=slot2, slot3=slot3,
                                                     slot1_weights=slot1_weights, slot2_weights=slot2_weights,
                                                     slot3_weights=slot3_weights,
                                                     slot1_start_offset=slot1_start_offset,
                                                     slot2_start_offset=slot2_start_offset,
                                                     slot3_start_offset=slot3_start_offset, equiv_names=equiv_names)
    response = send_serif_request(
        'POST SerifXMLRequest', request,
        verbose=verbose, hostname=hostname, port=int(port), timeout=timeout, num_tries=num_tries)
    if re.match('HTTP/.* 200 OK', response):
        body = response.split('\r\n\r\n', 1)[1]
        return body
    else:
        raise ValueError(response)


def escape_xml(s):
    s = s.replace('&', '&amp;')
    s = s.replace('<', '&lt;')
    s = s.replace('>', '&gt;')
    s = s.replace('\r', '&#xD;')
    return s


def send_serifxml_document(content, docid='anonymous', language='English',
                           hostname=HOSTNAME, port=PORT,
                           end_stage='output', input_type='auto', verbose=False,
                           timeout=0, num_tries=1, document_date=None):
    """
    Create a <Document> object from the given text content and docid,
    and use `send_serifxml_pd_request()` to send it to a Serif HTTP
    server.  If successful, then return a `Document` object containing
    the processed document.  If unsuccessful, then raise an exception
    with the response message from the server.

    @param content: The text content that should be processed by Serif.
    @param docid: The document identifier for the created <Document>.
    @param language: The language used by the document.
    @param hostname: The hostname of the Serif HTTP server.
    @param port: The port on which the Serif HTTP server is listening.
    @param end_stage: The end stage for processing.
    """
    xml_request = DOCUMENT_TEMPLATE % dict(
        docid=docid, content=escape_xml(content), language=language)
    return send_serifxml_pd_request(
        xml_request, end_stage=end_stage, input_type=input_type, verbose=verbose,
        hostname=hostname, port=port, timeout=timeout, num_tries=num_tries,
        document_date=document_date)


def send_serif_request(header, msg, hostname=HOSTNAME, port=PORT,
                       verbose=False, timeout=0, num_tries=1):
    """
    Send an HTTP request message to the serif server, and return
    the response message (as a string).
    """
    # Construct the HTTP request message.
    encoded_length = len(msg.encode('utf-8'))
    request = (header + ' HTTP/1.0\r\n' +
               'content-length: {}\r\n\r\n'.format(encoded_length) + msg)
    if verbose:
        DIV = '-' * 75
        print('%s\n%s%s' % (DIV, request.replace('\r\n', '\n'), DIV))

    for attempt in range(num_tries):
        if attempt > 0:
            print("send_serif_request() on attempt %d of %d" % (attempt + 1, num_tries))

        # Send the message.
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        s.connect((hostname, int(port)))
        s.sendall(request.encode('utf-8'))
        s.setblocking(0)

        # Read and return the response.
        result = ''
        inputs = [s]
        data_bytes = bytes()
        problem_encountered = False
        while inputs and not problem_encountered:
            if timeout > 0:
                rlist, _, xlist = select.select(inputs, [], inputs, timeout)
            else:
                rlist, _, xlist = select.select(inputs, [], inputs)
            for s in rlist:
                data = s.recv(4096)
                # print "send_serif_request() received data of length %d" % len(data)
                if data:
                    data_bytes += data
                else:
                    result = data_bytes.decode('utf-8')
                    inputs.remove(s)
                    s.close()
            for s in xlist:
                print("send_serif_request() handling exceptional condition for %s" % s.getpeername())
                problem_encountered = True
                inputs.remove(s)
                s.close()
            if rlist == [] and xlist == []:
                print("send_serif_request() timed out while waiting for input from the socket!")
                problem_encountered = True
                inputs.remove(s)
                s.close()
        if problem_encountered:
            if result.endswith('</SerifXML>') or result.endswith('<SerifXML/>'):
                print("send_serif_request() encountered a problem but the result looks valid.")
                break  # Assume this means we got the result correctly
            else:
                continue  # Didn't get something we recognize, try again if attempt < num_tries
        else:
            break  # Success!  Break out of the num_tries loop

    if verbose: print('%s\n%s' % (result, DIV))
    return result


def send_shutdown_request(hostname=HOSTNAME, port=PORT):
    """
    Send a shutdown request to the serif server.

    @param hostname: The hostname of the Serif HTTP server.
    @param port: The port on which the Serif HTTP server is listening.
    """
    print("Sending shutdown request to %s:%d" % (hostname, int(port)))
    send_serif_request('POST Shutdown', '', hostname, port)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-i", '--input_batch_file', dest='input_batch_file', help="Path to input batch file", default="")
    parser.add_option("-o", '--output_dir', dest='output_dir', help="Path to output directory", default="")
    opts, args = parser.parse_args()

    if len(args) == 2 and args[0] in ('process_batch_file', 'shutdown'):
        ip_address_regex = re.compile('[^:]+:\d+$')
        if ip_address_regex.match(args[1]):
            hostname, port = args[1].split(':')
        else:
            server_info_file = args[1]
            seconds_to_sleep = 5
            total_seconds_slept = 0
            if not os.path.exists(server_info_file):
                sys.stdout.write('Waiting for {} to exist'.format(server_info_file))
                sys.stdout.flush()
            while not os.path.exists(server_info_file):
                time.sleep(seconds_to_sleep)
                total_seconds_slept += seconds_to_sleep
                sys.stdout.write('.')
                sys.stdout.flush()
                if total_seconds_slept == 600:
                    raise RuntimeError("Timed out while waiting for {} to exist".format(server_info_file))
            hostname, port = open(server_info_file).read().strip().split(':')
    else:
        sys.exit(
            "Usage is %s process_batch_file|shutdown server_info_file|host:port (-i <input_batch_file> -o <output_dir>)" %
            sys.argv[0])

    if args[0] == 'process_batch_file':
        if opts.output_dir != "" and not os.path.exists(opts.output_dir):
            os.makedirs(opts.output_dir)  # Create our output dir if it doesn't exist
        lines = [x.strip() for x in open(opts.input_batch_file, 'r', encoding='utf-8').readlines()]
        for index, line in enumerate(lines):
            # Allow for an optional tab-delimited format: path\tdocument_date
            if len(line.split('\t')) == 2:
                input_file, document_date = line.split('\t')
            else:
                input_file = line
                document_date = None
            filename = os.path.basename(input_file)
            print("Processing file (%d of %d): %s" % (index, len(lines), filename))
            sys.stdout.flush()
            data = open(input_file, 'r', encoding='utf-8').read()
            doc = send_serifxml_document(data, docid=filename, hostname=hostname, port=port, document_date=document_date)
            doc.save(os.path.join(opts.output_dir, "%s.xml" % filename))
    elif args[0] == 'shutdown':
        send_shutdown_request(hostname, port)

#     if 0:
#         send_serifxml_document('foo', hostname='language-01')
#     if 0:
#         print 'sending doc to serif...'
#         e = send_serifxml_document('''
#         John talked to his sister Mary.
#         The president of Iran, Joe Smith, said he wanted to resume talks.
#         I saw peacemaker Bob at the mall.
#         ''', 'anonymous', 'English', 'localhost', 8081)
#         print 'serif is done!'
#         print e
#
#     if 0:
#         f = ('c:/TextGroup/Core/arabia/output-local0-x86_64/'
#              'actor/output/actor00.txt.xml')
#         #f = '/home/eloper/code/text/Core/SERIF/build/expts/arabic_translit_new/output/DIGRESSING_20050102.0130.sgm.xml'
#         #f = '/home/eloper/code/textDoc/Core/SERIF/build/expts/arabic_translit_new/output/DIGRESSING_20050102.0130.sgm.xml'
#         #f = '/home/eloper/tmp/serifxml_out/XIN_ENG_20030405.0080.sgm.xml'
#
#     if 0:
#         apf = send_serif_request('POST sgm2apf', '<DOC><DOCID>1</DOCID><TEXT>Bob and Mary went to the store. The store was called Safeway.</TEXT></DOC>', 'azamania11desk', 8081, True)
#         print apf
#     if 0:
#         doctheory = Document("icews.xml")
#         print doctheory.icews_actor_mention_set.pprint(depth=10)
#         print doctheory.icews_event_mention_set.pprint(depth=10)
#     if 0:
#         doctheory = send_serifxml_pm_request('<SerifXML version="18"><Document href="file://C:/temp/sample-doc.xml"></Document></SerifXML>', 'localhost', 8081)
#         print doctheory


# Load a document.
# etree = ET.parse(f).getroot()

# Set some "unknown" attributes
# etree.attrib['foo'] = 'bar'
# etree[0].attrib['baz'] = 'bap'

# d = Document(etree)
# s = d.sentences[0]
# print s.text
# s.save_text()
# print d
# print d.sentences[0]
# del d
# print s.text
# print s.mention_set[1]
# Modify the document.
# d.sentences[0].parse.root = d.sentences[0].parse.root[0]
# del d.sentences._children[1:]
# Save the document.
# d.save('foo.xml')
