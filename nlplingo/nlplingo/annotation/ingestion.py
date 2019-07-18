import codecs
import os
import re
import json

from nlplingo.annotation.ace import AceAnnotation
from nlplingo.annotation.serif import to_lingo_doc
from nlplingo.text.text_theory import Document
from nlplingo.annotation.idt import process_idt_file
from nlplingo.annotation.enote import process_enote_file
from nlplingo.annotation.spannotator import process_span_file
#from nlplingo.annotation.stanford_corenlp import process_corenlp_file
#from nlplingo.annotation.srl_column import process_srl_file


def parse_filelist_line(line):
    text_file = None
    idt_file = None
    enote_file = None
    acetext_file = None # ACE text file, where we only care about texts not within xml-tags
    apf_file = None     # ACE xml file
    span_file = None    # similar to Spannotator format
    corenlp_file = None
    srl_file = None
    serif_file = None
    lingo_file = None

    print(line)
    for file in line.strip().split():
        if file.startswith('TEXT:'):
            text_file = file[len('TEXT:'):]
        elif file.startswith('IDT:'):
            idt_file = file[len('IDT:'):]
        elif file.startswith('ENOTE:'):
            enote_file = file[len('ENOTE:'):]
        elif file.startswith('ACETEXT:'):
            acetext_file = file[len('ACETEXT:'):]
        elif file.startswith('APF:'):
            apf_file = file[len('APF:'):]
        elif file.startswith('SPAN:'):
            span_file = file[len('SPAN:'):]
        elif file.startswith('CORENLP'):
            corenlp_file = file[len('CORENLP:'):]
        elif file.startswith('SRL'):
            srl_file = file[len('SRL:'):]
        elif file.startswith('SERIF'):
            serif_file = file[len('SERIF:'):]
        elif file.startswith('LINGO'):
            lingo_file = file[len('LINGO:'):]
    print(serif_file)

    if text_file is None and acetext_file is None and serif_file is None and lingo_file is None:
        raise ValueError('TEXT, ACETEXT, SERIF, or LINGO must be present!')
    return (text_file, idt_file, enote_file, acetext_file, apf_file, span_file, corenlp_file, srl_file, serif_file, lingo_file)


def read_doc_annotation(filelists):
    """
    :type filelists: list[str]
    Returns:
        list[nlplingo.text.text_theory.Document]
    """
    docs = []
    """:type docs: list[text.text_theory.Document]"""
    for file_index, line in enumerate(filelists):
        (text_file, idt_file, enote_file, acetext_file, apf_file, span_file, corenlp_file, srl_file, serif_file, lingo_file) = parse_filelist_line(line)

        # TODO: we probably want to restrict having only text_file, serif_file, or acetext_file
        if text_file is not None:
            docid = os.path.basename(text_file)
            text_f = codecs.open(text_file, 'r', encoding='utf-8')
            all_text = text_f.read()
            text_f.close()
            doc = Document(docid, all_text.strip())

        if serif_file is not None:
            doc = to_lingo_doc(serif_file)

        if acetext_file is not None:
            docid = re.match(r'(.*).sgm', os.path.basename(acetext_file)).group(1)
            text_list = AceAnnotation.process_ace_textfile_to_list(acetext_file)
            # sometimes, e.g. for ACE, we need to keep the sentence strings separate. In ACE sgm files, it contains
            # things like '&amp;' which Spacy normalizes to a single character '&' and Spacy thus changed the original
            # character offsets. This is bad for keeping correspondences with the .apf file for character offsets.
            # So we let it affect 1 sentence or 1 paragraph, but not the rest of the document.

            # Some ACE text files have words that end with a dash e.g. 'I-'. This presents a problem. The annotation
            # file annotates 'I', but Spacy keeps it as a single token 'I-', and then I won't be able to find
            # Spacy tokens to back the Anchor or EntityMention. To prevent these from being dropped, we will replace
            # all '- ' with '  '.
            text_list = [s.replace(r'- ', '  ') for s in text_list]
            text_list = [s.replace(r' ~', '  ') for s in text_list]
            text_list = [s.replace(r'~ ', '  ') for s in text_list]
            text_list = [s.replace(r' -', '  ') for s in text_list]
            text_list = [s.replace(r'.-', '. ') for s in text_list] # e.g. 'U.S.-led' => 'U.S. led', else Spacy splits to 'U.S.-' and 'led'
            text_list = [s.replace(r'/', ' ') for s in text_list]

            doc = Document(docid, text=None, sentence_strings=text_list)

        if lingo_file is not None:
            with codecs.open(lingo_file, 'r', encoding='utf-8') as f:
                doc = Document.from_json(json.load(f))

        if idt_file is not None:
            doc = process_idt_file(doc, idt_file)  # adds entity mentions
        if enote_file is not None:
            doc = process_enote_file(doc, enote_file, auto_adjust=True)  # adds events
        if apf_file is not None:
            doc = AceAnnotation.process_ace_xmlfile(doc, apf_file)
        if span_file is not None:
            doc = process_span_file(doc, span_file)
        if corenlp_file is not None:
            doc = process_corenlp_file(doc, corenlp_file)
        if srl_file is not None:
            doc = process_srl_file(doc, srl_file)

        if doc is not None:
            docs.append(doc)

        if (file_index % 20) == 0:
            print('Read {} input documents out of {}'.format(str(file_index+1), str(len(filelists))))
    return docs
