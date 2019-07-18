import os
import sys
import codecs
from collections import defaultdict
import re


def filter_serifxml(inpath, spans):
    ret = []

    add = True
    with codecs.open(inpath, 'r', encoding='utf-8') as f:
        for line in f:
            #line = line.rstrip()
            if '<Sentence ' in line:
                offset_tokens = re.search(r' char_offsets="(.*?)"', line).group(1).split(':')
                start = int(offset_tokens[0])
                end = int(offset_tokens[1])

                found_offset = False
                for offset in spans:
                    if start <= offset[0] and offset[1] <= end: 
                        found_offset = True
                        break
                if not found_offset:
                    add = False
                #if offset not in spans:
                #    add = False
            if add:
                ret.append(line)
            if '</Sentence>' in line:
                add = True
            if '</Sentences>' in line:
                ret.append('  </Document>')
                ret.append('</SerifXML>')
                break

    sents = [s for s in ret if '<Sentence ' in s]
    sentence_count = len(sents)
    return ret, sentence_count

def get_serifxml_docid_from_filename(filename):
    if filename.endswith('.serifxml'):
        docid = filename[:-len('.serifxml')]
    elif filename.endswith('.sgm.xml'):
        docid = filename[:-len('.sgm.xml')]
    elif filename.endswith('.xml'):
        docid = filename[:-len('.xml')]
    else:
        raise ValueError('ERROR: filename %s ends with an unknown extension' % (filename))
    return docid

def shrink_serifxml(serif_list,span_list,output_dir,new_span_serif_file):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    docid_filepath = dict()

    with codecs.open(serif_list, 'r', encoding='utf-8') as f:
        for line in f:
            for token in line.strip().split():
                if token.startswith('SERIF'):
                    filepath = token.split(':')[1]
                    filename = os.path.basename(filepath)
                    docid = get_serifxml_docid_from_filename(filename)

                    if docid in docid_filepath:
                        print('ERROR: %s already in docid_filepath' % (docid))
                    else:
                        docid_filepath[docid] = filepath


    docid_spans = defaultdict(set)
    with codecs.open(span_list, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            docid = tokens[0]
            start = int(tokens[1])
            end = int(tokens[2]) - 1

            if docid not in docid_filepath:
                print('ERROR: docid %s cannot be found in docid_filepath' % (docid))
            else:
                docid_spans[docid].add((start, end))

    docid_new_filepath = dict()
    for docid in docid_spans:
        print(docid)
        doc_spans = docid_spans[docid]	# these are sentence offsets

        assert docid in docid_filepath
        serifxml_filepath = docid_filepath[docid]
        new_serifxml_filepath = os.path.join(output_dir, docid)
        docid_new_filepath[docid] = new_serifxml_filepath

        outlines, sentence_count = filter_serifxml(serifxml_filepath, doc_spans)
        if sentence_count != len(doc_spans):
            print(' - discarding %d/%d sentence spans' % (len(doc_spans)-sentence_count, len(doc_spans)))
        with codecs.open(new_serifxml_filepath, 'w', encoding='utf-8') as o:
            for line in outlines:
                o.write(line)
                #o.write('\n')


    new_span_serif_lines = []
    with codecs.open(serif_list, 'r', encoding='utf-8') as f:
        for line in f:
            out_tokens = []
            for token in line.strip().split():
                if token.startswith('SERIF'):
                    filepath = token.split(':')[1]
                    filename = os.path.basename(filepath)
                    docid = get_serifxml_docid_from_filename(filename)
                    out_tokens.append('SERIF:%s' % (docid_new_filepath[docid]))
                else:
                    out_tokens.append(token)
            new_span_serif_lines.append(' '.join(out_tokens))

    with codecs.open(new_span_serif_file, 'w', encoding='utf-8') as o:
        for line in new_span_serif_lines:
            o.write(line)
            o.write('\n')



if __name__ == "__main__":
    serif_list = sys.argv[1]
    span_list = sys.argv[2]
    output_dir = sys.argv[3]
    new_span_serif_file = sys.argv[4]
    shrink_serifxml(serif_list,span_list,output_dir,new_span_serif_file)
