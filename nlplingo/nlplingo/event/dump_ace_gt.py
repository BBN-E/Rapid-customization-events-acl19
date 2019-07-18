import codecs
import argparse

from nlplingo.annotation.ingestion import read_doc_annotation
import nlplingo.common.io_utils as io_utils


def _main(args):
    docs = read_doc_annotation(io_utils.read_file_to_list(args.filelists))
    trigger_output_file_path = args.output_prefix + 'triggers'
    args_output_file_path = args.output_prefix + '.args'
    f = codecs.open(trigger_output_file_path, 'w', encoding='utf8')
    f_arg = codecs.open(args_output_file_path, 'w', encoding='utf8')

    for doc in docs:
        for event in doc.events:
            for anchor in event.anchors:
                f.write('{}\t{}\t{}\t{}\n'.format(
                    doc.docid,
                    anchor.label,
                    anchor.start_char_offset(),
                    anchor.end_char_offset()
                    )
                )
                for arg in event.arguments:
                    f_arg.write(
                        '{}\t{}\t{}\t{}\t{}\n'.format(
                            doc.docid,
                            anchor.label,
                            arg.label,
                            arg.start_char_offset(),
                            arg.end_char_offset()
                        )
                    )
    f.close()
    print('Done')


def parse_setup():
    parser = argparse.ArgumentParser(
        description='This program takes in a span file and renders ground truth annotations'
                    ' for scoring outside of nlplingo'
    )
    parser.add_argument('filelists', type=str, help='Span file list')
    parser.add_argument('output_prefix', type=str, help='Output prefix for .triggers and .args files generated')
    return parser


if __name__ == '__main__':
    _main(parse_setup().parse_args())