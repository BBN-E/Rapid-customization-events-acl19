

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

    if text_file is None and acetext_file is None and serif_file is None and lingo_file is None:
        raise ValueError('TEXT, ACETEXT, SERIF, or LINGO must be present!')
    return (text_file, idt_file, enote_file, acetext_file, apf_file, span_file, corenlp_file, srl_file, serif_file, lingo_file)

