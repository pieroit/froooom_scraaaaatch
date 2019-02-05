# -*- coding: utf-8 -*-
from __future__ import print_function
import spacy


if __name__ == '__main__':

    nlp = spacy.load('en_core_web_lg')

    clothes  = nlp(u'Cotton pijamas at 20$ with 15% discount. The warmest in Minnesota!!')
    clothes2 = nlp(u'Do you want to stay cool? Try the cotton trousers from Cotton Company l.t.d..')
    banking  = nlp(u'Keep your money safe with the Bank of Texas credit card.')

    print(clothes.similarity(clothes2))
    print(clothes.similarity(banking))

    print('\n\nTOKENS')
    for word in clothes:
        print('>', word.text)
        print('\tlemma\t', word.lemma_)
        print('\tpos\t', word.pos_, '\t...', spacy.explain(word.pos_))
        print('\ttag\t', word.tag_, '\t...', spacy.explain(word.tag_))
        print('\tdep\t', word.dep_, '\t...', spacy.explain(word.dep_))
        print('\tshape\t', word.shape_)

    print('\n\nENTITIES')
    for ent in clothes.ents:
        print('\t', ent.text, ent.label_)

    print('\n\nNOUN CHUNKS')
    for chunk in clothes.noun_chunks:
        print(chunk.text, '...', chunk.root.text)