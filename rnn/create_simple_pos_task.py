import codecs, os
from collections import OrderedDict, Counter
import nltk

tagged_sents = nltk.corpus.brown.tagged_sents(tagset='universal')

words_by_line, tags_by_line = zip(*[zip(*sen)
                                    for sen in list(tagged_sents)[:10000]])


# print tagged_sents[:100], words_by_line[:100], tags_by_line[:100]

word_count = Counter([w for line in words_by_line for w in line])
tag_count = Counter([w for line in tags_by_line for w in line])

# print word_count, tag_count

vocab_word = [word for word, count  in word_count.most_common(100)]
vocab_tag = [tag for tag, count in tag_count.most_common(100)]
# print vocab_word, vocab_tag

word2idx = {word: idx for idx, word in enumerate(vocab_word)}
idx2word = {idx: word for idx, word in enumerate(vocab_word)}

tag2idx = {tag: idx for idx, tag in enumerate(vocab_tag)}
idx2tag = {idx: tag for idx, tag in enumerate(vocab_tag)}

words_by_line = [[word2idx.get(w,len(word2idx)+1) for w in line] for line in words_by_line]
tags_by_line = [[tag2idx.get(w, len(tag2idx)+1) for w in line] for line in tags_by_line]

sen_lens = map(lambda x:len(x) , words_by_line)

sent_len_hist = Counter(sen_lens)

# print words_by_line[:100], tags_by_line[:100], sent_len_hist

def get_label(words):
    if len(words) < 11:
        return 0
    if len(words) < 21:
        return 1
    return 2

with open('toy_feats_n.txt', 'w') as feats, \
     open('toy_tags_n.txt', 'w') as tags, \
     open('toy_class_n.txt', 'w') as cls:

    for index, line in enumerate(words_by_line):
        # print(line)
        feats.write('sent' + str(index) + ' ' + ' '.join(map(str,line)) + '\n')
        tags.write('sent' + str(index) + ' ' + ' '.join(map(str, tags_by_line[index])) + '\n')
        cls.write('sent' + str(index) + ' ' + str(get_label(line)) + '\n')
