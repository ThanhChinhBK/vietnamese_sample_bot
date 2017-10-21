import os
import re
import random
import argparse
from nltk.tokenize import sent_tokenize


def read_srt(fname, sents):    
    pattern = re.compile(r'((\d+\:){2}\d+\,\d+ --> (\d+\:){2}\d+,)')
    
    with open(fname, "rb") as f:
        #lines = f.readlines()
        for line in f:
            try: line.decode('utf-8')
            except:
                continue
            if line.strip() == b"":
                continue
            if pattern.match(line.strip().decode('utf-8')) or line.strip().isdigit():
                continue
            line_sents = sent_tokenize(line.strip().decode('utf-8'))
            if len(line_sents) == 0: continue
            sents.append(line_sents) #senteces decoded

def read_ass(fname, sents):
    with open(fname, "rb") as f:
        for line in f:
            if not line.strip().startswith(b"Dialogue:"):
                continue
            line = line.strip().split(b",,")[-1].decode("utf-8")
            line = line.replace("\\N", " ")
            line = re.sub(r"({.*})", " ", line)
            line_sents = sent_tokenize(line.strip())
            if len(line_sents) == 0: continue
            sents.append(line_sents) # sentences decoded

def question_answers(sents):
    """ Divide the dataset into two sets: questions and answers. """
    ques = []
    ans = []
    for i in range(len(sents) - 1):
        if sents[i][-1].endswith("?") and not sents[i+1][0].endswith("?"):
            ques.append(sents[i][-1].encode("utf-8"))
            ans.append(sents[i+1][-0].encode("utf-8"))
    return ques, ans

def basic_tokenizer(line, normalize_digits=True):
    """ A basic tokenizer to tokenize text into tokens.
    Feel free to change this to suit your need. """
    
    line = re.sub(b'<.*>', b'', line)
    line = re.sub(b'\[', b'', line)
    line = re.sub(b'\]', b'', line)
    line = re.sub(b'\\n', b' ', line)
    words = []
    _WORD_SPLIT = re.compile(br"([.,!?\"'-<>:;)(])")
    _DIGIT_RE = re.compile(br"\d")
    for fragment in line.strip().lower().split():
        for token in re.split(_WORD_SPLIT, fragment):
            if not token:
                continue
            if normalize_digits:
                token = re.sub(_DIGIT_RE, b'#', token)
            words.append(token)
    return words

def build_vocab(sents, preprocess_path, normalize_digits=True):
    
    out_path = os.path.join(preprocess_path, 'vocab.txt')

    vocab = {}
    vocab[b"<pad>"] = 0
    vocab[b"<unk>"] = 0
    for line in sents:
        for token in basic_tokenizer(line):
            if not token in vocab:
                vocab[token] = 0
            vocab[token] += 1
                
    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
    with open(out_path, 'w') as f:
        #f.write('<pad>' + '\n')
        #f.write('<unk>' + '\n')
        index = 2
        for word in sorted_vocab:
            f.write(word.decode("utf-8") + '\n')
            index += 1
    sorted_vocab_dict  = {sorted_vocab[i]:i for i in range(len(sorted_vocab))} 
    return sorted_vocab_dict

def sentence2id(vocab, line):
    return [vocab.get(token, vocab[b'<unk>']) for token in basic_tokenizer(line)]

def corpus2id(ques, ans, vocab):
    id_ques , id_ans = [], []
    for q in ques:
        id_ques.append(sentence2id(vocab, q))
    for a in ans:
        id_ans.append(sentence2id(vocab, a))
    return id_ques, id_ans

def prepare_data(ques, ans, vocab, test_size):
    id_ques, id_ans = corpus2id(ques, ans, vocab)
    files = []
    filenames = ['train.que', 'train.ans', 'test.que', 'test.ans']
    test_ids = random.sample([i for i in range(len(ques))], test_size)
    for filename in filenames:
        files.append(open(os.path.join("preprocess", filename),'w'))

    for i in range(len(ques)):
        if i in test_ids:
            files[2].write(' '.join(str(ids) for ids in id_ques[i]) + '\n')
            files[3].write(' '.join(str(ids) for ids in id_ans[i]) + '\n')
        else:
            files[0].write(' '.join(str(ids) for ids in id_ques[i]) + '\n')
            files[1].write(' '.join(str(ids) for ids in id_ans[i]) + '\n')

    for file in files:
        file.close()
    
def write_raw_data(ques, ans, preprocess_path):
    f = open(os.path.join(preprocess_path, "raw_data.txt"), "w")
    for q, a in zip(ques, ans):
        f.write(q.decode("utf-8")+"\n")
        f.write("\t" + a.decode("utf-8") + "\n")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_size', type=int, help='size of test set', required=True)
    args = parser.parse_args()
    sents = []
    for fname in os.listdir("data"):
        if fname.endswith(".srt"):
            read_srt(os.path.join("data", fname), sents)
        if fname.endswith(".ass"):
            read_ass(os.path.join("data", fname), sents)

    print("got {} lines".format(len(sents)))
    ques, ans = question_answers(sents)
    write_raw_data(ques, ans, "preprocess/")
    print("got {} pairs of question-answer".format(len(ques)))
    vocab = build_vocab(ques + ans, "preprocess/")
    print("get {} words in vocab".format(len(vocab)))
    prepare_data(ques, ans, vocab, args.test_size)
