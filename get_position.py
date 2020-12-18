import pickle
import tqdm
from collections import Counter


class Vocab_post(object):
    def __init__(self, counter, specials=['<pad>', '<unk>']):
        self.pad_index = 0
        self.unk_index = 1
        counter = counter.copy()
        self.itos = list(specials)
        for tok in specials:
            del counter[tok]

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            self.itos.append(word)

        # stoi is simply a reverse dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __eq__(self, other):
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def extend(self, v):
        words = v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1
        return self

    @staticmethod
    def load_vocab(vocab_path: str):
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)

import json
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare vocab for ASC.')
    parser.add_argument('data_dir', default='./datasets/acl-14-short-data', help='Twitter directory.')
    parser.add_argument('vocab_dir', default='./datasets/acl-14-short-data', help='Output vocab directory.')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # input files
    train_file = args.data_dir + '_train.raw'
    test_file = args.data_dir + '_test.raw'

    vocab_post_file = args.vocab_dir + 'vocab_post.pkl'

    # load files
    print("loading files...")
    train_max_len = get_max_length(train_file)
    test_max_len = get_max_length(test_file)

    # position embedding
    max_len = max(train_max_len, test_max_len)
    post_counter = Counter(list(range(-max_len, max_len)))
    post_vocab = Vocab_post(post_counter, specials=['<pad>', '<unk>'])

    with open(vocab_post_file, "wb") as f:
        pickle.dump(post_vocab, f)

    print("all done.")

def get_max_length(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    max_len = 0
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        sentence = text_left+' '+aspect+' '+text_right
        max_len = max(len(sentence), max_len)
    return max_len

if __name__ == '__main__':
    main()
