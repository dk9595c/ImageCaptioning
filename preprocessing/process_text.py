import collections
from typing import Counter

def pad_captions(captions: list[list[str]], window_size: int) -> None:
    """
    Pads each caption in-place to exactly window_size + 1
    tokens by appending '<pad>' tokens.

    After calling preprocess_captions every caption looks like:
        ['<start>', w1, w2, ..., wN, '<end>']
    where the total length is between 2 and window_size + 1.  This function
    ensures every caption is exactly window_size + 1 tokens long so they can
    be stacked into a rectangular numpy array.

    Args:
        captions    - list of token lists, modified in-place
        window_size - int; the maximum caption length (excluding padding)

    TODO: For each caption, append enough '<pad>' tokens so that the caption
          has exactly window_size + 1 tokens total.
    """
    for caption in captions:
        padding_needed = (window_size + 1) - len(caption)
        if padding_needed > 0:
            caption.extend(['<pad>'] * padding_needed)


def unk_captions(captions: list[list[str]], word_count: Counter[str], minimum_frequency: int) -> None:
    """
    Replaces infrequent tokens with '<unk>' in-place.

    Any token whose frequency in word_count is <= minimum_frequency is
    considered rare and should be replaced with the special '<unk>' token.

    Args:
        captions          - list of token lists, modified in-place
        word_count        - collections.Counter mapping token -> frequency
        minimum_frequency - int; tokens with count <= this value are replaced

    TODO: For each token in each caption, check its frequency in word_count.
          If the frequency is <= minimum_frequency, replace it with '<unk>'.
    """
    for caption in captions:
        for i in range(len(caption)):
            if word_count[caption[i]] <= minimum_frequency:
                caption[i] = '<unk>'


def build_word_dictionary(train_captions: list[list[str]], test_captions: list[list[str]]) -> dict[str, int]:
    """
    Builds a word-to-index mapping from the training captions and converts
    both train and test captions from token lists to integer index lists
    in-place.

    Steps:
        1. Iterate over every token in train_captions in order.  The first
           time a token is seen, assign it the next available integer index
           (starting from 0).  Replace the token in the list with its index.
        2. Iterate over every token in test_captions.  Replace each token
           with its index from word2idx; for unknown tokens, use the index of
           '<unk>'.

    Args:
        train_captions - list of token lists (modified in-place → int lists)
        test_captions  - list of token lists (modified in-place → int lists)

    Returns:
        word2idx - dict mapping str token → int index

    TODO: Build a word2idx dictionary by scanning train_captions.
          Then convert test_captions using the same mapping (use '<unk>'
          for any word not seen during training).
    """
    word2idx = {}
    vocab_size = 0

    for caption in train_captions:
        for i in range(len(caption)):
            word = caption[i]
            if word not in word2idx:
                word2idx[word] = vocab_size
                vocab_size += 1
            caption[i] = word2idx[word]

    for caption in test_captions:
        for i in range(len(caption)):
            word = caption[i]
            caption[i] = word2idx.get(word, word2idx['<unk>'])

    return word2idx
