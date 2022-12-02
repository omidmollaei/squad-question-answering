"""load and preprocess SQuAD dataset, build tokenizer and convert dataset into a tf-dataset"""

import os
import datasets
import pandas as pd
import tensorflow as tf
from tqdm import tqdm


class Parameters(object):
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)


def shorten_context(context: str, question: str, top_n: int = 2) -> str:
    """To shorten the context sequence, we would use a heuristic to extract those sentences
     which the answer lies in. The solution is simple: we should extract top n sentences from
     the context paragraph which have the most common words with related question."""

    question_split = question.lower().split(" ")
    sentences = context.split(".")
    all_match = list()
    for sent_no, s in enumerate(sentences):
        s = s.lower()
        num_match = 0
        words = s.split(" ")
        for w_q in question_split:
            if w_q in words:
                num_match += 1
        all_match.append((sent_no, num_match))
    matches = pd.DataFrame(all_match, columns=["sent_no", "n_match"])
    matches.sort_values(by=["n_match", "sent_no"], ascending=False, inplace=True)
    top_matched = matches.iloc[:top_n]
    best_sentences = list(all_match["sent_no"])
    best_sentences = [sentences[s_no] for s_no in best_sentences]
    return ". ".join(best_sentences)
