"""load and preprocess SQuAD dataset, build tokenizer and convert dataset into a tf-dataset"""

import datasets
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
from typing import Tuple

ds_name = "squad"


class Parameters(object):
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)


def shorten_context(context: str, question: str, top_n: int = 2) -> str:
    """to shorten the context sequence, we would use a heuristic to extract those sentences
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
    best_sentences = list(top_matched["sent_no"])
    best_sentences = [sentences[s_no] for s_no in best_sentences]
    return ". ".join(best_sentences)


def shorten_squad(ds) -> Tuple[list, list, list]:
    contexts, questions, answers = [], [], []
    for sample in tqdm(ds):
        main_context, question = sample["context"], sample['question']
        answer = sample['answers']['text'][0]
        new_context = shorten_context(main_context, question, top_n=2)
        if answer not in new_context:
            continue
        contexts.append(new_context)
        questions.append(question)
        answers.append(answer)
    return contexts, questions, answers


def padding(params, c, q, a):
    tokenized_contexts = tf.keras.preprocessing.sequence.pad_sequences(
        c, maxlen=params.max_context_len, padding="post")
    tokenized_questions = tf.keras.preprocessing.sequence.pad_sequences(
        q, maxlen=params.max_question_len, padding="post")
    tokenized_answers = tf.keras.preprocessing.sequence.pad_sequences(
        a, maxlen=params.max_question_len, padding="post")
    return tokenized_contexts, tokenized_questions, tokenized_answers


def tokenize_and_padding(params, contexts, questions, answers, tokenizer):
    tokenized_contexts, tokenized_questions, tokenized_answers = [], [], []
    for c, q, a in tqdm(zip(contexts, questions, answers)):
        c = params.start_token + tokenizer.encode(c) + params.end_token
        q = params.start_token + tokenizer.encode(q) + params.end_token
        a = params.start_token + tokenizer.encode(a) + params.end_token
        if len(c) <= params.max_context_len and len(q) <= params.max_question_len \
                and len(a) <= params.max_question_len:     # filter long contexts|questions|answers
            tokenized_contexts.append(c)
            tokenized_questions.append(q)
            tokenized_answers.append(a)

    contexts, questions, answers = padding(params, tokenized_contexts, tokenized_questions, tokenized_answers)
    return contexts, questions, answers


def load_squad_dataset(params: Parameters):
    """main function to load squad dataset. The resulting dataset is a tf-dataset which contains two
    inputs (context and question) and one output (answer text)"""

    print("loading dataset from huggingface ...")
    squad_dataset = datasets.load_dataset(ds_name, split='train')

    print("shortening samples ...")
    contexts, questions, answers = shorten_squad(squad_dataset)

    print("initializing tokenizer ...")
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        contexts + questions[:25_000], target_vocab_size=params.vocab_size)
    tokenizer.save_to_file("saved_tokenizer")
    params.start_token = [tokenizer.vocab_size]
    params.end_token = [tokenizer.vocab_size + 1]
    params.vocab_size = params.vocab_size + 2

    print("tokenize and padding ... ")
    contexts, questions, answers = tokenize_and_padding(params, contexts, questions, answers, tokenizer)

    dataset = tf.data.Dataset.from_tensor_slices(({"context_input": contexts, "question_input": questions}, answers))
    dataset = dataset.cache().shuffle(len(contexts))
    dataset = dataset.batch(params.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset, tokenizer
