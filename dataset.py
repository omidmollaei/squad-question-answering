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


def shorten_squad(ds, info: Parameters) -> Tuple[list, list, list]:
    contexts, questions, answers = [], [], []
    for sample in tqdm(ds):
        main_context, question = sample["context"], sample['question']
        answer = sample['answers']['text'][0]
        new_context = shorten_context(main_context, question, top_n=2)
        if answer not in new_context:
            info.num_dropped += 1
            continue
        contexts.append(new_context)
        questions.append(question)
        answers.append(answer)
    return contexts, questions, answers


def padding(params, context, question, answer):
    context_padded = tf.keras.preprocessing.sequence.pad_sequences(
        context, maxlen=params.max_context_len, padding="post")
    question_padded = tf.keras.preprocessing.sequence.pad_sequences(question, padding="post")
    answer_padded = tf.keras.preprocessing.sequence.pad_sequences(answer, padding="post")
    return context_padded, question_padded, answer_padded


def update_info(info, data):
    con, que, ans = data
    for c, q, a in zip(con, que, ans):
        info.contexts_lengths.append(len(c))
        info.questions_lengths.append(len(q))
        info.answers_lengths.append(len(a))
    return None


def tokenize_and_padding(params, info, contexts, questions, answers, tokenizer):
    context_tokenized, question_tokenized, answer_tokenized = [], [], []
    for c, q, a in tqdm(zip(contexts, questions, answers)):
        c = params.start_token + tokenizer.encode(c) + params.end_token     # tokenize context
        q = params.start_token + tokenizer.encode(q) + params.end_token     # tokenize question
        a = params.start_token + tokenizer.encode(a) + params.end_token     # tokenize answer
        if len(c) <= params.max_context_len:
            context_tokenized.append(c)
            question_tokenized.append(q)
            answer_tokenized.append(a)
        else:
            info.num_dropped += 1

    update_info(info, (context_tokenized, question_tokenized, answer_tokenized))
    contexts, questions, answers = padding(params, context_tokenized, question_tokenized, answer_tokenized)
    return contexts, questions, answers


def load_squad_dataset(params: Parameters):
    """main function to load squad dataset. The resulting dataset is a tf-dataset which contains two
    inputs (context and question) and one output (answer text)"""
    ds_info = Parameters(
        num_dropped=0,
        contexts_lengths=list(),
        questions_lengths=list(),
        answers_lengths=list(),
    )  # we use an instance of this to keep the info about dataset.

    print("loading dataset from huggingface ...")
    squad_dataset = datasets.load_dataset(ds_name, split='train')

    print("shortening samples ...")
    contexts, questions, answers = shorten_squad(squad_dataset, ds_info)

    print("initializing tokenizer ...")
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        contexts + questions[:30_000], target_vocab_size=params.vocab_size)

    tokenizer.save_to_file("saved_tokenizer")
    params.start_token = [tokenizer.vocab_size]
    params.end_token = [tokenizer.vocab_size + 1]
    params.vocab_size = params.vocab_size + 2

    print("tokenize and padding ... ")
    contexts, questions, answers = tokenize_and_padding(params, ds_info, contexts, questions, answers, tokenizer)

    dataset = tf.data.Dataset.from_tensor_slices((
        {"context_input": contexts, "question_input": questions}, answers
    ))
    dataset = dataset.cache().shuffle(len(contexts))
    dataset = dataset.batch(params.batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset, tokenizer, ds_info
