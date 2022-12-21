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


def padding(params, context_question, answer):
    context_question_padded = tf.keras.preprocessing.sequence.pad_sequences(
        context_question, maxlen=params.max_input_len, padding="post")
    answer_padded = tf.keras.preprocessing.sequence.pad_sequences(
        answer, padding="post")
    return context_question_padded, answer_padded


def concat_and_tokenize(params, ds_info, contexts, questions, answers, tokenizer):
    concatenated = []
    answers_tokenized = []
    for c, q, a in zip(contexts, questions, answers):
        concat = params.start_token + tokenizer.encode(c) + params.concat_token + tokenizer.encode(q) + params.end_token
        answer_tokenized = params.start_token + tokenizer.encode(a) + params.end_token
        if len(concat) > params.max_input_len:
            ds_info.num_dropped += 1
            continue
        ds_info.inputs_len.append(len(concat))
        ds_info.answers_len.append(len(answer_tokenized))
        concatenated.append(concat)
        answers_tokenized.append(answer_tokenized)
    return concatenated, answers_tokenized


def load_squad_dataset(params: Parameters):
    """main function to load squad dataset. The resulting dataset is a tf-dataset which contains two
    inputs (1. concatenated context and question for encoder input, 2. answers for decoder input)
     and one output which is the answers again."""

    ds_info = Parameters(
        num_dropped=0,
        inputs_len=list(),
        answers_len=list(),
    )  # we use an instance of this to keep info about dataset.

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
    params.concat_token = [tokenizer.vocab_size + 2]
    params.vocab_size = tokenizer.vocab_size + 3

    print("concat context with question and tokenize them ...")
    context_question, answers = concat_and_tokenize(params, ds_info, contexts, questions, answers, tokenizer)

    print("padding ... ")
    context_question, answers = padding(params, context_question, answers)

    dataset = tf.data.Dataset.from_tensor_slices((
        {"inputs": context_question, "dec_inputs": answers[:, :-1]}, answers[:, 1:]
    ))
    dataset = dataset.cache().shuffle(len(context_question))
    dataset = dataset.batch(params.batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset, tokenizer, ds_info


def tokens_to_text(tokenizer, tokens):
    last_idx = tf.reduce_sum(tf.cast(tf.not_equal(tokens, 0), tf.int32)) - 1
    tokens_list = list(tokens)
    sep_val = tokens_list[last_idx.numpy()] + 1
    sep_idx = tokens_list.index(sep_val)
    return tokenizer.decode(tokens[1:sep_idx]) + "[SEP]" + tokenizer.decode(tokens[sep_idx+1:last_idx])
