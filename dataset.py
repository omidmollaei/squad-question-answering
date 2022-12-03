"""load and preprocess SQuAD dataset, build tokenizer and convert dataset into a tf-dataset"""

import datasets
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

ds_name = "squad"


class Parameters(object):
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)


def parameters_hint():
    hint = f"""
    An instance of Parameters class must contains the following attributes:
    --------------------------------------------------------------------------------------------------
    1) vocab_size {'':20}: Total vocabulary size, required by tokenizer.                             
    2) batch_size {'':20}: Size of each batch in tf-dataset.                                         
    3) model_dim  {'':20}: Model dimension while processing sequences                                
    4) num_heads  {'':20}: Number of attention heads in self and cross attention encoder.            
    5) atten_encoder_num_layers {'':6}: Number of layers in self-attention encoder.                  
    6) recurrent_encoder_num_layers {'':2}: Number of layers in recurrent-attention encoder.         
    7) dense_units {'':19}: Number of units to project during model sequence processing.             
    8) activation {'':20}: Activation function to be used during model processing.                   
    9) dropout_rate {'':20}: Rate for Dropout layer. 
    --------------------------------------------------------------------------------------------------
    """
    return hint


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


def add_special_tokens(s: str):
    s = tf.strings.join(inputs=["[SOS]", s, "[EOS]"], separator=" ")
    s = tf.strings.strip(s)
    return s


def build_tokenizer(vocab_size: int, text_data: list):
    tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        standardize=add_special_tokens,
        ragged=True
    )
    tokenizer.adapt(text_data)
    return tokenizer


def load_squad_dataset(params: Parameters, with_info: bool = False, tokenize_answers: bool = True):
    """main function to load squad dataset. The resulting dataset is a tf-dataset which contains two
    inputs (context and question) and three outputs (answer, answer-start and answer-len)."""

    num_dropped = 0
    ds_info = {
        "contexts_length": [],
        "questions_length": [],
        "answers_length": [],
    }
    contexts, questions, answers = [], [], []
    answers_start, answers_len = [], []

    print("loading dataset from huggingface ...")
    squad_dataset = datasets.load_dataset(ds_name, split='train')

    print("shortening samples ...")
    for sample in tqdm(squad_dataset):
        main_context, question = sample["context"], sample['question']
        answer = sample['answers']['text'][0]
        new_context = shorten_context(main_context, question, top_n=3)
        if answer not in new_context:
            num_dropped += 1
            continue
        contexts.append(new_context)
        questions.append("[SOA] [LOA] " + question)
        answers.append(answer)
        answers_start.append(new_context.find(answer))
        answers_len.append(len(answer))
        ds_info['contexts_length'].append(len(new_context.split(" ")))
        ds_info['questions_length'].append(len(question.split(" ")))
        ds_info['answers_length'].append(len(answer.split(" ")))

    ds_info["num_dropped"] = num_dropped

    print("initializing tokenizer ...")
    tokenizer = build_tokenizer(vocab_size=params.vocab_size,
                                text_data=contexts + questions[:40_000])
    print("build tf-dataset ...")
    if tokenize_answers:
        answers = tokenizer(answers).to_tensor()

    dataset = tf.data.Dataset.from_tensor_slices((
        {"context_input": contexts, "question_input": questions},
        {"answer_text": answers, "answer_start": answers_start, "answer_length": answers_len}
    ))
    dataset = dataset.cache().shuffle(len(contexts))
    dataset = dataset.batch(params.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    if with_info:
        return dataset, tokenizer, ds_info
    return dataset, tokenizer
