"""load and preprocess SQuAD dataset, build tokenizer and convert dataset into a tf-dataset"""

import datasets
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
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


def tokenization(params, contexts: list, questions: list, answers: list):
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        contexts + questions[:25_000], target_vocab_size=params.vocab_size
    )
    tokenizer.save_to_file("saved_tokenizer")
    tokenized_context_questions, tokenized_answers = [], []
    for (context, question), answer in tqdm(zip(zip(contexts, questions), answers)):
        sentence1 = params.start_token + tokenizer.encode(context) + params.sep_token + \
                    tokenizer.encode(question) + params.end_token
        sentence2 = params.start_token + tokenizer.encode(answer)
        tokenized_context_questions.append(sentence1)
        tokenized_answers.append(sentence2)

    return tokenized_context_questions, tokenized_answers, tokenizer


def load_squad_dataset(params: Parameters):
    """main function to load squad dataset. The resulting dataset is a tf-dataset which contains two
    inputs (context and question) and one output (answer text)"""

    contexts, questions, answers = [], [], []

    print("loading dataset from huggingface ...")
    squad_dataset = datasets.load_dataset(ds_name, split='train')

    print("shortening samples ...")
    for sample in tqdm(squad_dataset):
        main_context, question = sample["context"], sample['question']
        answer = sample['answers']['text'][0]
        new_context = shorten_context(main_context, question, top_n=3)
        if answer not in new_context:
            continue
        contexts.append(new_context)
        questions.append(question)
        answers.append(answer)

    print("tokenization ...")
    context_questions, answers, tokenizer = tokenization(contexts, questions, answers)

    print("padding ...")
    context_questions = tf.keras.layers.preprocessing.sequnce.pad_sequence(
        context_questions, maxlen=params.max_len
    )

    dataset = tf.data.Dataset.from_tensor_slices((
        {"context_input": contexts, "question_input": questions},
        answers
    ))
    dataset = dataset.cache().shuffle(len(contexts))
    dataset = dataset.batch(params.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    if with_info:
        return dataset, tokenizer, ds_info
    return dataset, tokenizer
