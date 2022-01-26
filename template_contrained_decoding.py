from typing import List, Iterable, Tuple, Any, Union, Dict, Optional, Callable, Hashable

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, AutoModelForSeq2SeqLM

# Helper functions

def set_scores_to_inf_for_banned_tokens(scores, banned_tokens):
# src: https://huggingface.co/transformers/v4.1.1/_modules/transformers/generation_logits_process.htm
    """
    Modifies the scores in place by setting the banned token positions to `-inf`. banned_tokens is expected to be a
    list of list of banned tokens to ban in the format [batch_0_banned_word_ids: List[int], batch_1_banned_word_ids: List[int], ... ]

    Args:
        scores: logits distribution of shape (batch size, vocabulary size)
        banned_tokens: list of lists of tokens to ban, of length (batch_size). 
            tokens are specified by token_ids (value of token in vocab), which correspond to position in `scores` last axis. 
    """
    banned_mask_list = []
    for idx, batch_banned_tokens in enumerate(banned_tokens):
        for token in batch_banned_tokens:
            banned_mask_list.append([idx, token])
    if not banned_mask_list:
        return scores

    banned_mask = torch.LongTensor(banned_mask_list)
    indices = torch.ones(len(banned_mask))

    banned_mask = (
        torch.sparse.LongTensor(banned_mask.t(), indices, scores.size()).to(scores.device).to_dense().bool()
    )
    scores = scores.masked_fill(banned_mask, -float("inf"))
    return scores

def listSplit(lst, delimeterElement):
    import itertools
    # as str.split(); return a splitted list (list of sub-lists), splitted by the delimeter
    return[list(y) for x, y in itertools.groupby(lst, lambda z: z == delimeterElement) if not x]

def vectorized_inclusion_func(included_within: Iterable[Any]):
    return np.frompyfunc(lambda x: x in included_within, 1, 1)
def vectorized_exclusion_func(excluded_from: Iterable[Any]):
    return np.frompyfunc(lambda x: x not in excluded_from, 1, 1)

class ConstrainedDecodingLogitsProcessor(LogitsProcessor):
    """
    Decode a set, i.e. a list of elements without rehearsals.
    We assume the set elements are sub-sequences separated by an explicit special separator token.
    Implementation: at each step of the beam search, consider previous "elements" occuring in output, 
     and ban tokens that complement the current element sub-sequence to be equivalent to an pre-occuring element.    
    """
    def __init__(self, tokenizer, qas_sep: str, q_a_sep: str, answers_sep: str):
        self.tokenizer = tokenizer
        self.qas_sep = qas_sep
        self.q_a_sep = q_a_sep
        self.answers_sep = answers_sep
        
        self.vocab_words = np.array(list(tokenizer.vocab.keys()))
        self.vocab_word_ids = np.array(list(tokenizer.vocab.values()))
        # TODO Prepare useful vectorized functions

        
    def __call__(self, input_ids, scores):
        """
        This method will be called during each step of the beam search algorithm. 
        The method takes as input the input_ids sequence of the partially generated beam and the scores of the next possible tokens.
        By manipulating these scores based on the tokens present in the input_ids, we can control the structure of the generated sentence.
        """
        banned_tokens = []
        for beam_index, (beam_input_ids, beam_scores) in enumerate(zip(input_ids, scores)):
            batch_banned_token_ids = []
            # Edit batch_banned_token_ids according to beam_input_ids 
            
            def is_token_forbidden(word: str) -> bool:
                # TODO
                pass
            # vectorize to apply on array of ids
            is_forbidden = np.frompyfunc(is_token_forbidden, 1, 1)  # numpy vectorized function
            banned_indexes = np.where(is_forbidden(self.vocab_words)) # get banned-words indexes in self.vocab_words, corresponding to indexes in self.vocab_word_ids
            batch_banned_token_ids = self.vocab_word_ids[banned_indexes]
            
            banned_tokens.append(batch_banned_token_ids)
        scores = set_scores_to_inf_for_banned_tokens(scores, banned_tokens)
        return scores