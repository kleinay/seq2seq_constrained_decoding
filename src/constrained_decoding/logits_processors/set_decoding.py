from typing import List, Iterable, Tuple, Any, Union, Dict, Optional, Callable

import torch
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizer, LogitsProcessor

Element = List[str] 

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

class SetDecodingLogitsProcessor(LogitsProcessor):
    """
    Decode a set, i.e. a list of elements without rehearsals.
    We assume the set elements are sub-sequences separated by an explicit special separator token.
    Implementation: at each step of the beam search, consider previous "elements" occuring in output, 
     and ban tokens that complement the current element sub-sequence to be equivalent to an pre-occuring element.    
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, elements_sep: str, element_eq_func: Optional[Callable[[Element, Element], bool]] = None):
        self.tokenizer = tokenizer
        self.elements_sep = elements_sep
        self.element_eq_func = element_eq_func
        self.vocab: Dict[str, int] = tokenizer.get_vocab()
        self.vocab_words = np.array(list(self.vocab.keys()))
        self.vocab_word_ids = np.array(list(self.vocab.values()))
    
    def get_prev_and_current_elements(self, b_input_ids) -> Tuple[List[Element], Element]:
        # Returns previous_elements, current_element
        input_words: List[str] = self.tokenizer.convert_ids_to_tokens(b_input_ids)
        elements = listSplit(input_words, self.elements_sep)
        previous_elements, current_element = elements[:-1], elements[-1]
        return previous_elements, current_element
        
    def __call__(self, input_ids, scores):
        """
        This method will be called during each step of the beam search algorithm. 
        The method takes as input the input_ids sequence of the partially generated beam and the scores of the next possible tokens.
        By manipulating these scores based on the tokens present in the input_ids, we can control the structure of the generated sentence.
        """
        banned_tokens = []
        for beam_index, (beam_input_ids, beam_scores) in enumerate(zip(input_ids, scores)):
            batch_banned_token_ids = []
            # edit batch_banned_tokens according to beam_input_ids 
            previous_elements, current_element = self.get_prev_and_current_elements(beam_input_ids)
            # collect forbidden next-words
            forbidden_next_words = set()
            for word in self.vocab_words:
                # if word complement current element to be equivalent to any of previous elements
                for prev_element in previous_elements:
                    if (self.element_eq_func is not None and self.element_eq_func(current_element + [word], prev_element)) or \
                        (self.element_eq_func is None and current_element + [word] == prev_element):
                        forbidden_next_words.add(word)
            # special case: ban separator itself if current_element is in previous elements
            for prev_element in previous_elements:
                if (self.element_eq_func is not None and self.element_eq_func(current_element, prev_element)) or \
                    (self.element_eq_func is None and current_element == prev_element):
                    forbidden_next_words.add(self.elements_sep)
            # vectorize to apply on array of ids
            is_forbidden = vectorized_inclusion_func(forbidden_next_words)  # numpy vectorized function
            banned_indexes = np.where(is_forbidden(self.vocab_words)) # get banned-words indexes in self.vocab_words, corresponding to indexes in self.vocab_word_ids
            batch_banned_token_ids = self.vocab_word_ids[banned_indexes]
            
            banned_tokens.append(batch_banned_token_ids)
        scores = set_scores_to_inf_for_banned_tokens(scores, banned_tokens)
        return scores
    

if __name__ == "__main__":
    # test SetDecodingLogitsProcessor
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    text = "A B A # I M A # A T A # A B # A B C # A B "
    # tokenizer.vocab = tokenizer.get_vocab() # {'I': 4, 'M': 5, 'T': 6, 'B': 2, 'A': 1, '#': 0, 'C': 3}
    text_ids = tokenizer.convert_tokens_to_ids(text.split())
    input_ids = torch.tensor([text_ids])

    # inputs = tokenizer(text)
    scores = torch.tensor([[1/len(tokenizer.get_vocab())] * len(tokenizer.get_vocab())])
    
    processor = SetDecodingLogitsProcessor(tokenizer, '#')
    print(f"Scores before: { {tokenizer.convert_ids_to_tokens(i): scores[0][i] for i in set(text_ids)} }")
    new_scores = processor(input_ids, scores)
    
    print(f"Scores After : { {tokenizer.convert_ids_to_tokens(i): new_scores[0][i] for i in set(text_ids)} }")
