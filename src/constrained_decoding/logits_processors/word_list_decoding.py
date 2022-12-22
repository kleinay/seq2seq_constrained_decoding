from typing import List, Iterable, Tuple, Any, Union, Dict, Optional, Callable

import torch
from transformers import LogitsProcessor

class WhiteListLogitsProcessor(LogitsProcessor):
    """
    A simple LogitsProcessor constraining the generation to "white list", i.e. a set of allowed tokens.     
    """
    def __init__(self, white_list_word_ids: Iterable[int]):
        self.white_list = white_list_word_ids  
            
    def __call__(self, input_ids, scores):
        """
        This method will be called during each step of the beam search algorithm. 
        The method takes as input the input_ids sequence of the partially generated beam and the scores of the next possible tokens.
        By manipulating these scores based on the tokens present in the input_ids, we can control the structure of the generated sentence.
        """
        mask = torch.ones(scores.shape)
        # put zeros in allowed tokens
        mask[:, self.white_list] = 0
        scores = scores.masked_fill(mask.bool(), -float("inf"))
        return scores
    
class BlackListLogitsProcessor(LogitsProcessor):
    """
    A simple LogitsProcessor constraining the generation by a "black list", i.e. a set of forbidden tokens.     
    """
    def __init__(self, black_list_word_ids: Iterable[int]):
        self.black_list = black_list_word_ids  
            
    def __call__(self, input_ids, scores):
        """
        This method will be called during each step of the beam search algorithm. 
        The method takes as input the input_ids sequence of the partially generated beam and the scores of the next possible tokens.
        By manipulating these scores based on the tokens present in the input_ids, we can control the structure of the generated sentence.
        """
        mask = torch.zeros(scores.shape)
        # put ones in allowed tokens
        mask[:, self.black_list] = 1
        scores = scores.masked_fill(mask.bool(), -float("inf"))
        return scores
    

