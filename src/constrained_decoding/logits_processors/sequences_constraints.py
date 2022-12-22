"""
LogitProcessors for defining a set of forbidden / allowed sequences or sub-sequences.
"""
from typing import List, Iterable, Tuple, Any, Union, Dict, Optional, Callable

import torch
from transformers import LogitsProcessor

def is_prefix(lst: List[Any], prefix: List[Any]):
    "Is `prefix` a prefix of of `lst` "
    return lst[:len(prefix)] == prefix

def filter_none(lst: Iterable[Any]) -> List[Any]:
    return [e for e in lst if e is not None]

class AllowedSequencesLogitsProcessor(LogitsProcessor):
    """
    A LogitsProcessor that constrain the output to be one of provided sequences.     
    """
    def __init__(self, allowed_sequences: List[List[int]], eos_token_id: Optional[int] = None):
        self.allowed_sequences_ids = allowed_sequences  
        self.eos_token_id = eos_token_id  
            
    def __call__(self, input_ids, scores):
        """
        This method will be called during each step of the beam search algorithm. 
        The method takes as input the input_ids sequence of the partially generated beam and the scores of the next possible tokens.
        By manipulating these scores based on the tokens present in the input_ids, we can control the structure of the generated sentence.
        """
        mask = torch.ones(scores.shape) # start with all ones - all forbiden
        # put zeros in allowed tokens
        for beam_index, (beam_input_ids, beam_scores) in enumerate(zip(input_ids, scores)):
            beam_input_ids = beam_input_ids.tolist()
            # does context so far match any allowed-sequences?
            possible_allowed_sequences = [allowed_seq 
                                          for allowed_seq in self.allowed_sequences_ids
                                          if is_prefix(allowed_seq, prefix=beam_input_ids)]
            if possible_allowed_sequences:
                # derive allowed next tokens
                cur_idx = len(beam_input_ids)
                allowed_next_tokens = filter_none(sorted({
                    seq[cur_idx] if cur_idx < len(seq) else self.eos_token_id
                    for seq in possible_allowed_sequences}))

         
                # in this beam, allow only `allowed_next_tokens` 
                mask[beam_index, allowed_next_tokens] = 0

        # use mask to forbid all tokens that correspond to 1 in mask
        scores = scores.masked_fill(mask.bool().to(scores.device), -float("inf"))
        return scores
    
    
def dictOfLists(pairs: List[Tuple[Any, Any]]) -> Dict[Any, List[Any]]:
    # return a { key : [values given to that key] } for the pair list.
    # e.g. dictOfLists( [(0, "r"), (4, "s"), (0, "e")])  will return {0: ["r", "e"], 4: ["s"]}
    from collections import defaultdict
    r = defaultdict(list)
    for k,v in pairs:
        r[k].append(v)
    return dict(r)

class ForbiddenSequencesLogitsProcessor(LogitsProcessor):
    """
    A LogitsProcessor that constrain the output to disallow any of the provided sequences.
    The constraint is only applied when decoding the the last token in the forbidden sequence.      
    """
    def __init__(self, forbidden_sequences: List[List[int]]):
        """ 

        Args:
            forbidden_sequences (List[List[int]]): the forbidden sequences (tokenized into token_ids).
        """
        # make sure `forbidden_sequences` is a list and not a tensor
        if torch.is_tensor(forbidden_sequences):
            if forbidden_sequences.ndim == 1:
                forbidden_sequences = [forbidden_sequences.tolist()] # only one seq is provided
            elif forbidden_sequences.ndim == 2:
                forbidden_sequences = forbidden_sequences.tolist() # list of sequences
            else:
                raise ValueError(f"Expected a list of tokenized sequences, got tensor with ndim>2 which cannot be converted to List[List[int]]")
        self.forbidden_sequences_ids = forbidden_sequences  
        context_and_last_token_pairs = [(tuple(seq[:-1]), seq[-1])
                                        for seq in self.forbidden_sequences_ids]
        self.context_to_forbidden_tokens = dictOfLists(context_and_last_token_pairs)
            
    def __call__(self, input_ids, scores):
        """
        This method will be called during each step of the beam search algorithm. 
        The method takes as input the input_ids sequence of the partially generated beam and the scores of the next possible tokens.
        By manipulating these scores based on the tokens present in the input_ids, we can control the structure of the generated sentence.
        """
        mask = torch.zeros(scores.shape) # start with all ones - all allowed
        # put zeros in allowed tokens
        for beam_index, (beam_input_ids, beam_scores) in enumerate(zip(input_ids, scores)):
            beam_input_ids = tuple(beam_input_ids.tolist())
            # does context so far triggers a constraint?
            if beam_input_ids in self.context_to_forbidden_tokens:
                # derive forbidden next tokens
                forbidden_next_tokens = self.context_to_forbidden_tokens[beam_input_ids]
         
                # in this beam, forbid only `forbidden_next_tokens` 
                mask[beam_index, forbidden_next_tokens] = 1

        # use mask to forbid all tokens that correspond to 1 in mask
        scores = scores.masked_fill(mask.bool().to(scores.device), -float("inf"))
        return scores