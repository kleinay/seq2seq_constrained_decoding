"""
LogitProcessors for mostly simple "X should/n't follow Y" constraints.
"""
from typing import List, Iterable, Tuple, Any, Union, Dict, Optional, Callable

import torch
from transformers import LogitsProcessor, PreTrainedTokenizer


def rindex(lst, e):
    "Index of last occurrence of `e` in `lst` "
    return len(lst) - 1 - lst[::-1].index(e)

def sublist_match_indices(lst, sublist):
    "Indices of `sublist` in `lst` "
    match_idxs = [index for index in range(len(lst)) 
                  if lst[index : index + len(sublist)] == sublist]
    return match_idxs

def rindex_sublist(lst, sublist):
    "Index of last occurrence of `sublist` in `lst` "
    match_idxs = sublist_match_indices(lst, sublist)
    if not match_idxs:
        raise ValueError(f"No occurrences of {sublist} in {lst}")
    return match_idxs[-1]

class XMustFollowYLogitsProcessor(LogitsProcessor):
    """
    A LogitsProcessor that constrain the following rule:
        After a sub-sequence identical to `context` (Y), the next token or list of tokens must perfectly match `constraint` (X).     
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, context: str, constraint: str):
        self.tokenizer = tokenizer  
        self.context = context  
        self.constraint = constraint 
        self.context_ids = tokenizer(context).input_ids
        self.context_size = len(self.context_ids)
        self.constraint_ids = tokenizer(constraint).input_ids
            
    def __call__(self, input_ids, scores):
        """
        This method will be called during each step of the beam search algorithm. 
        The method takes as input the input_ids sequence of the partially generated beam and the scores of the next possible tokens.
        By manipulating these scores based on the tokens present in the input_ids, we can control the structure of the generated sentence.
        """
        mask = torch.zeros(scores.shape) # start with all zeros - all allowed
        # put zeros in allowed tokens
        for beam_index, (beam_input_ids, beam_scores) in enumerate(zip(input_ids, scores)):
            beam_input_ids = beam_input_ids.tolist()
            context_occurrence_idxs = sublist_match_indices(beam_input_ids, self.context_ids)
            if context_occurrence_idxs:     # context condition is met somwhere in the past
                idx_start_context = context_occurrence_idxs[-1]   # get most recent occurrence
                full_relevant_context = beam_input_ids[idx_start_context:] # including the prefix of `constraint`
                requirement: List[int] = (self.context_ids + self.constraint_ids)[:len(full_relevant_context)] # 
                if full_relevant_context == requirement and len(requirement) < len(self.context_ids + self.constraint_ids): # we have a match - need to constrain next token
                    obligated_next_token_id = (self.context_ids + self.constraint_ids)[len(full_relevant_context)] 
                    # in this beam, all other tokens excpet `obligated_next_token_id` should be forbidden
                    mask[beam_index, :] = 1
                    mask[beam_index, obligated_next_token_id] = 0

        # use mask to forbid all tokens that correspond to 1 in mask
        scores = scores.masked_fill(mask.bool().to(scores.device), -float("inf"))
        return scores

class XMustNotFollowYLogitsProcessor(LogitsProcessor):
    """
    A LogitsProcessor that constrain the following rule:
        After a sub-sequence identical to `context` (Y), the next token or list of tokens must *NOT* match `constraint` (X).     
    """
    def __init__(self, tokenizer, context: str, constraint: str):
        self.tokenizer = tokenizer  
        self.context = context  
        self.constraint = constraint 
        self.context_ids = tokenizer(context).input_ids
        self.context_size = len(self.context_ids)
        self.constraint_ids = tokenizer(constraint).input_ids
        self.combined_ids = tokenizer(context + constraint).input_ids
            
    def __call__(self, input_ids, scores):
        """
        This method will be called during each step of the beam search algorithm. 
        The method takes as input the input_ids sequence of the partially generated beam and the scores of the next possible tokens.
        By manipulating these scores based on the tokens present in the input_ids, we can control the structure of the generated sentence.
        """
        mask = torch.zeros(scores.shape) # start with all zeros - all allowed
        # put zeros in allowed tokens
        for beam_index, (beam_input_ids, beam_scores) in enumerate(zip(input_ids, scores)):
            beam_input_ids = beam_input_ids.tolist()
            context_occurrence_idxs = sublist_match_indices(beam_input_ids, self.context_ids)
            if context_occurrence_idxs:     # context condition is met somwhere in the past
                idx_start_context = context_occurrence_idxs[-1]   # get most recent occurrence
                full_relevant_context = beam_input_ids[idx_start_context:] # including the prefix of `constraint`
                requirement: List[int] = (self.context_ids + self.constraint_ids)[:len(full_relevant_context)] # 
                if full_relevant_context == requirement and len(requirement) < len(self.context_ids + self.constraint_ids): # we have a match - need to constrain next token
                    forbidden_next_token_id = (self.context_ids + self.constraint_ids)[len(full_relevant_context)] 
                    # in this beam, `forbidden_next_token_id` should be forbidden
                    mask[beam_index, forbidden_next_token_id] = 1
            # context_occurrence_idxs = sublist_match_indices(beam_input_ids, self.context_ids)
            # if context_occurrence_idxs:     # context condition is met somwhere in the past
            #     idx_start_context = context_occurrence_idxs[-1]   # get most recent occurrence
            #     full_relevant_context = beam_input_ids[idx_start_context:] # including the prefix of `constraint`
            #     requirement: List[int] = (self.context_ids + self.constraint_ids)[:len(full_relevant_context)] # 
            #     if full_relevant_context == requirement and len(requirement) < len(self.context_ids + self.constraint_ids): # we have a match - need to constrain next token
            #         forbidden_next_token_id = (self.context_ids + self.constraint_ids)[len(full_relevant_context)] 
            #         # in this beam, `forbidden_next_token_id` should be forbidden
            #         mask[beam_index, forbidden_next_token_id] = 1

        # use mask to forbid all tokens that correspond to 1 in mask
        scores = scores.masked_fill(mask.bool().to(scores.device), -float("inf"))
        return scores
