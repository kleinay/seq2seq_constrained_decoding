"""
LogitProcessors for mostly simple "X should/n't follow Y" constraints.
"""
from typing import List, Iterable, Tuple, Any, Union, Dict, Optional, Callable

import torch
from transformers import LogitsProcessor


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
    def __init__(self, tokenizer, context: str, constraint: str):
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




if __name__ == "__main__":
    def strip_prefix(s, prefix) -> str:
        if s.startswith(prefix):
            s = s[len(prefix):]
        return s
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    default_generate_kwargs = dict(do_sample=True,
                                temperature=1, 
                                # max_length=200, 
                                top_p=0.8,
                                num_return_sequences=1)
    class GPT2():
        def __init__(self, cuda=None, **gen_kwargs):
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.cuda=cuda
            self.default_generate_kwargs = default_generate_kwargs.copy()
            self.default_generate_kwargs.update(gen_kwargs)
            
            
        def run(self, prompt: str, **gen_kwargs) -> List[str]:
            "Simple textual model-generate function "
            tokenized_prompt = self.tokenizer(prompt, return_tensors="pt")
            input_ids = tokenized_prompt.input_ids
            attention_mask = tokenized_prompt.attention_mask
            if self.cuda:
                input_ids = input_ids.to('cuda')
                attention_mask = attention_mask.to('cuda')
            gen_kwargs = dict(self.default_generate_kwargs, **gen_kwargs)   # integrate self defaults
            gen_tokens = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                **gen_kwargs
            )
            gen_texts = self.tokenizer.batch_decode(gen_tokens) 
            gen_texts = [strip_prefix(gen_text, prefix=prompt) for gen_text in gen_texts]
            return gen_texts
    
    gpt = GPT2()
    # constrainer = XMustFollowYLogitsProcessor(gpt.tokenizer, "Inspiration:", "a dog walked")
    constrainer1 = XMustNotFollowYLogitsProcessor(gpt.tokenizer, "\nInspiration:", "\n")
    constrainer2 = XMustNotFollowYLogitsProcessor(gpt.tokenizer, "\nInspiration:", " cat")

    pipeline_associate_generate_prompt = """Provide funny responses to surreal shopping requests. Choose a concept that is related to the shopping request, and use it as Inspiration for the funny response.

Shopping Request: buy hornets.
Inspiration: drones
Funny response using Inspiration: I'd sell you a hornet, but we're still working on drone delivery.

Shopping Request: buy sleep.
Inspiration: night
Funny response using Inspiration: Sorry, I don't think the delivery guys work during the night.

Shopping Request: buy a pandas.
Inspiration: black and white
Funny response using Inspiration: The rules are black and white on this one.

Shopping Request: buy time.
Inspiration: present
Funny response using Inspiration: I wish I could. but by the time I wrap the present, it turns into the past.

Shopping Request: buy cat.
Inspiration:"""

    responses = gpt.run(pipeline_associate_generate_prompt, num_return_sequences=3, max_length=320, 
                                            logits_processor= [constrainer1, constrainer2],
                                            top_p=0.9)
    print(responses)