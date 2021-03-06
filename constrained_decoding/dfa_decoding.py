from typing import List, Iterable, Tuple, Any, Union, Dict, Optional, Callable

import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    T5TokenizerFast, 
    LogitsProcessor, 
)

from .dfa import DFA
from .dfa_constrained_beam_search import dfa_constrained_beam_search
from .dfa_constrained_generate import dfa_constrained_generate
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

def vectorized_exclusion_func(excluded_from: Iterable[Any]):
    return np.frompyfunc(lambda x: x not in excluded_from, 1, 1)
   

class DfaDecodingLogitsProcessor(LogitsProcessor):
    """
    Decode following a given Deterministic Finite Automaton (DFA).
    We assume the vocabulary of the state machine is comprised of tokens; 
    each decoded token corresponds to a transition on the automaton.
    Note: the decoding algorithm can be set to not consider "accpeting" states altogether, by 
     setting `enforce_accept_state` to `False`; but by default , it gurantees that the output is both 
     valid according to the DFA (i.e. that executing the DFA on the output would be successful), and 
     is accepted by it (i.e. that the DFA execution on output terminates in an accepting state).  
    Implementation: at each step of the beam search, run the automata om previous tokens occuring in output, 
     to retrieve the current DFA state. Then ban all tokens except those permitted by the DFA's transitions.
     If current state is an accpeting state (and `enforce_accept_state`==True), add the "end-of-sequence" token 
     to allowed tokens. 
     
    Note: Using this class is probably not computationally optimal.
    You can also achieve dfa-constrained decoding using `set_decoding_to_dfa_constrained` (below).
        
    """
    def __init__(self, tokenizer, dfa: DFA, enforce_accept_state: bool = True):
        self.tokenizer = tokenizer
        adjusted_dfa = dfa.adjust_for_tokenizer(tokenizer)
        self.orig_dfa = dfa
        self.dfa = adjusted_dfa
        self.enforce_accept_state = enforce_accept_state
        
        self.vocab_words = np.array(list(tokenizer.vocab.keys()))
        self.vocab_word_ids = np.array(list(tokenizer.vocab.values()))

        
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
            # 1. run DFA on output-so-far
            b_input_tokens = self.tokenizer.convert_ids_to_tokens(beam_input_ids, skip_special_tokens=True)
            current_state_iterator = self.dfa.iterator()
            success, end_state, in_accept_state = current_state_iterator(b_input_tokens)
            # success should usually be true, because we are banning invalid tokens in previous steps
            # but for cases like going through the white space tokens, we block this thread post hoc
            if not success:
                # print(f"Warning: Decoding with {b_input_tokens} deviated from DFA's transitions for state {current_state}") 
                allowed_next_words = set() # ban all vocab
            # 2. Determine allowed next-tokens
            else:
                allowed_next_words = set(current_state_iterator.get_allowed_transitions().keys())
                # add special "empty" / white-space tokens
                allowed_next_words.update({'???', ' '})
                # eos token is explictly allowed if this is an accepting state or if we shouldn't enforce accpetance
                if in_accept_state or not self.enforce_accept_state:
                    allowed_next_words.add(self.tokenizer.eos_token)
                   
            # vectorize to apply on array of ids
            is_forbidden = vectorized_exclusion_func(allowed_next_words)  # numpy vectorized function --  x not in allowed_next_words
            banned_indexes = np.where(is_forbidden(self.vocab_words)) # get banned-words indexes in self.vocab_words, corresponding to indexes in self.vocab_word_ids
            batch_banned_token_ids = self.vocab_word_ids[banned_indexes]
            
            banned_tokens.append(batch_banned_token_ids)
        scores = set_scores_to_inf_for_banned_tokens(scores, banned_tokens)
        return scores 
    

def test_DFA_Decoding_LogitsProcessor():
    tokenizer = T5TokenizerFast.from_pretrained("t5-small")
    dfa_d={ 0:{'my':1, 'the':1},
            1:{'name':2, 'dog':2},
            2:{'of':0, 'is': 3, 'was':1},
            3:{'John':4, 'complication':4}}
    dfa = DFA(dfa_d, 0, accept_states=[4])
    text = "the name is my dog of good complication John " 
    # tokenizer.vocab = tokenizer.get_vocab() # {'I': 4, 'M': 5, 'T': 6, 'B': 2, 'A': 1, '#': 0, 'C': 3}
    text_ids = tokenizer(text).input_ids
    import torch
    input_ids = torch.tensor([text_ids[:3]])

    # inputs = tokenizer(text)
    scores = torch.tensor([[1/len(tokenizer.get_vocab())] * len(tokenizer.get_vocab())])
    
    processor = DfaDecodingLogitsProcessor(tokenizer, dfa)
    print(f"\nScores before: { {tokenizer.convert_ids_to_tokens(i): scores[0][i] for i in set(text_ids)} }")
    new_scores = processor(input_ids, scores)
    
    print(f"\nScores After : { {tokenizer.convert_ids_to_tokens(i): new_scores[0][i] for i in set(text_ids)} }")
    
    # test de-facto
    from transformers import (
    LogitsProcessorList,
    AutoModelForCausalLM, LogitsProcessor, AutoModelForSeq2SeqLM
    )

    # how many beams to track during the Viterbi algorithm
    num_beams = 10
    # how many beams to return after the algorithm
    num_return_beams = 10

    # the prompt to continue
    # input_seq = 'summarize: Lately I feel that I very much enjoy walking with my cute litte dog from time to time .'
    input_seq = 'summarize: ' + text

    # tokenizing the prompt

    input_ids = tokenizer.encode(input_seq, return_tensors='pt')

    # instantiating a list of LogitsProcessor instances
    # using our custom ABCLogits class
    logits_processors = LogitsProcessorList([processor])

    # running beam search using our custom LogitsProcessor
    model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')
    generated = model.generate(input_ids, 
                            max_length=20,
                            num_beams=num_beams,
                            num_return_sequences=num_return_beams,
                            logits_processor=logits_processors) 
    for index, output_tokenized in enumerate(generated):
        output = tokenizer.decode(output_tokenized)
        print(f'beam {index}: {output}')
    

def set_decoding_to_dfa_constrained(model, 
                                    dfa: Optional[DFA] = None, 
                                    dfa_factory: Optional[Callable[[List[int], ], DFA]] = None, 
                                    dfas: Optional[List[DFA]] = None,
                                    dfa_factories: Optional[List[Callable[[List[int], ], DFA]]] = None,
                                    tokenizer=None):
    """ Set the beam search method of the model to be constrained to decoding according to a Deterministic Finite Automaton.
        Either `dfa` or `dfa_factory` must be specified. 
    Args:
        model ([type]): A Huggingface model supporting text generation. The function will modify `model.beam_search`. 
        dfa (Optional[DFA]): When specified, the same DFA is used for all batch instances and beams, regardless of input sequence. Defaults to None.
        dfa_factory (Optional[Callable[[List[int]], DFA]]): When specified, used for instanciating a `DFA` for each batch item.
            Assumed to be a functions which gets `input_ids` (token ids of input sequence) and returns a `DFA`. Defaults to None.
        tokenizer ([type], optional): The instanciated DFAs or provided `dfa` would be adjusted to this tokenizer.
            if `dfa` is provided and `dfa.tokenizer` is not None, `tokenizer` would not be used.

    """
    # Validate arguments
    if not (dfa or dfa_factory or dfas or dfa_factories):
        raise ValueError("Either `dfa`, `dfas`, `dfa_factory` or `dfa_factories` must be provided.")
    from transformers.generation_utils import GenerationMixin
    assert isinstance(model, GenerationMixin), "Model must be an instance of `transformers.generation_utils.GenerationMixin` to be applied the dfa-constrained `beam_search` method."
    
    # Replace model's beam_search method with our custom function as a bound method
    import types
    model.beam_search = types.MethodType(dfa_constrained_beam_search, model)
    # Provide our custom beam_search method with dfa or dfa_factory (as function-object attributes)  
    if dfa or dfas:
        dfas = dfas or [dfa]
        for dfa in dfas:
            # make sure DFA is adjusted to tokenizer
            if dfa.tokenizer is None and tokenizer is not None:
                dfa = dfa.adjust_for_tokenizer(tokenizer, convert_to_word_ids=True)
            elif dfa.tokenizer is None and tokenizer is None:
                raise ValueError("Either `dfa` should be adjusted to model's tokenizer, or `tokenizer` should be provided for adjusting the `dfa` to it.")
        # provide the beam_search method with dfa
        dfa_constrained_beam_search.multiple_dfas = dfas
    elif dfa_factory or dfa_factories:
        dfa_factories = dfa_factories or [dfa_factory]
        dfa_constrained_beam_search.multiple_dfa_factories = dfa_factories
        dfa_constrained_beam_search.tokenizer = tokenizer
    # needs also to replace `model.generate` to our custom generate function that sends `encoder_input_ids` to model.beam_search 
    model.generate = types.MethodType(dfa_constrained_generate, model)
 

if __name__ == "__main__":
    test_DFA_Decoding_LogitsProcessor()
    dfa_d={ 0:{'a':1, 'b':2},
            1:{'a':2, 'l':1},
            2:{'r':1, 'e':3}}
    dfa1 = DFA(dfa_d, 0, accept_states={3})
    result_dfa = dfa1.as_cyclic(bridge = '~')