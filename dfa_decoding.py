from typing import List, Iterable, Tuple, Any, Union, Dict, Optional, Callable, Hashable
from collections import UserDict, defaultdict
from copy import deepcopy

import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, 
    T5TokenizerFast, 
    AutoModelForCausalLM, 
    LogitsProcessor, 
    AutoModelForSeq2SeqLM
)

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

State = Any
Alphabet = str
Transitions = Dict[State, Dict[Alphabet, State]]

def find_available_prefix(existing_names: Iterable[State]) -> str:
    prefixes = [f"<{i}>_" for i in range(1,100000)]
    for prefix in prefixes:
        if not any(str(name).startswith(prefix) for name in existing_names):
            return prefix
    raise ValueError("available prefix not found")

class DFA(UserDict):
    """
    Deterministic Finite Automaton.
    Defined as a special subclass of dict, referring to the automaton's `transitions`.
    Transition info (`self.data`) is in the shape: Dict[State, Dict[Alphabet, State]].
    
    Notice it is a DFA and not an NFA because internal dictionaries can only have one value (target state) per key (input symbol).   
    """
    WILDCARD: Alphabet = "***@Wild-Card@***" # one can use DFA.WILDCARD as a wildcard symbol to allow every symbol
    def __init__(self, 
                 transitions: Transitions, 
                 s0: State, 
                 accept_states: Optional[Iterable[State]] = None):
        super().__init__()
        self.data = deepcopy(transitions)  # refer to self as a wrapper of transitions
        self.s0 = s0
        # add all states to transitions (if not specified, add with empty dict)
        self.states = transitions.keys() | {target_state 
                                            for source_state in transitions  
                                            for target_state in transitions[source_state].values()}
        self.data.update({s: dict() for s in self.states - self.data.keys()})
        # by default, all states are accpeting
        if accept_states is None:
            accept_states = self.states
        self.accept_states = set(accept_states)
        # the `tokenizer` attribute would only be assigned for DFAs initialized by `DFA.adjust_for_tokenizer`
        self.tokenizer = None
    
    def copy(self) -> 'DFA':
        return DFA(self.data, self.s0, self.accept_states.copy())
    
    def add_transition(self, src_state, symbol, dest_state):
        if src_state in self:
            self[src_state][symbol] = dest_state
        else:
            self[src_state] = {symbol: dest_state}

    # def add_transition_no_override(self, src_state, symbol, dest_state):
    #     if symbol not in self[src_state]:
    #         self[src_state][symbol] = dest_state
    #     else:
    #         if dest_state == self[src_state][symbol]:   # no collision
    #             return
    #         # handle collision by adding a merged state, which its out-transitions are a union of the `dest_states`s' transitions
    #         orig_dest_state = self[src_state][symbol]
    #         orig_dest_transitions = self[orig_dest_state]
    #         dest_transitions = self[dest_state]
    #         if orig_dest_transitions == dest_transitions: # this means the two destination states are equivalent
    #             self[src_state][symbol] = orig_dest_state
    #         else:   
    #             # define a new artificial destination state which unifies all transitions
    #             unified_dest_state = f"{{ {orig_dest_state}|{dest_state} }}"
    #             self.states.add(unified_dest_state)
    #             self[src_state][symbol] = unified_dest_state
    #             # unify destinations' transitions 
    #             self[unified_dest_state].update(dict(orig_dest_transitions))
    #             # to avoid new collisions, recurse
    #             for symb, state in dest_transitions.items():
    #                 self.add_transition_no_override(unified_dest_state, symb, state) 
                    
    def __getitem__(self, key: State) -> Dict[Alphabet, State]:
        # generate new dict for every new key
        if key not in self.data:
            self.data[key] = dict()
        return self.data.get(key)
            
            
    def __call__(self, input: Iterable[Alphabet]) -> Tuple[bool, State, bool]:
        """
        Try to run the automaton on a string. 
        Returns: (success, end_state, is_accpeting), where `success` is False iff 
        the automaton cannot execute the string.  
        """
        success, end_state = self.execute_string(self.s0, input)
        is_accpeting = end_state in self.accept_states
        return success, end_state, is_accpeting and success
         
    def execute_string(self, current_state: State, input: Iterable[Alphabet]) -> Tuple[bool, State]:
        """
        A Recursive function which runs the automaton from a specific state on a string.
        Returns: (success, end_state), where `success` is False iff 
        the automaton cannot execute the string.
        """
        if current_state not in self.states:
            raise ValueError(f"{current_state} is not a state in this automaton")   
        
        # exhausted the input - finish execution
        if len(input) == 0:
            return True, current_state
        
        next_input_symbol, rest_input = input[0], input[1:]
        success, next_state = self.step(current_state, next_input_symbol)
        
        if not success:
            return success, current_state    
        return self.execute_string(next_state, rest_input) 
    
    def step(self, current_state: State, input_symbol: Alphabet) ->  Tuple[bool, State]:
        """
        Run one step on the automaton. 
        Assume automaton is in `current_state`, and try to execute `input_symbol`.
        Returns: (success, end_state), where `success` is False iff 
        the automaton cannot execute the string. 
        """
        possible_transitions = self[current_state]
        if input_symbol not in possible_transitions and DFA.WILDCARD not in possible_transitions:
            # Log? return the state where the automaton failed
            return False, current_state
        
        if input_symbol in possible_transitions:
            next_state = possible_transitions[input_symbol]
        else: # DFA.WILDCARD in possible_transitions
            next_state = possible_transitions[DFA.WILDCARD]
        return True, next_state
    
    def adjust_for_tokenizer(self, tokenizer, inplace=False, convert_to_word_ids=False) -> 'DFA':
        """ 
        Adjust the DFA alphabet to fit tokenizer vocabulary. 
        
        :param convert_to_word_ids (bool, default `False`): is `True`, converts the automaton's alphabet to word_ids (integers). 
        
        Implementation:
        1. Replace automaton's alphabet with corresponding vocabulary entries. 
        2. If automaton's transitions' alphabet include out-of-vocabulry symbols, replace symbols with tokenizer.unk_token within the automaton's transitions.     
        3. If automaton's transitions' alphabet include multi-token symbols, fix by adding "bridge states" between the word's subtokens.
        4. For all accepting states, add transition to a "end-of-sequence" state which will allow for the eos_token;
            From there, only allow a self-referring transition with a "pad" token (padding the rest of the output).
        """
        dfa = self if inplace else self.copy() 
        invariant = self.copy() if inplace else self    # a non-modified version of the DFA for iteration    
        collision_info = defaultdict(set)
        def add_token_transition(src_state, token_symbol, dest_state):
            if convert_to_word_ids:
                token_symbol = tokenizer.convert_tokens_to_ids(token_symbol)
            dfa.add_transition(src_state, token_symbol, dest_state)
            collision_info[(src_state, token_symbol)].add(dest_state)
        
        for src_state, transitions in reversed(list(invariant.items())):
            for symbol, tgt_state in transitions.copy().items():
                if symbol == DFA.WILDCARD:
                    continue
                # tok_symbol = list(map(get_tok_surface, tokenizer.tokenize(symbol)))
                # new_symbol = symbol
                tok_symbol = tokenizer.tokenize(symbol)
                if len(tok_symbol) == 0:
                    tok_symbol = [symbol]
                if len(tok_symbol) == 1:
                    new_symbol = tok_symbol[0]                    
                # if new_symbol in tokenizer.vocab:
                    # 2. handle OOV symbols
                    if new_symbol == tokenizer.unk_token:
                        # Notice that this may cause problems, in cased there are multiple unknown (i.e OOV) symbols per src_state - they 
                        # will collide into the same transition, so one of them will be overriden. 
                        # TODO find a solution for these cases  
                        assert tokenizer.unk_token not in dfa[src_state],   f"""transition from {src_state} to {dfa[src_state][tokenizer.unk_token]} is overriden,  
                                as more than one symbol from {invariant[src_state].keys()} is out-of-vocabulary and should be replaced with {tokenizer.unk_token}."""
                    # 1. Replace alphabet with vocab entry
                    if new_symbol != symbol or convert_to_word_ids:
                        dfa[src_state].pop(symbol)
                        add_token_transition(src_state, new_symbol, tgt_state)
                    
                # 3. handle multitoken symbols
                else:
                    assert len(tok_symbol) > 0, f"DFA symbol {symbol} is tokenized to empty list of tokens"
                    dfa[src_state].pop(symbol)
                    
                    # instead, add unique transition between sub-tokens using new "bridge" states
                    
                    # Try Backward
                    iter_state = tgt_state 
                    for i in range(len(tok_symbol)-1,0,-1):
                        subtok1, subtok2 = tok_symbol[i-1], tok_symbol[i]
                        intermediate_state = f"{src_state}:{subtok1}~>{subtok2}"
                        dfa.states.add(intermediate_state)
                        add_token_transition(intermediate_state, subtok2, iter_state)
                        # iteration step - move iter_state forward
                        iter_state = intermediate_state 
                    # and then transit from last subtoken to original destination state.
                    # (notice that in end of iteration, subtok2 == tok_symbol[-1])
                    add_token_transition(src_state, subtok1, iter_state)
                    
                    # iter_state = src_state 
                    # for i in range(len(tok_symbol)-1):
                    #     subtok1, subtok2 = tok_symbol[i], tok_symbol[i+1]
                    #     intermediate_state = f"{src_state}:{subtok1}~>{subtok2}"
                    #     dfa.states.add(intermediate_state)
                    #     transition_triplets_to_add.append((iter_state, subtok1, intermediate_state))
                    #     # iteration step - move iter_state forward
                    #     iter_state = intermediate_state 
                    # # and then transit from last subtoken to original destination state.
                    # # (notice that in end of iteration, subtok2 == tok_symbol[-1])
                    # transition_triplets_to_add.append((iter_state, tok_symbol[-1], tgt_state))

            
        # 4. For all accepting states, add transition to a "end-of-sequence" state which will allow for the eos_token;
        #    From there, have a self-referring "pad" transition (padding the rest of the output).
        eos_state = "<end-of-sequence-state>"
        dfa.states.add(eos_state)
        for accpet_state in dfa.accept_states:
            add_token_transition(accpet_state, tokenizer.eos_token, eos_state) # can only get to eos_state after reading eos_token
        add_token_transition(eos_state, tokenizer.pad_token, eos_state) # cycle to pad the rest of the sequence  
        dfa.accept_states.add(eos_state)
        
        # now handle collisions
        collisions = [(src, symb, dests) 
                      for (src, symb), dests in collision_info.items() 
                      if len(dests)>1]
        for (src, symb, dests) in collisions:
            # make a unique new state for (src, symb) which its forawrd transitions will be the union of the dests' transitions
            merge_state = f"{{ {'|'.join(str(state) for state in dests)} }}"
            dfa.states.add(merge_state)
            dfa[src][symb] = merge_state
            # unify destinations' transitions 
            transitions_of_dests = dict()
            for dest in dests:
                transitions_of_dests.update(dfa[dest])
                # instead, for debug - to make sure no collision
                # for symb2, dest2 in dfa[dest].items():
                #     internal_collision = symb2 in transitions_of_dests
                #     if internal_collision:
                #         print(f"collision in {src} {symb}: for {symb2}, {dest2} overrides {transitions_of_dests[symb2]}.")
                #     transitions_of_dests[symb2] = dest2 
            dfa[merge_state].update(transitions_of_dests)
            
        
        # keep tokenizer on dfa instance 
        dfa.tokenizer = tokenizer
        return dfa
    
    def as_cyclic(self, end_states: Optional[Iterable[State]] = None, bridge: Optional[Alphabet] = None) -> 'DFA':
        """
        Return a "cyclic" copy of the DFA. Cycle is composed by merging or connecting `end_states` with `self.s0`.
        :param end_states (optional): the states to merge or connect with self.s0. 
            By default selects all the accept states as end states. 
        :param bridge (optional): if None (default), cycle is composed by merging the end_states with the initial state
             to be the same state (would work best for a single end_state with out-degree=0).
            If provided, `bridge` is used as the alphabet symbol on the new connecting transitions 
             from `end_states` to `self.s0`.      
        """
        dfa = self.copy()
        end_states = end_states or dfa.accept_states
        if bridge is None:
            for end_state in end_states:
                dfa.rename_state(end_state, dfa.s0, merging_existing_states=True)
        else:
            for end_state in end_states:
                dfa.add_transition(end_state, bridge, dfa.s0)
        return dfa
            
    def rename_state(self, state: State, new_name: State, merging_existing_states=False):
        if state == new_name:
            return
        assert state in self.states, f"state {state} not in states. DFA: {self}"
        if not merging_existing_states:
            assert new_name not in self.states, f"new name {new_name} already in states. DFA: {self}"
        
        # rename target states
        for src_state, transitions in self.items():
            for symbol, tgt_state in transitions.copy().items():
                if tgt_state == state:
                    transitions.pop(symbol)
                    transitions[symbol] = new_name
        # rename src state
        if state in list(self.data):
            transitions = self.data.pop(state)
            if new_name not in self:
                self.data[new_name] = transitions
            else:   # merge `state`'s transitions into `new_name`'s transitions
                self.data[new_name].update(transitions)
        # update self.states, self.s0, self.accept_states
        self.states.remove(state)
        self.states.add(new_name)
        if self.s0 == state:
            self.s0 = new_name
        if state in self.accept_states:
            self.accept_states.remove(state)
            self.accept_states.add(new_name)         
    
    @staticmethod
    def from_slots(slots: List[List[Alphabet]] ) -> 'DFA':
        dfa_as_dict = {i: {possible_token : i+1
                        for possible_token in possible_tokens} 
                    for i, possible_tokens in enumerate(slots)}
        return DFA(dfa_as_dict, 0, accept_states={len(slots)})
    
    @staticmethod
    def concat_two(dfa1: 'DFA', dfa2: 'DFA', 
                   bridge_symbol: Optional[Alphabet] = None, 
                   tie_by_merge: Optional[bool] = None) -> 'DFA':
        if tie_by_merge is None:
            tie_by_merge = bridge_symbol is None
            
        # rename dfa2 states to avoid collision
        result_dfa = dfa1.copy()
        dfa2_to_merge = dfa2.copy()
        state_prefix = find_available_prefix(dfa1.states)
        for state in dfa2.states:
            dfa2_to_merge.rename_state(state, f"{state_prefix}{state}")
        # merge transitions and states
        result_dfa.data = dict(result_dfa, **dfa2_to_merge) 
        result_dfa.states.update(dfa2_to_merge.states)  
        # connect accpet-states of dfa1 to dfa2.s0, either by merging them, either by bridge transitions 
        if tie_by_merge:
            for left_accept_state in dfa1.accept_states:
                result_dfa.rename_state(left_accept_state, dfa2_to_merge.s0, merging_existing_states=True)
        else:
            for left_accept_state in dfa1.accept_states:
                result_dfa.add_transition(left_accept_state, bridge_symbol, dfa2_to_merge.s0)
        # set accpet states
        result_dfa.accept_states = dfa2_to_merge.accept_states
        
        return result_dfa
    
    @staticmethod
    def concat(dfas: List['DFA'], 
               bridge: Optional[Union[Alphabet, List[Alphabet]]] = None) -> 'DFA':
        """
        Helper method to concatenate a list of DFAs. 
        :param dfas: a sequence of DFAs to concatenate to eac other.
        :param bridge: the alphabet symbol that allows transition from dfa to dfa.
            Can be a single symbol, or a list, sized as len(dfas)-1, to have a unique transition symbol for each pair of adjacent DFAs.     
        Implementation: Accepting states of dfas[i] are bind to transit into dfa[i+1].s0 through the `bridge` symbol.
        """
        if len(dfas) == 0:
            return None
        elif len(dfas) == 1:
            return dfas[0]
        
        # iteratively concatenate two adjacent DFAs
        accum_result_dfa = dfas[0].copy()
        for i in range(len(dfas)-1):
            if bridge is None:
                accum_result_dfa = DFA.concat_two(accum_result_dfa, dfas[i+1], tie_by_merge=True)
            if isinstance(bridge, list) and len(bridge) == len(dfas)-1:
                accum_result_dfa = DFA.concat_two(accum_result_dfa, dfas[i+1], bridge_symbol=bridge[i])
            else:
                accum_result_dfa = DFA.concat_two(accum_result_dfa, dfas[i+1], bridge_symbol=bridge)
        return accum_result_dfa       
    

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
            success, current_state, in_accept_state = self.dfa(b_input_tokens)
            # success should usually be true, because we are banning invalid tokens in previous steps
            # but for cases like going through the white space tokens, we block this thread post hoc
            if not success:
                # print(f"Warning: Decoding with {b_input_tokens} deviated from DFA's transitions for state {current_state}") 
                allowed_next_words = set() # ban all vocab
            # 2. Determine allowed next-tokens
            else:
                allowed_next_words = set(self.dfa[current_state].keys())
                # add special "empty" / white-space tokens
                allowed_next_words.update({'▁', ' '})
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


def test_DFA():
    # example from https://stackoverflow.com/questions/35272592/how-are-finite-automata-implemented-in-code/35279645
    dfa_d={ 0:{'0':0, '1':1},
            1:{'0':2, '1':0},
            2:{'0':1, '1':2}}
    dfa = DFA(dfa_d, 0, accept_states={0})
    assert dfa('1011101') == (True, 0, True)
    assert dfa('10111011') == (True, 1, False)
    assert dfa('102') == (False, 2, False)

def test_concat_dfas():
    # example from https://stackoverflow.com/questions/35272592/how-are-finite-automata-implemented-in-code/35279645
    dfa_d={ 0:{'a':1, 'b':2, '0':0},
            1:{'a':2, '0':1},
            2:{'0':2, 'r':0},
            3:{'l':3}}
    dfa1 = DFA(dfa_d, 0, accept_states={2})
    dfa2 = DFA(dfa_d, 2, accept_states={0})
    dfa3 = DFA(dfa_d, 3, accept_states={3})
    result_dfa = DFA.concat([dfa1, dfa2, dfa3], '~')
    
    assert result_dfa('aarb~r~lll') == (True, '<2>_3', True)
    return result_dfa
    

def test_DFA_Decoding():
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
    

if __name__ == "__main__":
    test_DFA()
    # test_DFA_Decoding()
    # dfa = test_concat_dfas()
    dfa_d={ 0:{'a':1, 'b':2},
            1:{'a':2, 'l':1},
            2:{'r':1, 'e':3}}
    dfa1 = DFA(dfa_d, 0, accept_states={3})

    # result_dfa = dfa1.as_cyclic(bridge = '~')