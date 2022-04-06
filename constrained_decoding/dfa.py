from typing import List, Iterable, Tuple, Any, Union, Dict, Optional, Callable, Hashable
from collections import UserDict, defaultdict
from copy import deepcopy

State = Any
Alphabet = str
Transitions = Dict[State, Dict[Alphabet, State]]
   

def find_available_prefix(existing_names: Iterable[State]) -> str:
    prefixes = [f"<{i}>_" for i in range(1,100000)]
    for prefix in prefixes:
        if not any(str(name).startswith(prefix) for name in existing_names):
            return prefix
    raise ValueError("available prefix not found")


class DFAIterator():
    """
    A "running automaton" object - keeps current state along with wrapping the DFA.
    
    To attain an unbound-memory automaton (computationally equivelant to a Turing machine), that can for example count repetitions etc.,
     one can extend this class with custom behaviour.
    """
    def __init__(self, dfa: 'DFA') -> None:
        self.dfa = dfa
        self.current_state = self.dfa.s0
    
    def __call__(self, input: Iterable[Alphabet]) -> Tuple[bool, State, bool]:
        """
        Try to run the automaton on a string. 
        Returns: (success, end_state, is_accpeting), where `success` is False iff 
        the automaton cannot execute the string.  
        """
        success, next_state = True, self.current_state
        for input_symbol in input:
            success, next_state = self.step(input_symbol)
            if not success:
                return success, next_state, False 
        
        is_accpeting = next_state in self.dfa.accept_states
        return success, next_state, is_accpeting 
             
    def get_allowed_transitions(self) -> Dict[Alphabet, State]:
        " Override this method to achieve custom behavior. "
        return self.dfa[self.current_state]
    
    def step(self, input_symbol: Alphabet) -> Tuple[bool, State]:
        """
        Move one step on the automaton. 
        Assume automaton is in `self.current_state`, and try to execute `input_symbol`.
        Returns: (success, end_state), where `success` is False iff input_symbol is invalid.
        the automaton cannot execute the string. 
        """
        possible_transitions = self.get_allowed_transitions()
        if input_symbol not in possible_transitions and DFA.WILDCARD not in possible_transitions:
            # Log? return the state where the automaton failed
            return False, self.current_state
        if input_symbol in possible_transitions:
            next_state = possible_transitions[input_symbol]
        else: # DFA.WILDCARD in possible_transitions
            next_state = possible_transitions[DFA.WILDCARD]  
        
        self.current_state = next_state
        return True, next_state    

    def copy(self, copy_dfa=False):
        """ 
        Returns a copy of the iterator; with the option of (not) copying the DFA.
        Override when extending `DFAIterator` to keep additional internal knowledge. 
        """
        if copy_dfa:
            _copy = DFAIterator(self.copy())
        else:
            _copy = DFAIterator(self.dfa)
        _copy.current_state = self.current_state
        return _copy

    def __repr__(self):
        return f"@state: {repr(self.current_state)}"
 
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
        # iterator factory - so that it would be easy to replace the default `DFAIterator` class with a subclass having unbound custom behavior
        self._iterator_factory = DFAIterator 
        # the `tokenizer` attribute would only be assigned for DFAs initialized by `DFA.adjust_for_tokenizer`
        self.tokenizer = None
    
    def copy(self) -> 'DFA':
        return DFA(self.data, self.s0, self.accept_states.copy())
    
    def add_transition(self, src_state, symbol, dest_state):
        if src_state in self:
            self[src_state][symbol] = dest_state
        else:
            self[src_state] = {symbol: dest_state}
        self.states.update([src_state, dest_state])
                    
    def __getitem__(self, key: State) -> Dict[Alphabet, State]:
        # generate new dict for every new key
        if key not in self.data:
            self.data[key] = dict()
        return self.data.get(key)
    
    def set_iterator_factory(self, iterator_factory):
        assert isinstance(iterator_factory(self), DFAIterator), "new iterator class must be a subclass of `DFAIterator`"
        self._iterator_factory = iterator_factory
    
    def iterator(self) -> 'DFAIterator':
        return self._iterator_factory(self)        
            
    def __call__(self, input: Iterable[Alphabet]) -> Tuple[bool, State, bool]:
        """
        Try to run the automaton on a string. 
        Returns: (success, end_state, is_accpeting), where `success` is False iff 
        the automaton cannot execute the string.  
        """
        dfa_iterator = self.iterator()
        return dfa_iterator(input)
             
    def step(self, current_state: State, input_symbol: Alphabet) ->  Tuple[bool, State]:
        """
        Compute one step on the automaton. 
        Assume automaton is in `current_state`, and try to execute `input_symbol`.
        Returns: (success, end_state), where `success` is False iff 
        the automaton cannot execute the string. 
        """
        dfa_iterator = self.iterator()
        dfa_iterator.current_state = current_state
        return dfa_iterator.step(input_symbol)
    
    def adjust_for_tokenizer(self, tokenizer, inplace=False, convert_to_word_ids=False, set_eos_loop=True) -> 'DFA':
        """ 
        Adjust the DFA alphabet to fit tokenizer vocabulary. 
        
        :param convert_to_word_ids (bool, default `False`): is `True`, converts the automaton's alphabet to word_ids (integers). 
        :
        
        Implementation:
        1. Replace automaton's alphabet with corresponding vocabulary entries. 
        2. If automaton's transitions' alphabet include out-of-vocabulry symbols, replace symbols with tokenizer.unk_token within the automaton's transitions.     
        3. If automaton's transitions' alphabet include multi-token symbols, fix by adding "bridge states" between the word's subtokens.
        4. (if `set_eos_loop`:) For all accepting states, add transition to a "end-of-sequence" state which will allow for the eos_token;
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
                    dfa[src_state].pop(symbol)
                    add_token_transition(src_state, new_symbol, tgt_state)
                    
                # 3. handle multitoken symbols
                else:
                    assert len(tok_symbol) > 0, f"DFA symbol {symbol} is tokenized to empty list of tokens"
                    dfa[src_state].pop(symbol)
                    
                    # instead, add unique transition between sub-tokens using new "bridge" states
                    
                    # Run Backward
                    iter_state = tgt_state 
                    for i in range(len(tok_symbol)-1,0,-1):
                        subtok1, subtok2 = tok_symbol[i-1], tok_symbol[i]
                        intermediate_state = f"{src_state}:{subtok1}~>{subtok2}"
                        dfa.states.add(intermediate_state)
                        add_token_transition(intermediate_state, subtok2, iter_state)
                        # iteration step - move iter_state backward
                        iter_state = intermediate_state 
                    # and then transit to first subtoken from original source state.
                    add_token_transition(src_state, subtok1, iter_state)
                    
                    # iter_state = src_state 
                    # for i in range(len(tok_symbol)-1):
                    #     subtok1, subtok2 = tok_symbol[i], tok_symbol[i+1]
                    #     intermediate_state = f"{src_state}:{subtok1}~>{subtok2}"
                    #     dfa.states.add(intermediate_state)
                    #     add_token_transition(iter_state, subtok1, intermediate_state)
                    #     # iteration step - move iter_state forward
                    #     iter_state = intermediate_state 
                    # # and then transit from last subtoken to original destination state.
                    # # (notice that in end of iteration, subtok2 == tok_symbol[-1])
                    # add_token_transition(iter_state, tok_symbol[-1], tgt_state)

            
        # 4. For all accepting states, add transition to a "end-of-sequence" state which will allow for the eos_token;
        #    From there, have a self-referring "pad" transition (padding the rest of the output).
        if set_eos_loop:
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
    
    def as_word_level(self, inplace=False) -> 'DFA':
        """ Adjust the DFA alphabet to fit single words (space delimited). """
        # define a SpaceTokenizer mimic object for `adjust_for_tokenizer`
        class SpaceTokenizer():
            def __init__(self):
                self.unk_token = None
            def tokenize(self, symbol: str) -> List[str]:
                return symbol.split(" ")
        space_tokenizer = SpaceTokenizer()
        ret_dfa = self.adjust_for_tokenizer(space_tokenizer, inplace=inplace, convert_to_word_ids=False, set_eos_loop=False)
        # unset `dfa.tokenizer` to avoid logical errors (as it is not a "real" tokenizer)
        ret_dfa.tokenizer = None
        return ret_dfa
    
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
    def permissive() -> 'DFA':
        " Returns a very basic DFA with a single (accepting) state that allows all symbols. "
        state = "<permissive>"
        return DFA({state: {DFA.WILDCARD: state}}, state, accept_states={state})
    
    @staticmethod
    def from_slots(slots: List[List[Alphabet]] ) -> 'DFA':
        dfa_as_dict = {i: {possible_token : i+1
                        for possible_token in possible_tokens} 
                    for i, possible_tokens in enumerate(slots)}
        return DFA(dfa_as_dict, 0, accept_states={len(slots)})
    
    @staticmethod
    def get_partial_dfa(intermediate_dfa: 'DFA', 
                        activating_symbols: Iterable[Alphabet],  
                        deactivating_symbols: Iterable[Alphabet] = [],
                        cyclic: bool = False,
                        deactivate_after_accept: bool = False) -> 'DFA':
        """
        Define a DFA that is "idle" until reading an "activating" symbol (from `activating_symbols`); 
          during these "idle" states all alphabet is permitted (self-loop with wildcard). 
        After "activation", the DFA is constraining according to the given `intermediate_dfa` logic; 
          and reading any symbol of the `deactivating_symbols`, or entering one of the `intermediate_dfa.accept_states` 
          (if `deactivate_after_accept`), will transmit again to the idle status. 
        `cyclic` determines whether after "deactivation", the automaton returns to the initial idle state and thus 
          can be reactivated (True), or else it proceeds to be infinitely idle (False).
        All "idle" states are accepting.
        """
        initial_idle_state = "s0-idle"
        final_idle_state = initial_idle_state if cyclic else "end-state-idle" 
        dfa = intermediate_dfa.copy()
        orig_s0 = dfa.s0
        # pre-active
        dfa.s0 = initial_idle_state
        dfa.add_transition(initial_idle_state, DFA.WILDCARD, initial_idle_state)
        for activating_symbol in activating_symbols:
            dfa.add_transition(initial_idle_state, activating_symbol, orig_s0)
        dfa.s0 = initial_idle_state
        # post active
        dfa.accept_states = set([initial_idle_state, final_idle_state])
        for intermediate_dfa_state in intermediate_dfa.states:
            for deactivating_symbol in deactivating_symbols:
                dfa.add_transition(intermediate_dfa_state, deactivating_symbol, final_idle_state)
        if deactivate_after_accept:
            for orig_accept_state in intermediate_dfa.accept_states:
                dfa.add_transition(orig_accept_state, DFA.WILDCARD, final_idle_state)  
            # original accept_states are also accept states now because they represent an "idle" state
            dfa.accept_states.update(intermediate_dfa.accept_states)  
        dfa.add_transition(final_idle_state, DFA.WILDCARD, final_idle_state)
        return dfa
        
    
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


if __name__ == "__main__":
    test_DFA()
    test_concat_dfas()