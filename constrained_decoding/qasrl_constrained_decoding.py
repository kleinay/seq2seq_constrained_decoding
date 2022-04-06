from typing import List, Iterable, Tuple, Any, Union, Dict, Optional, Callable, overload
import itertools
from collections import defaultdict
import json
from pathlib import Path

import torch
import numpy as np
import transformers
from transformers import AutoTokenizer, LogitsProcessorList

# from .set_decoding import SetDecodingLogitsProcessor
from .dfa import DFA, DFAIterator, State, Alphabet
from .dfa_decoding import set_decoding_to_dfa_constrained


# Define a LogitsProcessorList that enforces valid qasrl-seq2seq output

separator_qa_pairs = "<QA_QA>"
separator_q_a = "<Q_A>"
separator_answers = "<A_A>"

new_word_ch = '▁'   # for T5 tokenizer

repo_root_path = Path(__file__).absolute().parent.parent

STOP_WORDS = """we you he she they is are was were will it the a an and or , in on at of ' " no my has have"""


def is_prefixed_by(list, sublist) -> bool:
    return list[:len(sublist)] == sublist

def get_copy_dfa(tok_source_seq: List[str], terminate_token: Optional[str] = None) -> DFA:
    """ 
    Get a DFA which simulates a possible 'copying' process of the decoder from the `source_sequence`.
    :param tok_source_seq: the tokenized source sequence to copy from.
    :param terminate_token (optional): if provided, dfa is defined to have a transition from every state to 
      a single accepting "<TERMINATE>" state via `terminate_token` (e.g. EOS token). 
      If not provided, all states are accept states. 
    """
    # start with a naive "forward" DFA which correspond to the full source sequence
    # states correspond to token indices
    dfa_dict = {idx: {token: idx+1}
                for idx, token in enumerate(tok_source_seq)} 
    # Ideally, we should now add a transition from s0 to all states, with the corresponding tok[i-1] transition symbol, 
    #  since the copying is not necessarily from the beginning of the sequence.
    # This would work for a sequence with no token repitition.
    # But now consider the case of common prefixes of sub-sequences; if a sub-sequence (e.g. ['a','b']) is occuring in sequence more than once,
    #  we need to construct an 'artificial' state that represents reading the sub-sequence 
    #  (without yet assigning it a specific index in the sequence).   
    subseqs = [(i,tuple(tok_source_seq[i:j])) for i in range(len(tok_source_seq)-1) for j in range(i+1, len(tok_source_seq))]
  
    prefix2next_index = defaultdict(list)
    for start_idx, subseq in subseqs:
        prefix2next_index[subseq].append(start_idx + len(subseq))
    # save only "common prefixes" (prefixes that occur more than once) - for building designated "states"
    common_prefix2next_index = {prefix: indices for prefix, indices in prefix2next_index.items() if len(indices)>1}
    sorted_common_prefixes = sorted(common_prefix2next_index.items(), key=lambda pair: len(pair[0]), reverse=True)
    additional_transitions = dict()
    for subseq, next_idxs in sorted_common_prefixes:
        seen_subseq_state = subseq
        additional_transitions[seen_subseq_state] = dict()
        for next_idx in next_idxs:
            symbol, tgt_state = list(dfa_dict[next_idx].items())[0]    # suppesed to be only one item in this transition dict
            if subseq + (symbol,) in additional_transitions:
                tgt_state = subseq + (symbol,)
            additional_transitions[seen_subseq_state][symbol] = tgt_state
    dfa_dict.update(additional_transitions)
    # update init state transitions to include every one-token transition
    first_transition_after_init = {tok: idx+1 for idx, tok in enumerate(tok_source_seq)}
    # ovveride with "common prefix" of length 1 - so that init state would transit into artificial "indetermined" state 
    first_transition_after_init.update({artificial_state[0]: artificial_state 
                                        for artificial_state in additional_transitions.keys() 
                                        if len(artificial_state) == 1 })
    dfa_dict[0] = first_transition_after_init
    dfa = DFA(dfa_dict, 0)
    # set transitions to "terminate" state
    if terminate_token is not None:
        terminate_state = '<TERMINATE>'
        dfa.states.add(terminate_state)
        for state in set(dfa.states):
            dfa.add_transition(state, terminate_token, terminate_state)
        dfa.accept_states = {terminate_state}
    return dfa
    
def test_get_copy_dfa():
    sentence = "abcabdssabcr"
    tok_source_seq: List[str] = list(sentence) # ['a', 'b', 'c', 'a', 'b', 'd', 's', 's', 'a', 'b', 'c', 'r'] 
    # test with terminate symbol
    dfa = get_copy_dfa(tok_source_seq, '<s>')
    subseqs = [(i,tuple(tok_source_seq[i:j])) for i in range(len(tok_source_seq)-1) for j in range(i+1, len(tok_source_seq))]
    for i,subseq in subseqs:
        assert dfa(subseq + ('<s>',)) == (True, '<TERMINATE>', True), f"substring {subseq} failed"
        assert not dfa(subseq)[2] 
    assert not dfa("aba")[0]
    # test without terminate symbol
    dfa = get_copy_dfa(tok_source_seq)
    for i,subseq in subseqs:
        assert dfa(subseq)[2], f"substring {subseq} failed"
        assert not dfa(subseq + ('c', 'b'))[2] 
    print("test_get_copy_dfa() passed successfully")
    
def get_qasrl_question_dfa(constrain_verb: bool):
    qasrl_slots = json.load(open(repo_root_path / "qasrl_slots.json")) 
    # if constraining also verb to be taken from the slot vocabulary
    slots_order = ['wh', 'aux', 'subj', 'verb', 'obj', 'prep', 'obj2', '?']
    qasrl_slots_as_list = [qasrl_slots[name] for name in slots_order]
    if constrain_verb:
        dfa = DFA.from_slots(qasrl_slots_as_list)     
    
    # if not constraining verb - need to concat two DFAs and put WILDCARD in the bridge 
    else: 
        dfa1 = DFA.from_slots(qasrl_slots_as_list[:3])    
        dfa2 = DFA.from_slots(qasrl_slots_as_list[4:])    
        # add transitions from end state of dfa1 (state 3) to a new state (4) for all vocabulary tokens
        dfa1.states.add(4)
        dfa1.add_transition(3, DFA.WILDCARD, 4)
        dfa1.accept_states = {4}
        # now concat them and merge end-state of dfa1 with dfa2.s0 
        dfa = DFA.concat_two(dfa1, dfa2, tie_by_merge=True)
    
    return dfa

def get_qasrl_answer_dfa(tokenized_sentence: List[str]) -> DFA:
    " Create a cyclic qasrl_answers_dfa out of a copy_dfa. "
    answer_dfa = get_copy_dfa(tokenized_sentence)
    # remove initial state (empty answer) from accept_states 
    answer_dfa.accept_states.remove(answer_dfa.s0)
    # start only with tokens that start new word
    non_start_tokens = [tok for tok in tokenized_sentence if not tok.startswith(new_word_ch) or tok==new_word_ch]
    for tok in set(non_start_tokens):
        answer_dfa[answer_dfa.s0].pop(tok)
    # block "half-words" - remove from dfa.accept_states all states corresponding to tokens not starting with the new-word prefix "_"  
    half_word_states = set() 
    non_end_tokens = []
    # last_state = [s for s in answer_dfa.states if answer_dfa[s]=={}][0]
    it = answer_dfa.iterator()
    # iterate dfa linearly through the sentence
    prev_state = it.current_state
    for tok in tokenized_sentence:
        success, state = it.step(tok)
        # assert len(next_transions)==1, f"{tok} hasn't a single transion"
        if not tok.startswith(new_word_ch):
            half_word_states.add(prev_state)
            non_end_tokens.append(tok)
        prev_state = state
        
    answer_dfa.accept_states.difference_update(half_word_states)
               
    return answer_dfa     
    

def get_qasrl_full_sequence_dfa(input_sentence: str, tokenizer, special_tokens_constants, convert_to_word_ids=True) -> DFA:
    tokenized_sentence = tokenizer.tokenize(input_sentence)
    # create a qasrl_answer_dfa based on a copy_dfa
    answer_dfa = get_qasrl_answer_dfa(tokenized_sentence)
    # create a cyclic qasrl_answers_dfa for multiple answers
    answers_dfa = answer_dfa.as_cyclic(bridge=special_tokens_constants.separator_output_answers)

    # combine qasrl_question_dfa and qasrl_answers_dfa to form a qa_dfa       
    qasrl_question_dfa = get_qasrl_question_dfa(constrain_verb=False)
    qa_dfa = DFA.concat_two(qasrl_question_dfa, answers_dfa, 
                            bridge_symbol=special_tokens_constants.separator_output_question_answer)
    # create a cyclic qasrl_qas_sequence from qa_dfa 
    qasrl_qas_sequence_dfa = qa_dfa.as_cyclic(bridge=special_tokens_constants.separator_output_pairs)  
    
    # adjust to tokenizer
    qasrl_qas_sequence_dfa = qasrl_qas_sequence_dfa.adjust_for_tokenizer(tokenizer,
                                                                         convert_to_word_ids=convert_to_word_ids)
    return qasrl_qas_sequence_dfa

def removeKeys(orig_dict: Dict[Any, Any], keys: List[Any], inplace=True) -> Dict[Any, Any]:
    """ remove these keys from the orig_dict. """
    if not inplace:
        orig_dict = orig_dict.copy()
    for key in keys:
        orig_dict.pop(key, None)
    return orig_dict 

class NoRepetitionDFAIterator(DFAIterator):
    def __init__(self, dfa: 'DFA', 
                 activate_symbols: Optional[Iterable[Alphabet]] = None,
                 deactivate_symbols: Optional[Iterable[Alphabet]] = None,
                 reinit_symbols: Iterable[Alphabet] = [],
                 allow_symbols: Iterable[Alphabet] = [], 
                 reinit_on_deactivation: bool = True,
                 convert_to_word_ids: bool = False,
                 tokenizer = None,
                 ) -> None:
        """ 
        `activate_symbols` - if provided, the "no-repeatition" strategy is enabled only after reading one the symbols in `activate`
        `deactivate_symbols` are symbols that when reading them, the iterator disables the "no-repeatition" strategy,
          and also (depending on `reinit_on_deactivation`) "re-initializes `self.seen_symbols` to be empty. 
        `allow_symbols` are symbols that are allowed to be repeated.
        `reinit_symbols` are symbols that when reading them, the iterator re-initializes `self.seen_symbols` to be empty. 
        """
        super().__init__(dfa)
        assert not convert_to_word_ids or tokenizer, "Must provide a tokenizer when `convert_to_token_ids` is True."
        self.seen_symbols = set()
        self.allowed_symbols = set(allow_symbols)
        self.reinit_symbols = set(reinit_symbols)
        self.activate_symbols = set(activate_symbols)
        self.deactivate_symbols = set(deactivate_symbols)
        self.reinit_on_deactivation = reinit_on_deactivation
        self.tokenizer = tokenizer
        # encode symbols by tokenizer to token_ids
        if convert_to_word_ids:
            self.allowed_symbols = set(tokenizer.convert_tokens_to_ids(self.allowed_symbols))
            self.reinit_symbols = set(tokenizer.convert_tokens_to_ids(self.reinit_symbols))
            self.activate_symbols = set(tokenizer.convert_tokens_to_ids(self.activate_symbols))
            self.deactivate_symbols = set(tokenizer.convert_tokens_to_ids(self.deactivate_symbols))
        # state of activation
        self.active = activate_symbols is None 
    
    def step(self, input_symbol: Alphabet) -> Tuple[bool, State]:
        success, end_state = super().step(input_symbol)
        # register input symbol
        if success and self.active:
            self.seen_symbols.add(input_symbol)
        # re-initialize memory
        if input_symbol in self.reinit_symbols:
            self.seen_symbols = set()
        # activate
        if input_symbol in self.activate_symbols:
            self.active = True
        # deactivate
        if input_symbol in self.deactivate_symbols:
            self.active = False
            if self.reinit_on_deactivation:
                self.seen_symbols = set()
            
        return success, end_state
    
    def get_allowed_transitions(self) -> Dict[Alphabet, State]:
        # Notice that the iterator cannot block the wildcard transition! it can only remove explicit transition symbols
        allowed_transitions = super().get_allowed_transitions()
        if self.active:
            forbidden_repeated_symbols = self.seen_symbols - self.allowed_symbols  
            allowed_transitions = removeKeys(allowed_transitions, forbidden_repeated_symbols, inplace=False)
        return allowed_transitions
        

def set_as_redundance_answer_disposer_dfa(base_dfa, special_tokens_constants, 
                                          tokenizer = None,
                                          convert_to_word_ids = False) -> DFA:
    """
    A partial DFA that is activated given Q_A_SEP and deactivated given QA_PAIR_SEP.
    During its active phase, while it iterate over the subsequence corresponding to the answers,
    it can remember generated ids (except from ANS_SEP - extra_id_3) and forbid them.
    The Goal is to prevent from answering the same question with multiple overlapping answers.
    """
    repetitions_allowed = [special_tokens_constants.separator_output_answers] 
    if tokenizer:
        repetitions_allowed += tokenizer.tokenize(STOP_WORDS)
    else:
        repetitions_allowed += STOP_WORDS.split()
        
    reinit_symbols = [special_tokens_constants.separator_output_pairs]
    
    no_repeat_iter_factory = lambda d: NoRepetitionDFAIterator(d,
                                                               activate_symbols=[special_tokens_constants.separator_output_question_answer],
                                                               deactivate_symbols=[special_tokens_constants.separator_output_pairs],
                                                               reinit_symbols=reinit_symbols,
                                                               allow_symbols=repetitions_allowed, 
                                                               reinit_on_deactivation=True,
                                                               tokenizer=tokenizer,
                                                               convert_to_word_ids=convert_to_word_ids
                                                               )
    base_dfa.set_iterator_factory(no_repeat_iter_factory)
    # construct a partial DFA that activates `no_repeat_dfa` when generating QASRL "answers"
    # no_repeat_dfa = DFA.get_partial_dfa(no_repeat_dfa, 
    #                                   activating_symbols=[special_tokens_constants.separator_output_question_answer],
    #                                   deactivating_symbols=[special_tokens_constants.separator_output_pairs],
    #                                   cyclic=True)
    return base_dfa


def test_qasrl_question_dfa():
    from .pipeline import QASRL_Pipeline
    pipe = QASRL_Pipeline("kleinay/qanom-seq2seq-model-joint")
    tokenizer = pipe.tokenizer
    qdfa = get_qasrl_question_dfa(constrain_verb=False)
    tqdfa = qdfa.adjust_for_tokenizer(tokenizer)
    def apply(question):
        return tqdfa(tokenizer.tokenize(question))
    # debugging with prints
    print(apply("what did someone use _"))
    print(apply("what did someone use _ on"))
    print(apply("how long did "))
    print(apply("how"))
    print(apply("how _ _ try _ _ _ ?"))
    # tests
    assert apply("how much _ _ try _ _ _ ?")[2]
    assert apply("how _ _ umbrella _ _ do something ?")[2]
    assert apply("how _ _ verb someone _ something ?")[2]
    print("test_qasrl_question_dfa() passed successfully")


def test_qasrl_answer_dfa():
    from .pipeline import QASRL_Pipeline
    pipe = QASRL_Pipeline("kleinay/qanom-seq2seq-model-joint")
    tokenizer = pipe.tokenizer
    # taking a sentence with some multi-token words
    sentence = "The two tails are suspects of murdering many ozone 2050 customers ceasefire . "
    tokenized_sentence = tokenizer.tokenize(sentence)
    # answer dfa is a copy dfa 
    sep = "<extra_id_9>"
    dfa = get_qasrl_answer_dfa(tokenized_sentence)
    def apply(seq):
        return dfa(tokenizer.tokenize(seq)) 
    assert apply("two tails")[2] 
    assert apply("two tail")[2] == False 
    assert apply("s of murdering")[2] == False 
    assert apply("many ozone")[2]  
    assert apply("many o")[2] == False  
    assert apply("The two tails")[2]
    assert apply("ceasefire")[2]
    
    
def test_qasrl_answers_dfa():
    from .pipeline import QASRL_Pipeline
    pipe = QASRL_Pipeline("kleinay/qanom-seq2seq-model-joint")
    tokenizer = pipe.tokenizer
    # taking a sentence with some multi-token words
    sentence = "The two tails are suspects of murdering many ozone 2050 customers ceasefire . "
    tokenized_sentence = tokenizer.tokenize(sentence)
    # answer dfa is a copy dfa 
    sep = "<extra_id_9>"
    dfa = get_qasrl_answer_dfa(tokenized_sentence)
    dfa = dfa.as_cyclic(bridge=sep)
    def apply(seq):
        return dfa(tokenizer.tokenize(seq)) 
    assert apply("two tails")[2] 
    assert apply(f"{sep} two tails")[2] == False
    assert apply(f"two tails {sep} suspects of murdering")[2]
    assert apply(f"ceasefire {sep} 2050 customers")[2]
    assert apply(f"ceasefire {sep} 2050 customers {sep}")[2] == False


def test_full_qasrl_dfa():
    from .pipeline import QASRL_Pipeline
    pipe = QASRL_Pipeline("kleinay/qanom-seq2seq-model-joint")
    tokenizer = pipe.tokenizer
    sentence = "The doctor was interested in Luke's treatment yesterday given by the other doctor ."
    dfa = get_qasrl_full_sequence_dfa(sentence, tokenizer, pipe.special_tokens, convert_to_word_ids=False)
    def apply(seq):
        return dfa(tokenizer.tokenize(seq))  
    q = "how _ _ verb someone _ something ?" 
    assert apply("how much _ _ try _ _ _ ?<extra_id_7> the other doctor </s>")[2]
    assert apply("how much _ _ try _ _ _ ?<extra_id_7> the other doctor<extra_id_3> Luke")[2]
    assert apply("how _ _ umbrella _ _ do something ?<extra_id_7> the other<extra_id_9>") == (True, 0, False)
    assert apply(f"{q}<extra_id_7> the other<extra_id_9>{q}<extra_id_7> the other")[2]
    print("test_full_qasrl_dfa() passed successfully")


def test_redundance_answer_disposer():
    convert_to_word_ids = True
    from .pipeline import QASRL_Pipeline
    pipe = QASRL_Pipeline("kleinay/qanom-seq2seq-model-joint")
    tokenizer = pipe.tokenizer
    sentence = "The doctor was interested in Luke 's treatment yesterday given by the other doctor ."
    dfa = get_qasrl_full_sequence_dfa(sentence, tokenizer, pipe.special_tokens, convert_to_word_ids=convert_to_word_ids)
    dfa_orig = dfa.copy()
    set_as_redundance_answer_disposer_dfa(dfa, pipe.special_tokens, tokenizer=tokenizer, convert_to_word_ids=convert_to_word_ids)
    def apply(seq):
        if convert_to_word_ids:
            return dfa(tokenizer.encode(seq)[:-1]) 
        else:
            return dfa(tokenizer.tokenize(seq)) 
    question = "how _ _ verb someone _ something ?"   
    assert apply(f"{question}<extra_id_7> by the other doctor<extra_id_3> was interested")[2]
    assert apply(f"{question}<extra_id_7> by the other doctor<extra_id_3> other") == (False, '<2>_0', False)
    assert apply(f"{question}<extra_id_7> by the other doctor<extra_id_3> the")[2]  # repeating a stop-word is OK
    assert apply(f"{question}<extra_id_7> by the other doctor<extra_id_9> {question}")[0] 
    assert apply(f"{question}<extra_id_7> by the other doctor<extra_id_9> {question}<extra_id_7> by the")[2] 
    assert apply(f"{question}<extra_id_7> by the other doctor<extra_id_3> was<extra_id_3> interested")[2]
    assert apply(f"{question}<extra_id_7> by the other<extra_id_9> {question}<extra_id_7> doctor<extra_id_3> doctor")[0] == False
    
    print("test_redundance_answer_disposer() passed successfully")



if __name__ == "__main__":
    # Unit Testing
    
    # test_get_copy_dfa()
    # test_redundance_answer_disposer()
    # test_get_qasrl_question_dfa()
    # test_full_qasrl_dfa()
    # test_qasrl_answer_dfa() # just the copying mechanism
    test_qasrl_answer_dfa() # 
    
    # Full test on real model
    from .pipeline import QASRL_Pipeline
    pipe = QASRL_Pipeline("kleinay/qanom-seq2seq-model-joint")
    sentence = "The doctor was interested to know about Luke 's bio-feedback treatment given by the nurse yesterday."
    # sentence = "The student was interested in Luke 's research about sea animals ."
    pipe_kwargs = dict(
        inputs="The doctor was interested to know about Luke 's bio-feedback <predicate> treatment given by the nurse yesterday.",
        verb_form="treat", 
        # inputs="The student was interested in Luke 's <predicate> research about sea animals .",
        # verb_form="research", 
        predicate_type="nominal")
    print("Baseline:")
    print(pipe(**pipe_kwargs))
    
    # Prior method - using a single `dfa` object
    # dfa = get_qasrl_full_sequence_dfa(sentence, pipe.tokenizer, pipe.special_tokens)
    # # enable special dfa-constrained beam search
    # set_decoding_to_dfa_constrained(pipe.model, dfa, pipe.tokenizer)
    
    # More generic - using a dfa_factory
    def dfa_factory(token_ids): 
        sentence = pipe.tokenizer.decode(token_ids, skip_special_tokens=True)
        print(sentence)
        return get_qasrl_full_sequence_dfa(sentence, pipe.tokenizer, pipe.special_tokens)
    # enable special dfa-constrained beam search
    set_decoding_to_dfa_constrained(pipe.model, dfa_factory=dfa_factory, tokenizer=pipe.tokenizer)
    
    # play with additional `generate` kwargs, including LogitsProcessors
    logits_processors = LogitsProcessorList([
        transformers.NoRepeatNGramLogitsProcessor(10),
        # transformers.TemperatureLogitsWarper(temperature=5),
        # transformers.RepetitionPenaltyLogitsProcessor(penalty=5.),  # make the beams different from one another
        transformers.ForcedEOSTokenLogitsProcessor(max_length=100, eos_token_id=pipe.tokenizer.vocab[pipe.tokenizer.eos_token]),
        # transformers.MinLengthLogitsProcessor(min_length=15, eos_token_id=tokenizer.vocab[tokenizer.eos_token]), # collision with defaults
    ])  
    generate_kwargs = dict(
                        #    max_length=80,
                        #    min_length=80,
                           num_beams=5,
                           num_return_sequences=5,
                        #    logits_processor=logits_processors,
                        #    early_stoping=True,
                            ) 
    
    print("Constrained:")
    print(pipe(**pipe_kwargs, generate_kwargs=generate_kwargs))
    
    # using model.generate directly
    pipe_input = "The doctor was interested to know about Luke 's bio-feedback <predicate> treatment given by the nurse yesterday."
    input_seq = pipe.preprocess(pipe_input, predicate_type="nominal", verb_form="treat")['input_ids']
    
