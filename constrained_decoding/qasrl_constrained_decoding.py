from typing import List, Iterable, Tuple, Any, Union, Dict, Optional, Callable
import itertools
from collections import defaultdict
import json
from pathlib import Path

import torch
import numpy as np
import transformers
from transformers import AutoTokenizer, LogitsProcessorList

# from .set_decoding import SetDecodingLogitsProcessor
from .dfa import DFA
from .dfa_decoding import set_decoding_to_dfa_constrained


# Define a LogitsProcessorList that enforces valid qasrl-seq2seq output

separator_qa_pairs = "<QA_QA>"
separator_q_a = "<Q_A>"
separator_answers = "<A_A>"

repo_root_path = Path(__file__).absolute().parent.parent

def is_prefixed_by(list, sublist) -> bool:
    return list[:len(sublist)] == sublist

def get_copy_dfa(tok_source_seq: List[str], terminate_token: Optional[str] = None) -> DFA:
    """ 
    Get a DFA which a possible 'copying' process of the decoder from the `source_sequence`.
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
    dfa = get_copy_dfa(tok_source_seq, '<s>')
    subseqs = [(i,tuple(tok_source_seq[i:j])) for i in range(len(tok_source_seq)-1) for j in range(i+1, len(tok_source_seq))]
    for i,subseq in subseqs:
        assert dfa(subseq + ('<s>',)) == (True, '<TERMINATE>', True), f"substring {subseq} failed"
        assert not dfa(subseq)[2] 
    assert not dfa("aba")[0]
    
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

def get_qasrl_full_sequence_dfa(input_sentence: str, tokenizer, special_tokens_constants, convert_to_word_ids=True) -> DFA:
    tokenized_sentence = tokenizer.tokenize(input_sentence)
    # create a cyclic qasrl_answers_dfa out of a copy_dfa
    answer_dfa = get_copy_dfa(tokenized_sentence)
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


def test_full_qasrl_dfa():
    from .pipeline import QASRL_Pipeline
    pipe = QASRL_Pipeline("kleinay/qanom-seq2seq-model-joint")
    tokenizer = pipe.tokenizer
    sentence = "The doctor was interested in Luke 's treatment yesterday given by the other doctor ."
    dfa = get_qasrl_full_sequence_dfa(sentence, tokenizer, pipe.special_tokens, convert_to_word_ids=False)
    def apply(seq):
        return dfa(tokenizer.tokenize(seq))    
    assert apply("how much _ _ try _ _ _ ?")[2]
    assert apply("how _ _ umbrella _ _ do something ?")[2]
    assert apply("how _ _ verb someone _ something ?")[2]


if __name__ == "__main__":
    # test_get_qasrl_question_dfa()
    # test_full_qasrl_dfa()
    # test on real model
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
                           min_length=80,
                           num_beams=10,
                           num_return_sequences=10,
                           logits_processor=logits_processors,
                        #    early_stoping=True,
                            ) 
    
    print("Constrained:")
    print(pipe(**pipe_kwargs, generate_kwargs=generate_kwargs))
    
    
