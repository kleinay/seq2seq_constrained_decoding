from typing import Optional, Callable, List

from constrained_decoding.dfa import DFA
from constrained_decoding.autoregressive_dfa_constraining.dfa_constrained_beam_search import dfa_constrained_beam_search
from constrained_decoding.autoregressive_dfa_constraining.dfa_constrained_generate import dfa_constrained_generate

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
 
 