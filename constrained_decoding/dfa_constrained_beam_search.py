from typing import Callable, Optional, Union, List

import torch
import torch.distributed as dist
from torch import nn

from transformers import (
    BeamScorer, 
    LogitsProcessorList, 
    StoppingCriteriaList,
    T5TokenizerFast
)
import transformers
from transformers.generation_utils import (
    BeamSearchOutput,
    BeamSearchDecoderOnlyOutput,
    BeamSearchEncoderDecoderOutput,
)
from transformers.generation_stopping_criteria import validate_stopping_criteria
import warnings
import numpy as np

import itertools       
from .dfa import DFA 
from .dfa_constrained_generate import dfa_constrained_generate 
    
def dfa_constrained_beam_search(
    self,
    input_ids: torch.LongTensor,
    beam_scorer: BeamScorer,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: Optional[bool] = None,
    encoder_input_ids: Optional[torch.LongTensor] = None, # @dfa
    **model_kwargs,
) -> Union[BeamSearchOutput, torch.LongTensor]:
    r"""
    This is an adapted beam_search method for contraining the search according to a custom Deterministic Finite Automaton (DFA).
    The DFA should be provided as a method attribute, by setting `dfa_constrained_beam_search.dfa = DFA(...)`.
    The original method is copied from transformers.generation_utils.GenerationMixins.beam_search(...).
        Github Ref: https://github.com/huggingface/transformers/blob/05fa1a7ac17bb7aa07b9e0c1e138ecb31a28bbfe/src/transformers/generation_utils.py#L1730
    Additions are denoted with a preceding "# @dfa:" comment.  
    ---------------
    Generates sequences for models with a language modeling head using beam search decoding.

    Parameters:

        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        beam_scorer (`BeamScorer`):
            An derived instance of [`BeamScorer`] that defines how beam hypotheses are
            constructed, stored and sorted during generation. For more information, the documentation of
            [`BeamScorer`] should be read.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from
            [`LogitsProcessor`] used to modify the prediction scores of the language modeling
            head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from
            [`StoppingCriteria`] used to tell if the generation loop should stop.
        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of
            generated tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`int`, *optional*):
            The id of the *end-of-sequence* token.
        output_attentions (`bool`, *optional*, defaults to *False*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to *False*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to *False*):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to *False*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If
            model is an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`generation_utilsBeamSearchDecoderOnlyOutput`],
        [`~generation_utils.BeamSearchEncoderDecoderOutput`] or obj:*torch.LongTensor*: A
        `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation_utils.BeamSearchDecoderOnlyOutput`] if
        `model.config.is_encoder_decoder=False` and `return_dict_in_generate=True` or a
        [`~generation_utils.BeamSearchEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.


    Examples:

    ```python
    >>> from transformers import (
    ...    AutoTokenizer,
    ...    AutoModelForSeq2SeqLM,
    ...    LogitsProcessorList,
    ...    MinLengthLogitsProcessor,
    ...    BeamSearchScorer,
    ... )
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

    >>> encoder_input_str = "translate English to German: How old are you?"
    >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


    >>> # lets run beam search using 3 beams
    >>> num_beams = 3
    >>> # define decoder start token ids
    >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
    >>> input_ids = input_ids * model.config.decoder_start_token_id

    >>> # add encoder_outputs to model keyword arguments
    >>> model_kwargs = {
    ...     "encoder_outputs": model.get_encoder()(encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True)
    ... }

    >>> # instantiate beam scorer
    >>> beam_scorer = BeamSearchScorer(
    ...     batch_size=1,
    ...     num_beams=num_beams,
    ...     device=model.device,
    ... )

    >>> # instantiate logits processors
    >>> logits_processor = LogitsProcessorList([
    ...     MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
    ... ])

    >>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)

    >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
    ```"""
    # @dfa: get DFA from function attribute 'dfa' or `dfa_factory` if exists
    dfa = vars(dfa_constrained_beam_search).get("dfa", None)
    dfa_factory = vars(dfa_constrained_beam_search).get("dfa_factory", None)
    # Applying multiple dfas concurrently means that generation is constrained to *conjuction*
    #  of allowed symbols of the states of all dfas.
    multiple_dfas = vars(dfa_constrained_beam_search).get("multiple_dfas", [])
    multiple_dfa_factories = vars(dfa_constrained_beam_search).get("multiple_dfa_factories", [])
    is_dfa = dfa or dfa_factory or multiple_dfas or multiple_dfa_factories
    
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    if len(stopping_criteria) == 0:
        warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
    pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
    output_scores = output_scores if output_scores is not None else self.config.output_scores
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams

    batch_beam_size, cur_len = input_ids.shape

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))
    
    # @dfa: collect all dfas that would be applied per beam into `multiple_dfa_factories`
    if dfa: multiple_dfas.append(dfa)
    if dfa_factory: multiple_dfa_factories.append(dfa_factory)
    for a_dfa in multiple_dfas:
        multiple_dfa_factories.append(lambda x: a_dfa)
    n_dfas_per_beam = len(multiple_dfa_factories)
    # @dfa: Init an array of iterators (holding current_states) for running the dfa efficiently per beam
    # To support applying multiple dfas concurrently on every beam, `current_state_iterators` would have 
    #  the shape (`n_dfas_per_beam` X `batch_beam_size`) 
    if multiple_dfa_factories: # === if is_dfa:    
        current_state_iterators = []
        for dfa_factory in multiple_dfa_factories:
            dfas = [[dfa_factory(input_seq)] * num_beams 
                    for input_seq in encoder_input_ids] 
            dfas = list(itertools.chain(*dfas)) # length batch_beam_size
            current_state_iterators.append([dfa.iterator() for dfa in dfas])
          
    this_peer_finished = False  # used by synced_gpus only
    while True:

        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]
        # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
        # cannot be generated both before and after the `nn.functional.log_softmax` operation.
        next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
        next_token_scores = nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)

        # @dfa: in every iteration we are directly constraining next tokens to dfa transitions
        if is_dfa:
            # to constrain to conjunction of all dfas (per beam), we will apply -inf masks iteratively
            for per_dfa_current_state_iterators in current_state_iterators:
                mask = torch.ones_like(next_token_scores) # (batch_beam_size, vocab_size); 1 will denote forbidden tokens
                for i, current_state_iterator in enumerate(per_dfa_current_state_iterators):
                    allowed_word_ids = list(current_state_iterator.get_allowed_transitions().keys())
                    if DFA.WILDCARD in allowed_word_ids: 
                    # allow all words
                        mask[i] = 0
                    else:
                    # allow only those words
                        mask[i, allowed_word_ids] = 0 # put zeros in allowed token ids       
                next_token_scores = next_token_scores.masked_fill(mask.bool(), -float("inf"))
        
        next_token_scores = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        next_token_scores, next_tokens = torch.topk(
            next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
        )

        next_indices = (next_tokens / vocab_size).long()
        next_tokens = next_tokens % vocab_size

        # stateless
        beam_outputs = beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        # @dfa: update automatons state
        if is_dfa:
            # copy object when indexed (to avoid references to same iter object) 
            current_state_iterators = [ [current_state_iterators[dfa_i][beam_i].copy() 
                                         for beam_i in beam_idx.cpu()]  # take the previous states of the selected beams 
                                       for dfa_i in range(n_dfas_per_beam)]
            for per_dfa_current_state_iterators in current_state_iterators:
                for i, dfa_iter in enumerate(per_dfa_current_state_iterators):
                    # step is applied internally in iterators
                    success, new_state = dfa_iter.step(beam_next_tokens[i].item())             
                
        input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        if model_kwargs["past"] is not None:
            model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

        # increase cur_len
        cur_len = cur_len + 1

        if beam_scorer.is_done or stopping_criteria(input_ids, scores):
            if not synced_gpus:
                break
            else:
                this_peer_finished = True

    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
    )

    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs["sequence_scores"] = None
        if self.config.is_encoder_decoder:
            return BeamSearchEncoderDecoderOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return BeamSearchDecoderOnlyOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return sequence_outputs["sequences"]     
    

def test_dfa_constrained_beam_search():
    tokenizer = T5TokenizerFast.from_pretrained("t5-small")
    dfa_d={ 0:{'my':1, 'the':1},
            1:{'name':2, 'dog':2},
            2:{'of':0, 'is': 3, 'was':1},
            3:{'John':4, 'complication':4}}
    dfa = DFA(dfa_d, 0, accept_states=[4]).adjust_for_tokenizer(tokenizer, convert_to_word_ids=True)
    dfa_d2={0:{'by':1, "king": 1},
            1:{'name':2, 'dog':2},
            2:{'of':0, 'is': 3, 'was':1},
            3:{'John':4, "mile": 4}}
    dfa2 = DFA(dfa_d2, 0, accept_states=[4]).adjust_for_tokenizer(tokenizer, convert_to_word_ids=True)
    text = "the name is my dog of good complication John " 

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

    # running our custom beam search 
    model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')
    
    # replace beam search method with our custom function as a bound method
    from .dfa_decoding import set_decoding_to_dfa_constrained
    set_decoding_to_dfa_constrained(model, dfas=[dfa])
    
    # play with additional Losits Processors
    logits_processors = LogitsProcessorList([
        # transformers.NoRepeatNGramLogitsProcessor(3),
        # transformers.TemperatureLogitsWarper(temperature=0.2),
        transformers.RepetitionPenaltyLogitsProcessor(penalty=2.),  # make the beams different from one another
        transformers.ForcedEOSTokenLogitsProcessor(max_length=30, eos_token_id=tokenizer.vocab[tokenizer.eos_token]),
        # transformers.MinLengthLogitsProcessor(min_length=15, eos_token_id=tokenizer.vocab[tokenizer.eos_token]), # collision with defaults
    ])
    
    generated = model.generate(input_ids, 
                            max_length=20,
                            num_beams=num_beams,
                            num_return_sequences=num_return_beams,
                            # logits_processor=logits_processors,
                            return_dict_in_generate=True,
                            output_scores=True,
                            ) 
    
    for index, output_tokenized in enumerate(generated["sequences"]):
        output = tokenizer.decode(output_tokenized)
        print(f'beam {index}: {output}')
 
 
if __name__ == "__main__":
    test_dfa_constrained_beam_search() 