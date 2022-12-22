import transformers

from constrained_decoding.dfa import DFA


def test_dfa_constrained_beam_search():
    tokenizer = transformers.T5TokenizerFast.from_pretrained("t5-small")
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
    from constrained_decoding.autoregressive_dfa_constraining import set_decoding_to_dfa_constrained
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