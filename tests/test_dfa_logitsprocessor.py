
from transformers import (
    T5TokenizerFast,
    LogitsProcessorList,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM, 
    LogitsProcessor, 
)
import torch

from constrained_decoding.dfa import DFA
from constrained_decoding.logits_processors.dfa_decoding import DfaDecodingLogitsProcessor
 

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
    input_ids = torch.tensor([text_ids[:3]])

    # inputs = tokenizer(text)
    scores = torch.tensor([[1/len(tokenizer.get_vocab())] * len(tokenizer.get_vocab())])
    
    processor = DfaDecodingLogitsProcessor(tokenizer, dfa)
    print(f"\nScores before: { {tokenizer.convert_ids_to_tokens(i): scores[0][i] for i in set(text_ids)} }")
    new_scores = processor(input_ids, scores)
    
    print(f"\nScores After : { {tokenizer.convert_ids_to_tokens(i): new_scores[0][i] for i in set(text_ids)} }")

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
    test_DFA_Decoding_LogitsProcessor()