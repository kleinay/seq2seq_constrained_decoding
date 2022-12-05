import sys
sys.path.append("/home/nlp/kleinay/Parsing/Seq2Seq_QASRL_Parsing/qasrl_bart/seq2seq_constrained_decoding")
from constrained_decoding.sequences_constraints import AllowedSequencesLogitsProcessor, ForbiddenSequencesLogitsProcessor

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch

gpt2 = AutoModelForCausalLM.from_pretrained("distilgpt2", return_dict_in_generate=True)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

def test_allowed(**further_kwargs):
    seq = "I am going home"
    tok_seq_ids = tokenizer(seq, return_tensors="pt").input_ids
    lgp = AllowedSequencesLogitsProcessor(tokenizer([
        "I would have fun today", 
        "I am leaving", 
        "I am going home for a nap on the soffa, otherwise"
        ]).input_ids, 
                                          eos_token_id=tokenizer.eos_token_id)
    gen_kwargs = {
        "pad_token_id": 1,
        "num_return_sequences": 1,
    }
    output = gpt2.generate(tok_seq_ids[:, :1], logits_processor=[lgp], 
                           **dict(gen_kwargs, **further_kwargs)) 
    output2 = gpt2.generate(tok_seq_ids[:, :1], 
                           **dict(gen_kwargs, **further_kwargs)) 
    # output = gpt2.generate(tok_seq_ids, **gen_kwargs)  # eos==50256
    output_str = tokenizer.batch_decode(output.sequences)
    output2_str = tokenizer.batch_decode(output2.sequences)
    print(output_str)
    assert output_str != output2_str


def test_forbiden(**further_kwargs):
    seq = "I am"
    forb_seq = ["I am not a ", "I remember", "I The first time","I.", "I.\n\n\n"]
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tok_seq_ids = tokenizer(forb_seq).input_ids
    lgp = ForbiddenSequencesLogitsProcessor(tok_seq_ids)
    #     "I would have fun today", 
    #     "I am leaving", 
    #     "I am going home for a nap on the soffa, otherwise"
    #     ]).input_ids, 
    #                                       eos_token_id=tokenizer.eos_token_id)
    gen_kwargs = {
        "pad_token_id": tokenizer.pad_token_id,
        "num_return_sequences": 1,
    }
    output = gpt2.generate(torch.tensor([tok_seq_ids[0][:1]]), 
                           logits_processor=[lgp], 
                           **dict(gen_kwargs, **further_kwargs)) 
    output2 = gpt2.generate(torch.tensor([tok_seq_ids[0][:1]]), 
                        **dict(gen_kwargs, **further_kwargs)) 
    # output = gpt2.generate(tok_seq_ids, **gen_kwargs)  # eos==50256
    output_str = tokenizer.batch_decode(output.sequences)
    output2_str = tokenizer.batch_decode(output2.sequences)
    print(output_str)
    assert output_str != output2_str


test_allowed(do_sample=False)
test_allowed(do_sample=True)
test_allowed(do_sample=False, num_beams=3)
test_allowed(do_sample=True, num_beams=3)
test_forbiden(do_sample=False)
test_forbiden(do_sample=True)
test_forbiden(do_sample=False, num_beams=4, max_length=10, num_return_sequences=4)
test_forbiden(do_sample=True, num_beams=4, max_length=10)
