import sys
sys.path.append("/home/nlp/kleinay/Parsing/Seq2Seq_QASRL_Parsing/qasrl_bart/seq2seq_constrained_decoding")
from constrained_decoding.sequential_conditions import XMustFollowYLogitsProcessor, XMustNotFollowYLogitsProcessor

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch

gpt2 = AutoModelForCausalLM.from_pretrained("distilgpt2", return_dict_in_generate=True)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

#TODO