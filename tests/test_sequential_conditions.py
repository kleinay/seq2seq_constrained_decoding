import sys
from constrained_decoding.logits_processors.sequential_conditions import XMustFollowYLogitsProcessor, XMustNotFollowYLogitsProcessor

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch

gpt2 = AutoModelForCausalLM.from_pretrained("distilgpt2", return_dict_in_generate=True)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

#TODO