# seq2seq_constrained_decoding

This project includes constrained-decoding utilities for *structured text generation* using Huggingface seq2seq models.

## Requirements
`pip install torch transformers`

The package is tested with `transformers==4.15.0`, might break for past or future versions.

# Use cases

Several use-cases leverage pretrained sequence-to-sequence models, such as BART or T5, 
for generating a (maybe partially) structured text sequence. 
For example, you may want to finetine the model to generate a set of key-points summarizing an input paragraph, or to produce a text sequence that follows some strict regularities. 

Below we detail about the use-cases supported by this project. 

## Word lists

The `word_list_decoding.py` module suggests simple `LogitsProcessor` classes which enforce allowed / forbidden words during text generation (see the `WhiteListLogitsProcessor` and `BlackListLogitsProcessor` classes respectively).

Example usage:

```python
from transformers import LogitsProcessorList, AutoTokenizer, AutoModel
from constrained_decoding.word_list_decoding import BlackListLogitsProcessor
tokenizer = AutoTokenizer.from_pretrained('t5-small')
model = AutoModel.from_pretrained('t5-small')

bad_words = "here are words that should not occur in the generated text"
bad_word_ids = tokenizer.encode(bad_words)
black_list_processor = BlackListLogitsProcessor(bad_word_ids)
good_words = "only these words can occur in the generated text"
good_word_ids = tokenizer.encode(good_words)
white_list_processor = WhiteListLogitsProcessor(good_word_ids)

input_seq = "here are the input words to condition generated text upon"
input_ids = tokenizer.encode(input_seq, return_tensors='pt')
out = model.generate(input_ids, num_beams=10)
print(tokenizer.batch_decode(out))
# ['<pad> Hier are the input words to condition generated text upon</s>']
out = model.generate(input_ids, num_beams=10, logits_processor=[black_list_processor])
print(tokenizer.batch_decode(out))
# ['<pad> Voici voici les input mots to condition a condition a condition a condition a']
out = model.generate(input_ids, num_beams=10, logits_processor=[white_list_processor])
print(tokenizer.batch_decode(out))
# ['<pad> in the words in the words in the words in the words in the words in the words in']
```

## Set decoding 

In some scenarios, you would like to regard the output sequence as expressing a set of elements comprised of sub-sequence. For example, you might finetune your Seq2Seq model on a multi-label document classification task (e.g. generating the set of relation types occuring in the input document). 

The `set_decoding.SetDecodingLogitProcessor` class can gurantee that no subsequence (e.g. a relation type) would occur more than once. Output subsequences are assumed to be defined using a special separator token.



## DFA-based constrained decoding

The most powerful and generic constrained decoding algorithm we propose is using a *Deterministic Finite Automata* (DFA).
You can instanciate a DFA with the `dfa.DFA` class, defined over a dictionary of dictionaries. 
For example, the following represents an automaton that accepts only binary numbers that are multiples of 3 (see [illustration](https://en.wikipedia.org/wiki/Deterministic_finite_automaton#/media/File:DFA_example_multiplies_of_3.svg) in the [Wikipedia article on DFA](https://en.wikipedia.org/wiki/Deterministic_finite_automaton)): 
```python
from dfa import DFA
transitions = {0:{'0':0, '1':1},
               1:{'0':2, '1':0},
               2:{'0':1, '1':2}} 
dfa = DFA(transitions, s0=0, accept_states=[0])
```

For defining constrained decoding using a DFA, the automaton's alphabet should correspond to tokens in the model's vocabulry.
The `DFA` class supports translating a `dfa` that uses regular words or phrases as alphabet into a tokenizer-adjusted dfa - 

```python
transitions = {0:{'John':1, 'Mike':1, 'Dan':1},
               1:{'went':2, 'ran':2, 'jogged':2},
               2:{'to':3, 'in':3},
               3:{'the':4, 'a':4},
               4:{'park':5}} 
words_dfa = DFA(transitions, s0=0, accept_states=[5])

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")
tokens_dfa = words_dfa.adjust_for_tokenizer(tokenizer)
```

Eventually, generation constraining is achieved by replacing the model's `beam_search` method with an adapted version (in `dfa_constrained_beam_search.py`) that
enforces every beam to follow the automaton transitions. The probability of vocabulary entries that are not accessible according to the dfa will be set to minus infinity.

For example:

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("t5-small")
from dfa_constrained_beam_search import set_model_beam_search_to_dfa_constrained
set_model_beam_search_to_dfa_constrained(model, tokens_dfa)

#   The previous two steps can equivalently be done in one call:
# set_model_beam_search_to_dfa_constrained(model, words_dfa, tokenizer) 
```

Other supported utility functions within the `DFA` class include:
* `DFA.from_slots` - for constructing a linear DFA out of a list of "slots", where each slot is represented by a list of allowed words / phrases.
* `DFA.concat_two` and `DFA.concat` - for concatenating two or more (linear) DFAs into a long DFA.
* `as_cyclic` - for converting a linear DFA into a cyclic one, by merging or connecting some end-states with the initial state. 


### QA-SRL

Our motivational use-case is [seq2seq-based QA-SRL parsing](https://huggingface.co/kleinay/qanom-seq2seq-model-joint). In that project, we finetune BART/T5 on the [Question-Answer driven Semantic Role Labeling](https://qasrl.org/) task. Given a verb or nominalization in context, the task is to generate Question-Answer pairs capturing the participants of the verbal event. 

To model the task using a seq2seq paradigm, the QAs are linearized into a single target sequence, using separators between QA pairs, between question and answer, and between multiple answers. Furthermore, QASRL questions must adhere a slot-based template, while answers could only be continuous spans copied from the input sentence. 

Check out the `qasrl_constrained_decoding.py` module to see how we leverage the `DFA` utilities for enforcing a valid QASRL output sequence.