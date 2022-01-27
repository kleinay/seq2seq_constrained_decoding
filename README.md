# seq2seq_constrained_decoding

This project includes constrained-decoding utilities for *structured text generation* using Huggingface seq2seq models.

## Use cases

Several use-cases leverage pretrained sequence-to-sequence models, such as BART or T5, 
for generating a (maybe partially) structured text sequence. 
For example, you may want to finetine the model to generate a set of key-points summarizing an input paragraph, or to produce a text sequence that follows some strict regaulrities. 

Below we detail about the use-cases supported by this project. 


### QA-SRL

Our motivational use-case is [seq2seq-based QA-SRL parsing](https://huggingface.co/kleinay/qanom-seq2seq-model-joint). In that project, we finetune BART/T5 on the [Question-Answer driven Semantic Role Labeling](https://qasrl.org/) task. Given a verb or nominalization in context, the task is to generate Question-Answer pairs capturing the participants of the verbal event. The QAs are linearized into a single target sequence, using separators between QA pairs, between question and answer, and between multiple answers. Furthermore, QASRL questions must adhere a slot based template, while answer could only be continuous spans copied from the input sentence.    




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
* `DFA.concat_two` and `DFA.concat` - for concatenating two (linear) DFAs into a long DFA.
* `as_cyclic` - for converting a linear DFA into a cyclic one, by merging or connecting some end-states with the initial state. 