# seq2seq_constrained_decoding

Constrained decoding utilities for text generation using Huggingface seq2seq models.


## DFA-based constrained decoding

The most powerful and generic constrained decoding algorithm is using a *Deterministic Finite Automata* (DFA).
You can instanciate a DFA with the `dfa.DFA` class, defined over a dictionary of dictionaries. 
For example, the following represents an automaton that accepts only binary numbers that are multiples of 3 (see [illustration](https://en.wikipedia.org/wiki/Deterministic_finite_automaton#/media/File:DFA_example_multiplies_of_3.svg) in the Wikipedia article on DFAs): 
```python
from dfa import DFA
transitions = {0:{'0':0, '1':1},
               1:{'0':2, '1':0},
               2:{'0':1, '1':2}} 
dfa = DFA(transitions, s0=0, accept_states=[0])
```

For defining constrained decoding using a DFA, the automaton's alphabet should correspond tokens in the model's vocabulry.
The `DFA` class supports translating a dfa that uses regular words or phrases as alphabet into a tokenizer-adjusted dfa - 

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

#   The two last steps can equivalently be achieved with:
# set_model_beam_search_to_dfa_constrained(model, words_dfa, tokenizer) 
```
