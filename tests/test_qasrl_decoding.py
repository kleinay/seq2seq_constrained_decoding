import transformers
import os, sys

repo_base_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(repo_base_dir, "src"))
sys.path.insert(0, os.path.join(repo_base_dir))
from constrained_decoding.qasrl import *


def test_qasrl_question_dfa():
    from tests.test_helpers.pipeline import QASRL_Pipeline
    pipe = QASRL_Pipeline("kleinay/qanom-seq2seq-model-joint")
    tokenizer = pipe.tokenizer
    qdfa = get_qasrl_question_dfa(constrain_verb=False)
    tqdfa = qdfa.adjust_for_tokenizer(tokenizer)
    def apply(question):
        return tqdfa(tokenizer.tokenize(question))
    # debugging with prints
    print(apply("what did someone use _"))
    print(apply("what did someone use _ on"))
    print(apply("how long did "))
    print(apply("how"))
    print(apply("how _ _ try _ _ _ ?"))
    # tests
    assert apply("how much _ _ try _ _ _ ?")[2]
    assert apply("how _ _ umbrella _ _ do something ?")[2]
    assert apply("how _ _ verb someone _ something ?")[2]
    print("test_qasrl_question_dfa() passed successfully")


def test_qasrl_answer_dfa():
    from tests.test_helpers.pipeline import QASRL_Pipeline
    pipe = QASRL_Pipeline("kleinay/qanom-seq2seq-model-joint")
    tokenizer = pipe.tokenizer
    # taking a sentence with some multi-token words
    sentence = "The two tails are suspects of murdering many ozone 2050 customers ceasefire . "
    tokenized_sentence = tokenizer.tokenize(sentence)
    # answer dfa is a copy dfa 
    sep = "<extra_id_9>"
    dfa = get_qasrl_answer_dfa(tokenized_sentence)
    def apply(seq):
        return dfa(tokenizer.tokenize(seq)) 
    assert apply("two tails")[2] 
    assert apply("two tail")[2] == False 
    assert apply("s of murdering")[2] == False 
    assert apply("many ozone")[2]  
    assert apply("many o")[2] == False  
    assert apply("The two tails")[2]
    assert apply("ceasefire")[2]
    
    
def test_qasrl_answers_dfa():
    from tests.test_helpers.pipeline import QASRL_Pipeline
    pipe = QASRL_Pipeline("kleinay/qanom-seq2seq-model-joint")
    tokenizer = pipe.tokenizer
    # taking a sentence with some multi-token words
    sentence = "The two tails are suspects of murdering many ozone 2050 customers ceasefire . "
    tokenized_sentence = tokenizer.tokenize(sentence)
    # answer dfa is a copy dfa 
    sep = "<extra_id_9>"
    dfa = get_qasrl_answer_dfa(tokenized_sentence)
    dfa = dfa.as_cyclic(bridge=sep)
    def apply(seq):
        return dfa(tokenizer.tokenize(seq)) 
    assert apply("two tails")[2] 
    assert apply(f"{sep} two tails")[2] == False
    assert apply(f"two tails {sep} suspects of murdering")[2]
    assert apply(f"ceasefire {sep} 2050 customers")[2]
    assert apply(f"ceasefire {sep} 2050 customers {sep}")[2] == False


def test_full_qasrl_dfa():
    from tests.test_helpers.pipeline import QASRL_Pipeline
    pipe = QASRL_Pipeline("kleinay/qanom-seq2seq-model-joint")
    tokenizer = pipe.tokenizer
    sentence = "The doctor was interested in Luke's treatment yesterday given by the other doctor ."
    dfa = get_qasrl_full_sequence_dfa(sentence, tokenizer, pipe.special_tokens, convert_to_word_ids=False)
    def apply(seq):
        return dfa(tokenizer.tokenize(seq))  
    q = "how _ _ verb someone _ something ?" 
    assert apply("how much _ _ try _ _ _ ?<extra_id_7> the other doctor </s>")[2]
    # assert apply("how much _ _ try _ _ _ ?<extra_id_7> the other doctor<extra_id_3> Luke")[2]
    assert apply("how _ _ umbrella _ _ do something ?<extra_id_7> the other<extra_id_9>") == (True, 0, False)
    assert apply(f"{q}<extra_id_7> the other<extra_id_9>{q}<extra_id_7> the other")[2]
    print("test_full_qasrl_dfa() passed successfully")


def test_redundance_answer_disposer():
    convert_to_word_ids = True
    from tests.test_helpers.pipeline import QASRL_Pipeline
    pipe = QASRL_Pipeline("kleinay/qanom-seq2seq-model-joint")
    tokenizer = pipe.tokenizer
    sentence = "The doctor was interested in Luke 's treatment yesterday given by the other doctor ."
    dfa = get_qasrl_full_sequence_dfa(sentence, tokenizer, pipe.special_tokens, convert_to_word_ids=convert_to_word_ids)
    dfa_orig = dfa.copy()
    set_as_redundance_answer_disposer_dfa(dfa, pipe.special_tokens, tokenizer=tokenizer, convert_to_word_ids=convert_to_word_ids)
    def apply(seq):
        if convert_to_word_ids:
            return dfa(tokenizer.encode(seq)[:-1]) 
        else:
            return dfa(tokenizer.tokenize(seq)) 
    question = "how _ _ verb someone _ something ?"   
    assert apply(f"{question}<extra_id_7> by the other doctor<extra_id_3> was interested")[2]
    assert apply(f"{question}<extra_id_7> by the other doctor<extra_id_3> other") == (False, '<2>_0', False)
    assert apply(f"{question}<extra_id_7> by the other doctor<extra_id_3> the")[2]  # repeating a stop-word is OK
    assert apply(f"{question}<extra_id_7> by the other doctor<extra_id_9> {question}")[0] 
    assert apply(f"{question}<extra_id_7> by the other doctor<extra_id_9> {question}<extra_id_7> by the")[2] 
    assert apply(f"{question}<extra_id_7> by the other doctor<extra_id_3> was<extra_id_3> interested")[2]
    assert apply(f"{question}<extra_id_7> by the other<extra_id_9> {question}<extra_id_7> doctor<extra_id_3> doctor")[0] == False
    
    print("test_redundance_answer_disposer() passed successfully")

   
def test_get_copy_dfa():
    sentence = "abcabdssabcr"
    tok_source_seq: List[str] = list(sentence) # ['a', 'b', 'c', 'a', 'b', 'd', 's', 's', 'a', 'b', 'c', 'r'] 
    # test with terminate symbol
    dfa = get_copy_dfa(tok_source_seq, '<s>')
    subseqs = [(i,tuple(tok_source_seq[i:j])) for i in range(len(tok_source_seq)-1) for j in range(i+1, len(tok_source_seq))]
    for i,subseq in subseqs:
        assert dfa(subseq + ('<s>',)) == (True, '<TERMINATE>', True), f"substring {subseq} failed"
        assert not dfa(subseq)[2] 
    assert not dfa("aba")[0]
    # test without terminate symbol
    dfa = get_copy_dfa(tok_source_seq)
    for i,subseq in subseqs:
        assert dfa(subseq)[2], f"substring {subseq} failed"
        assert not dfa(subseq + ('c', 'b'))[2] 
    print("test_get_copy_dfa() passed successfully")
  
def test_autoregressive_on_model():
    # Full test on real model
    from tests.test_helpers.pipeline import QASRL_Pipeline
    pipe = QASRL_Pipeline("kleinay/qanom-seq2seq-model-joint")
    sentence = "The doctor was interested to know about Luke 's bio-feedback treatment given by the nurse yesterday."
    # sentence = "The student was interested in Luke 's research about sea animals ."
    pipe_kwargs = dict(
        inputs="The doctor was interested to know about Luke 's bio-feedback <predicate> treatment given by the nurse yesterday.",
        verb_form="treat", 
        # inputs="The student was interested in Luke 's <predicate> research about sea animals .",
        # verb_form="research", 
        predicate_type="nominal")
    print("Baseline:")
    print(pipe(**pipe_kwargs))
    
    # Prior method - using a single `dfa` object
    # dfa = get_qasrl_full_sequence_dfa(sentence, pipe.tokenizer, pipe.special_tokens)
    # # enable special dfa-constrained beam search
    # set_decoding_to_dfa_constrained(pipe.model, dfa, pipe.tokenizer)
    
    # More generic - using a dfa_factory
    def dfa_factory(token_ids): 
        sentence = pipe.tokenizer.decode(token_ids, skip_special_tokens=True)
        print(sentence)
        return get_qasrl_full_sequence_dfa(sentence, pipe.tokenizer, pipe.special_tokens)
    # enable special dfa-constrained beam search
    set_decoding_to_dfa_constrained(pipe.model, dfa_factory=dfa_factory, tokenizer=pipe.tokenizer)
    
    # play with additional `generate` kwargs, including LogitsProcessors
    logits_processors = transformers.LogitsProcessorList([
        transformers.NoRepeatNGramLogitsProcessor(10),
        # transformers.TemperatureLogitsWarper(temperature=5),
        # transformers.RepetitionPenaltyLogitsProcessor(penalty=5.),  # make the beams different from one another
        transformers.ForcedEOSTokenLogitsProcessor(max_length=100, eos_token_id=pipe.tokenizer.vocab[pipe.tokenizer.eos_token]),
        # transformers.MinLengthLogitsProcessor(min_length=15, eos_token_id=tokenizer.vocab[tokenizer.eos_token]), # collision with defaults
    ])  
    generate_kwargs = dict(
                        #    max_length=80,
                        #    min_length=80,
                           num_beams=5,
                           num_return_sequences=5,
                        #    logits_processor=logits_processors,
                        #    early_stoping=True,
                            ) 
    
    print("Constrained:")
    print(pipe(**pipe_kwargs, generate_kwargs=generate_kwargs))
    
    # # using model.generate directly
    # pipe_input = "The doctor was interested to know about Luke 's bio-feedback <predicate> treatment given by the nurse yesterday."
    # input_seq = pipe.preprocess(pipe_input, predicate_type="nominal", verb_form="treat")['input_ids']

def test_full_logitProcessor():
    from tests.test_helpers.pipeline import QASRL_Pipeline
    pipe = QASRL_Pipeline("kleinay/qanom-seq2seq-model-joint")
    sentence = "The doctor was interested to know about Luke 's bio-feedback treatment given by the nurse yesterday."
    # sentence = "The student was interested in Luke 's research about sea animals ."
    pipe_kwargs = dict(
        inputs="The doctor was interested to know about Luke 's bio-feedback <predicate> treatment given by the nurse yesterday.",
        verb_form="treat", 
        # inputs="The student was interested in Luke 's <predicate> research about sea animals .",
        # verb_form="research", 
        predicate_type="nominal")
    print("Baseline:")
    print(pipe(**pipe_kwargs))
    
    # Prior method - using a single `dfa` object
    dfa = get_qasrl_full_sequence_dfa(sentence, pipe.tokenizer, pipe.special_tokens)
    #TODO

if __name__ == "__main__":
    # Unit Testing
    
    test_get_copy_dfa()
    test_redundance_answer_disposer()
    test_full_qasrl_dfa()
    test_qasrl_answer_dfa() # just the copying mechanism
    test_qasrl_answers_dfa() # 
    test_autoregressive_on_model()
    test_full_logitProcessor()
    
