from typing import Optional
import json
from argparse import Namespace
from pathlib import Path
from transformers import Text2TextGenerationPipeline, AutoModelForSeq2SeqLM, AutoTokenizer

def get_markers_for_model(is_t5_model: bool) -> Namespace:
    special_tokens_constants = Namespace() 
    if is_t5_model:
        # T5 model have 100 special tokens by default
        special_tokens_constants.separator_input_question_predicate = "<extra_id_1>"
        special_tokens_constants.separator_output_answers = "<extra_id_3>"
        special_tokens_constants.separator_output_questions = "<extra_id_5>"  # if using only questions
        special_tokens_constants.separator_output_question_answer = "<extra_id_7>"
        special_tokens_constants.separator_output_pairs = "<extra_id_9>"
        special_tokens_constants.predicate_generic_marker = "<extra_id_10>" 
        special_tokens_constants.predicate_verb_marker = "<extra_id_11>" 
        special_tokens_constants.predicate_nominalization_marker = "<extra_id_12>" 

    else:
        special_tokens_constants.separator_input_question_predicate = "<question_predicate_sep>"
        special_tokens_constants.separator_output_answers = "<answers_sep>"
        special_tokens_constants.separator_output_questions = "<question_sep>"  # if using only questions
        special_tokens_constants.separator_output_question_answer = "<question_answer_sep>"
        special_tokens_constants.separator_output_pairs = "<qa_pairs_sep>"
        special_tokens_constants.predicate_generic_marker = "<predicate_marker>" 
        special_tokens_constants.predicate_verb_marker = "<verbal_predicate_marker>" 
        special_tokens_constants.predicate_nominalization_marker = "<nominalization_predicate_marker>" 
    return special_tokens_constants

def load_trained_model(name_or_path):
    import huggingface_hub as HFhub
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(name_or_path)  
    # load preprocessing_kwargs from the model repo on HF hub, or from the local model directory
    kwargs_filename = None
    if name_or_path.startswith("kleinay/") and 'preprocessing_kwargs.json' in HFhub.list_repo_files(name_or_path):
        kwargs_filename = HFhub.hf_hub_download(repo_id=name_or_path, filename="preprocessing_kwargs.json")
    elif Path(name_or_path).is_dir() and (Path(name_or_path) / "experiment_kwargs.json").exists():
        kwargs_filename = Path(name_or_path) / "experiment_kwargs.json"
    
    if kwargs_filename:
        preprocessing_kwargs = json.load(open(kwargs_filename)) 
        # integrate into model.config (for decoding args, e.g. "num_beams"), and save also as standalone object for preprocessing
        model.config.preprocessing_kwargs = Namespace(**preprocessing_kwargs)
        model.config.update(preprocessing_kwargs)
    return model, tokenizer


class QASRL_Pipeline(Text2TextGenerationPipeline):
    def __init__(self, model_repo: str, **kwargs):
        model, tokenizer = load_trained_model(model_repo)
        super().__init__(model, tokenizer, framework="pt")
        self.is_t5_model = "t5" in model.config.model_type
        self.special_tokens = get_markers_for_model(self.is_t5_model)
        # self.preprocessor = preprocessing.Preprocessor(model.config.preprocessing_kwargs, self.special_tokens)
        self.data_args = model.config.preprocessing_kwargs 
        # backward compatibility - default keyword values implemeted in `run_summarization`, thus not saved in `preprocessing_kwargs`
        if "predicate_marker_type" not in vars(self.data_args):
            self.data_args.predicate_marker_type = "generic"
        if "use_bilateral_predicate_marker" not in vars(self.data_args):
            self.data_args.use_bilateral_predicate_marker = True
        if "append_verb_form" not in vars(self.data_args):
            self.data_args.append_verb_form = True
        
        
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs, forward_kwargs, postprocess_kwargs = {}, {}, {} # super()._sanitize_parameters(**kwargs)
        forward_kwargs.update(kwargs.get("generate_kwargs", dict()))
        forward_kwargs.update(kwargs.get("model_kwargs", dict()))
        preprocess_keywords = ("predicate_marker", "predicate_type", "verb_form")
        for key in preprocess_keywords:
            if key in kwargs:
                preprocess_kwargs[key] = kwargs[key]

        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, inputs, predicate_marker="<predicate>", predicate_type=None, verb_form=None):
        # Here, inputs is string or list of strings; apply string postprocessing
        if isinstance(inputs, str):
            processed_inputs = self._preprocess_string(inputs, predicate_marker, predicate_type, verb_form)
        elif hasattr(inputs, "__iter__"):
            processed_inputs = [self._preprocess_string(s, predicate_marker, predicate_type, verb_form) for s in inputs]
        else:
            raise ValueError("inputs must be str or Iterable[str]")
        # Now pass to super.preprocess for tokenization
        return super().preprocess(processed_inputs)
    
    def _preprocess_string(self, seq: str, predicate_marker: str, predicate_type: Optional[str], verb_form: Optional[str]) -> str:
        sent_tokens = seq.split(" ")
        assert predicate_marker in sent_tokens, f"Input sentence must include a predicate-marker token ('{predicate_marker}') before the target predicate word"
        predicate_idx = sent_tokens.index(predicate_marker)
        sent_tokens.remove(predicate_marker)
        sentence_before_predicate = " ".join([sent_tokens[i] for i in range(predicate_idx)])
        predicate = sent_tokens[predicate_idx]
        sentence_after_predicate = " ".join([sent_tokens[i] for i in range(predicate_idx+1, len(sent_tokens))])
        
        if self.data_args.predicate_marker_type == "generic":
            predicate_marker = self.special_tokens.predicate_generic_marker    
        #  In case we want special marker for each predicate type: """
        elif self.data_args.predicate_marker_type == "pred_type":
            assert predicate_type is not None, "For this model, you must provide the `predicate_type` either when initializing QASRL_Pipeline(...) or when applying __call__(...) on it"
            assert predicate_type in ("verbal", "nominal"), f"`predicate_type` must be either 'verbal' or 'nominal'; got '{predicate_type}'"
            predicate_marker = {"verbal": self.special_tokens.predicate_verb_marker , 
                                "nominal": self.special_tokens.predicate_nominalization_marker 
                                }[predicate_type]

        if self.data_args.use_bilateral_predicate_marker:
            seq = f"{sentence_before_predicate} {predicate_marker} {predicate} {predicate_marker} {sentence_after_predicate}"
        else:
            seq = f"{sentence_before_predicate} {predicate_marker} {predicate} {sentence_after_predicate}"

        # embed also verb_form
        if self.data_args.append_verb_form and verb_form is None:
            raise ValueError(f"For this model, you must provide the `verb_form` of the predicate when applying __call__(...)")
        elif self.data_args.append_verb_form:
            seq = f"{seq} {self.special_tokens.separator_input_question_predicate} {verb_form} "
        else:
            seq = f"{seq} "
    
        # append source prefix (for t5 models)
        prefix = self._get_source_prefix(predicate_type)
        
        return prefix + seq
    
    def _get_source_prefix(self, predicate_type: Optional[str]):
        if not self.is_t5_model or self.data_args.source_prefix is None:
            return ''
        if "Generate QAs for <predicate_type> QASRL: " in self.data_args.source_prefix:
            if predicate_type is None:
                raise ValueError("source_prefix includes 'Generate QAs for <predicate_type> QASRL: ' but input has no `predicate_type`.")
            if self.data_args.source_prefix == "Generate QAs for <predicate_type> QASRL: ": # backwrad compatibility - "Generate QAs for <predicate_type> QASRL: " alone was a sign for a longer prefix 
                return f"Generate QAs for {predicate_type} QASRL: "
            else:
                return self.data_args.source_prefix.replace("Generate QAs for <predicate_type> QASRL: ", predicate_type)
        else:
            return self.data_args.source_prefix
        
    
    def _forward(self, *args, **kwargs):
        outputs = super()._forward(*args, **kwargs)
        return outputs


    def postprocess(self, model_outputs):
        output_seq = self.tokenizer.decode(
            model_outputs["output_ids"][0],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        output_seq = output_seq.strip(self.tokenizer.pad_token).strip(self.tokenizer.eos_token).strip()
        qa_subseqs = output_seq.split(self.special_tokens.separator_output_pairs)
        qas = [self._postrocess_qa(qa_subseq) for qa_subseq in qa_subseqs]
        return {"generated_text": output_seq,
                "QAs": qas}
        
    def _postrocess_qa(self, seq: str) -> str:
        # split question and answers
        if self.special_tokens.separator_output_question_answer in seq:
            question, answer = seq.split(self.special_tokens.separator_output_question_answer)[:2]
        else:
            print("invalid format: no separator between question and answer found...")
            return None
            # question, answer = seq, '' # Or: backoff to only question  
        # skip "_" slots in questions
        question = ' '.join(t for t in question.split(' ') if t != '_')
        answers = [a.strip() for a in answer.split(self.special_tokens.separator_output_answers)]
        return {"question": question, "answers": answers}
    
    
if __name__ == "__main__":
    pipe = QASRL_Pipeline("kleinay/qanom-seq2seq-model-baseline")
    res1 = pipe("The student was interested in Luke 's <predicate> research about see animals .", verb_form="research", predicate_type="nominal")
    res2 = pipe(["The doctor was interested in Luke 's <predicate> treatment .",
                 "The Veterinary student was interested in Luke 's <predicate> treatment of sea animals ."], verb_form="treat", predicate_type="nominal", num_beams=10)
    res3 = pipe("A number of professions have <predicate> developed that specialize in the treatment of mental disorders .", verb_form="develop", predicate_type="verbal")
    print(res1)
    print(res2)
    print(res3)
    