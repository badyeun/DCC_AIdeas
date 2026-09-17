"""Turn separators preserve text-only inputs and the complete dialogue tail."""
from types import SimpleNamespace

import pytest
from transformers import BertTokenizerFast

from mission3_ver7.data import candidate_texts
from mission3_ver7.ver4_runtime.data import dialogue_text, target_vector
from mission3_ver7.neural import tokenize_dialogues


def test_turn_boundaries_use_existing_sep_without_speaker_or_labels(tmp_path):
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "alpha", "beta", "omega"]), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(path), do_lower_case=False)
    source = {"symptom": ["고열"], "speaker": "metadata", "utterances": [
        {"text": " alpha\t beta ", "speaker": "caller"},
        {"text": " omega ", "speaker": "dispatcher"}]}
    text = dialogue_text(source)
    changed = {"symptom": ["두통"], "speaker": "unrelated", "utterances": [
        {"text": " alpha\t beta ", "speaker": 193}, {"text": " omega ", "speaker": 99}]}
    assert target_vector(source) != target_vector(changed)
    count = len(tokenizer)
    actual = candidate_texts([text, dialogue_text(changed)], tokenizer)
    assert actual == ["alpha beta [SEP] omega"] * 2
    assert len(tokenizer) == count
    ids = tokenize_dialogues(actual, tokenizer, max_length=12, stride=3)[0][0]["input_ids"]
    assert ids.count(tokenizer.sep_token_id) == 2  # Internal turn and ordinary final SEP.
    assert tokenizer.convert_tokens_to_ids("omega") in ids


def test_empty_missing_and_other_symptom_calls_are_uniform():
    tokenizer = SimpleNamespace(sep_token="[SEP]")
    records = [{}, {"symptom": []}, {"symptom": ["not one of nine"]}]
    assert candidate_texts([dialogue_text(row) for row in records], tokenizer) == ["", "", ""]
    assert candidate_texts([" alpha\n\n  beta  \t omega \n", "alpha", ""], tokenizer) == [
        "alpha [SEP] beta omega", "alpha", ""]
    assert candidate_texts([], tokenizer) == []


def test_invalid_nontext_inputs_and_missing_sep_fail():
    with pytest.raises(TypeError, match="never annotations"):
        candidate_texts([{"text": "alpha", "symptom": []}], SimpleNamespace(sep_token="[SEP]"))
    with pytest.raises(TypeError, match="sequence"):
        candidate_texts("alpha", SimpleNamespace(sep_token="[SEP]"))
    with pytest.raises(ValueError, match="sep_token"):
        candidate_texts(["alpha"], SimpleNamespace(sep_token=None))
