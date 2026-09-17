"""Real tiny offline encoder, cached/live heads, split safety and exact resume."""
import copy
from unittest.mock import patch

import numpy as np
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from mission3_ver6 import neural
from mission3_ver6.ver4_runtime.neural import DialogueClassifier as LegacyClassifier


@pytest.fixture
def assets(tmp_path):
    torch.set_num_threads(1)
    vocabulary=["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]","alpha","beta","omega"]
    path=tmp_path/"vocab.txt"
    path.write_text("\n".join(vocabulary),encoding="utf-8")
    tokenizer=BertTokenizerFast(vocab_file=str(path),do_lower_case=False)
    configuration=BertConfig(vocab_size=len(vocabulary),hidden_size=12,num_hidden_layers=3,
                             num_attention_heads=3,intermediate_size=24,max_position_embeddings=32,
                             hidden_dropout_prob=.1,attention_probs_dropout_prob=.1)
    torch.manual_seed(7)
    legacy=LegacyClassifier(BertModel(configuration),dropout=.1,window_batch_size=2)
    initial=tmp_path/"fold_1.ckpt"
    torch.save({"format_version":1,"encoder_config":configuration.to_dict(),
                "architecture":{"num_labels":9,"dropout":.1,"window_batch_size":2},
                "model_state":legacy.state_dict(),"model_name":"local-tiny-bert","epoch":3,"seed":1051},initial)
    texts=["alpha "*(1+i%4)+" beta "*(i%3)+" omega" for i in range(15)]
    texts[0]="alpha "*23+"omega"
    texts[-1]=""
    documents=neural.tokenize_dialogues(texts,tokenizer,max_length=12,stride=3)
    targets=((np.arange(15)[:,None]+np.arange(9)[None,:])%3==0).astype(np.float32)
    legacy.eval()
    store=[]
    with torch.no_grad():
        for document in documents:
            windows=[]
            for window in document:
                batch={key:torch.tensor([value]) for key,value in window.items()}
                hidden=legacy.encoder(**batch).last_hidden_state[0].detach()
                content=torch.tensor([token not in neural._special_ids(tokenizer) for token in window["input_ids"]])
                windows.append({"hidden":hidden,"content_mask":content})
            store.append(windows)
    settings={"device":"cpu","head_epochs":2,"head_patience":4,"head_batch_size":4,
              "finetune_epochs":2,"finetune_top_layers":1,"attention_dim":8,"batch_size":2,
              "window_batch_size":2,"inference_batch_size":3,"gradient_accumulation_steps":2,
              "gradient_checkpointing":True,"dropout":.1,"learning_rate":1e-3,
              "head_learning_rate":2e-3,"finetune_head_learning_rate":1e-3,
              "warmup_ratio":0,"weight_decay":0,"layerwise_lr_decay":1,"log_every":0}
    return tokenizer,legacy,initial,documents,targets,store,settings


def train(assets,path,**overrides):
    tokenizer,_,initial,documents,targets,store,settings=assets
    return neural.train_model(documents[:9],targets[:9],documents[9:],targets[9:],tokenizer,
                              {**settings,**overrides.pop("config",{})},path,seed=42,
                              init_checkpoint_path=initial,run_signature=overrides.pop("run_signature","same-run"),
                              feature_store=store,fit_indices=np.arange(9),stop_indices=np.arange(9,15),
                              stop_groups=np.arange(6),**overrides)


def model_for(assets,head_type="laat",dependency=False):
    tokenizer,legacy,initial,_,targets,_,settings=assets
    base=neural._read_checkpoint(initial)
    profile={"head_type":head_type,"dependency":dependency}
    model=neural.DialogueClassifier(copy.deepcopy(legacy.encoder),head_type=head_type,
                                   dropout=.1,window_batch_size=2,attention_dim=8,
                                   special_token_ids=neural._special_ids(tokenizer),dependency=dependency)
    head=neural._head_from_base(base,profile,settings,targets[:9])
    model.head.load_state_dict(head.state_dict())
    return model


def test_legacy_head_copies_weights_and_cached_live_logits_match(assets):
    tokenizer,legacy,_,documents,_,store,_=assets
    model=model_for(assets,"legacy").eval()
    batch=neural.DialogueCollator(tokenizer)([{"windows":document} for document in documents[:3]])
    with torch.no_grad():
        expected=legacy(**batch)
        live=model(**batch)
        cached=model.head(store[:3])
    torch.testing.assert_close(live,expected,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(live,cached,atol=1e-6,rtol=1e-5)
    for key,value in legacy.state_dict().items():
        mapped=key if key.startswith("encoder.") else "head."+key
        torch.testing.assert_close(model.state_dict()[mapped],value,atol=0,rtol=0)


def test_legacy_summaries_avoid_token_reads_and_preserve_logits_and_gradients(assets):
    *_, store, _ = assets
    head = model_for(assets, "legacy").head.eval()
    summarized_head = copy.deepcopy(head)

    class SummaryStore:
        def __getitem__(self, index):
            raise AssertionError("Legacy trials must not reread token features")

        def get_legacy(self, index):
            return [{"legacy_summary": torch.cat([window["hidden"][0], window["hidden"].mean(0)])}
                    for window in store[index]]

    original = head(store[:3])
    batch = neural._cached_batch(SummaryStore(), np.arange(3), torch.device("cpu"), legacy=True)
    summarized = summarized_head(batch)
    torch.testing.assert_close(summarized, original, atol=0, rtol=0)
    original.square().sum().backward()
    summarized.square().sum().backward()
    for first, second in zip(head.parameters(), summarized_head.parameters()):
        torch.testing.assert_close(first.grad, second.grad, atol=0, rtol=0)
    predictions = neural._predict_head(summarized_head, SummaryStore(), np.arange(3), 2, torch.device("cpu"))
    np.testing.assert_allclose(predictions, original.detach().sigmoid().numpy(), atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("dependency",[False,True])
def test_direct_attention_masks_specials_tail_gradient_and_cached_live_equivalence(assets,dependency):
    tokenizer,_,_,documents,_,store,_=assets
    model=model_for(assets,dependency=dependency).eval()
    selected=[documents[0],documents[-1]]
    batch=neural.DialogueCollator(tokenizer)([{"windows":document} for document in selected])
    live=model(**batch)
    cached=model.head([store[0],store[-1]])
    torch.testing.assert_close(live,cached,atol=1e-6,rtol=1e-5)
    assert torch.isfinite(live).all()
    live.sum().backward()
    assert model.head.token_projection.weight.grad.abs().sum()>0
    assert model.head.token_queries.grad.abs().sum()>0
    assert model.head.label_weights.grad.abs().sum()>0
    assert model.encoder.embeddings.word_embeddings.weight.grad[tokenizer.convert_tokens_to_ids("omega")].abs().sum()>0
    altered=copy.deepcopy([store[0],store[-1]])
    for document in altered:
        for window in document:
            window["hidden"][~window["content_mask"]]+=1000
    torch.testing.assert_close(model.head(altered),cached,atol=0,rtol=0)
    altered=copy.deepcopy([store[0],store[-1]])
    altered[0][-1]["hidden"][1]+=10
    assert not torch.equal(model.head(altered)[0],cached[0])
    # Real cache stores FP16; its rounding difference is bounded, not bit-exact.
    rounded=[[{"hidden":w["hidden"].half().float(),"content_mask":w["content_mask"]} for w in doc]
             for doc in [store[0],store[-1]]]
    torch.testing.assert_close(model.head(rounded),cached,atol=2e-4,rtol=2e-3)


def test_only_selected_top_blocks_train_and_frozen_dropout_stays_disabled(assets):
    tokenizer,legacy,_,documents,_,_,settings=assets
    model=model_for(assets)
    original={name:value.clone() for name,value in model.named_parameters()}
    neural._configure_finetune(model,neural._settings(settings))
    neural._train_mode(model,settings)
    blocks=model.encoder.encoder.layer
    assert not model.encoder.embeddings.training
    assert not blocks[0].training and not blocks[1].training and blocks[2].training
    assert model.head.training
    optimizer=neural._optimizer(model,{**settings,"head_learning_rate":settings["finetune_head_learning_rate"]})
    included={id(p) for group in optimizer.param_groups for p in group["params"]}
    assert included=={id(p) for p in model.parameters() if p.requires_grad}
    batch=neural.DialogueCollator(tokenizer)([{"windows":document} for document in documents[:2]])
    model(**batch).sum().backward()
    optimizer.step()
    for name,parameter in model.named_parameters():
        if name.startswith("encoder.embeddings") or name.startswith("encoder.encoder.layer.0") or name.startswith("encoder.encoder.layer.1"):
            assert parameter.grad is None
            torch.testing.assert_close(parameter,original[name],atol=0,rtol=0)
    assert any(not torch.equal(p,original[name]) for name,p in model.named_parameters() if name.startswith("encoder.encoder.layer.2"))


def _without_timing(history):
    return [{key:value for key,value in row.items() if key!="minutes"} for row in history]


def assert_nested_equal(first,second):
    if isinstance(first,torch.Tensor):
        torch.testing.assert_close(first,second,rtol=0,atol=0)
    elif isinstance(first,dict):
        assert first.keys()==second.keys()
        for key in first:
            assert_nested_equal(first[key],second[key])
    elif isinstance(first,(list,tuple)):
        assert len(first)==len(second)
        for a,b in zip(first,second):
            assert_nested_equal(a,b)
    else:
        assert first==second


@pytest.mark.parametrize("interrupt_phase",["head","finetune"])
def test_real_all_profile_training_and_atomic_exact_rng_resume(assets,tmp_path,interrupt_phase):
    full=tmp_path/"full"/"candidate_best.ckpt"
    resumed=tmp_path/"resumed"/"candidate_best.ckpt"
    with patch.object(neural.AutoModel,"from_pretrained",side_effect=AssertionError("network forbidden")):
        report=train(assets,full)
    assert len(report["head_trials"])==6
    assert report["selected_profile"] in neural.PROFILES
    assert [x["phase"] for x in report["history"]]==["selected_head","finetune","finetune"]
    original_save=neural._atomic_save

    def interrupted_save(state,path):
        original_save(state,path)
        head=state.get("format")==neural.HEAD_FORMAT and state.get("profile")=="D" and state.get("completed_epochs")==1
        fine=state.get("format")==neural.LAST_FORMAT and state.get("completed_epochs")==1
        if head if interrupt_phase=="head" else fine:
            raise KeyboardInterrupt("Committed epoch interruption")

    with patch.object(neural,"_atomic_save",side_effect=interrupted_save):
        with pytest.raises(KeyboardInterrupt):
            train(assets,resumed)
    resumed_report=train(assets,resumed)
    assert resumed_report["resumed"]
    assert resumed_report["selected_profile"]==report["selected_profile"]
    assert _without_timing(resumed_report["history"])==_without_timing(report["history"])
    assert resumed_report["best_stop_macro_f1"]==report["best_stop_macro_f1"]
    full_last=neural._read_checkpoint(full.with_name("candidate_best_last.ckpt"))
    resumed_last=neural._read_checkpoint(resumed.with_name("candidate_best_last.ckpt"))
    for key in ("model_state","optimizer","scheduler","scaler","rng"):
        assert_nested_equal(full_last[key],resumed_last[key])
    for profile in neural.PROFILES:
        a=neural._read_checkpoint(full.parent/"head_trials"/f"profile_{profile}.ckpt")
        b=neural._read_checkpoint(resumed.parent/"head_trials"/f"profile_{profile}.ckpt")
        assert all(not name.startswith("encoder.") for name in a["model_state"])
        for key in ("model_state","best_state","optimizer","scaler","rng","best_score"):
            assert_nested_equal(a[key],b[key])
    tokenizer,_,_,documents,_,_,settings=assets
    np.testing.assert_array_equal(neural.predict_checkpoint(full,documents,tokenizer,settings),
                                  neural.predict_checkpoint(resumed,documents,tokenizer,settings))
    assert neural.predict_checkpoint(full,[],tokenizer,settings).shape==(0,9)
    # Completed-run path must not perform any training or model construction.
    with patch.object(neural,"_run_head_trial",side_effect=AssertionError("already complete")):
        assert train(assets,resumed)["resumed"]
    with pytest.raises(ValueError,match="run signature"):
        train(assets,resumed,run_signature="different")
    with pytest.raises(ValueError,match="config fingerprint"):
        train(assets,resumed,config={"head_learning_rate":.01})
    assert not list(tmp_path.rglob("*.tmp"))


def test_head_rdrop_runs_twice_but_default_finetune_runs_encoder_once(assets):
    tokenizer,_,_,documents,targets,store,settings=assets
    model=model_for(assets).train()
    profile=neural.PROFILES["E"]
    criterion=neural.MultilabelLoss("bce",targets[:9])
    labels=torch.tensor(targets[:2])
    with patch.object(model.head,"forward",wraps=model.head.forward) as forward:
        neural._loss(model.head,store[:2],labels,criterion,profile)
        assert forward.call_count==2
    batch=neural.DialogueCollator(tokenizer)([{"windows":document} for document in documents[1:3]])
    with patch.object(model,"forward",wraps=model.forward) as forward:
        neural._loss(model,batch,labels,criterion,profile,head_phase=False)
        assert forward.call_count==1


def test_t4_precision_uses_native_check_and_fp16_scaling_policy():
    with patch.object(torch.cuda,"is_available",return_value=True),patch.object(torch.cuda,"is_bf16_supported",return_value=False) as check:
        assert not neural.native_bf16_supported()
        check.assert_called_once_with(including_emulation=False)
    with patch.object(torch.cuda,"is_available",return_value=True),patch.object(torch.cuda,"is_bf16_supported",side_effect=TypeError),patch.object(torch.cuda,"get_device_capability",return_value=(7,5)):
        assert not neural.native_bf16_supported()


@pytest.mark.parametrize("interrupt_phase", ["head", "finetune"])
def test_changed_execution_identity_blocks_resume_before_more_training(assets, tmp_path, interrupt_phase):
    path = tmp_path / "candidate_best.ckpt"
    original_save = neural._atomic_save

    def interrupt(state, destination):
        original_save(state, destination)
        if state.get("format") == (neural.HEAD_FORMAT if interrupt_phase == "head" else neural.LAST_FORMAT):
            raise KeyboardInterrupt("Committed epoch")

    with patch.object(neural, "_atomic_save", side_effect=interrupt):
        with pytest.raises(KeyboardInterrupt):
            train(assets, path, config={"profiles": ["D"]})
    # CPU execution supplies the actual tiny test. Mock a GPU/precision change
    # only in metadata, so rejection is tested without accelerator allocation.
    changed = {**neural.execution_identity("cpu"), "gpu": "different GPU family", "amp_dtype": "bfloat16"}
    with patch.object(neural, "execution_identity", return_value=changed), patch.object(neural, "_cached_batch", side_effect=AssertionError("must reject before training")):
        with pytest.raises(ValueError, match="same execution environment"):
            train(assets, path, config={"profiles": ["D"]})


@pytest.mark.parametrize("config",[{"finetune_top_layers":-2},{"head_epochs":0},{"head_batch_size":True},{"profiles":["D","D"]}])
def test_invalid_settings_rejected(config):
    with pytest.raises(ValueError):
        neural._settings(config)


def test_missing_shared_features_rejected(assets,tmp_path):
    tokenizer,_,initial,documents,targets,_,settings=assets
    with pytest.raises(ValueError,match="shared feature_store"):
        neural.train_model(documents[:9],targets[:9],documents[9:],targets[9:],tokenizer,settings,
                           tmp_path/"bad.ckpt",42,initial,"test")
