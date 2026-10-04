from src.config import PipelineConfig
from src.reproduce import _load_probe_metrics


def test_pipeline_excludes_withdrawn_comparison_and_includes_sequence_inference():
    from src.reproduce import PIPELINE_STEPS
    assert "cross_model_tokenization_comparison" not in PIPELINE_STEPS
    assert "sequence_genomic_controls" in PIPELINE_STEPS
    assert "sequence_classifier_intervals" in PIPELINE_STEPS


def test_encoder_falls_back_to_native_hooks_when_lens_encoder_is_unavailable(monkeypatch):
    from src import modeling
    from src.config import ModelConfig
    import torch
    model=torch.nn.Sequential(torch.nn.Linear(2,2))
    tokenizer=object()
    monkeypatch.setattr(modeling,"load_hf_components",lambda config:(tokenizer,model,"cpu"))
    def unavailable():
        raise ImportError("HookedEncoder unavailable")
    monkeypatch.setattr(modeling,"_select_hooked_encoder_class",unavailable)
    bundle=modeling.load_hooked_encoder(ModelConfig())
    assert bundle.hf_model is model and bundle.hooked_model.model is model
    assert bundle.instrumentation_backend=="huggingface_forward_hooks"
    assert "HookedEncoder unavailable" in bundle.instrumentation_error


def test_remote_compatibility_patch_uses_the_requested_revision(monkeypatch):
    from types import SimpleNamespace
    from transformers import dynamic_module_utils
    from src.modeling import _patch_remote_masked_lm_class
    calls=[]
    class RemoteModel:
        pass
    def resolve(class_ref, model_name, **kwargs):
        calls.append((class_ref,model_name,kwargs))
        return RemoteModel
    monkeypatch.setattr(dynamic_module_utils,"get_class_from_dynamic_module",resolve)
    _patch_remote_masked_lm_class("model",SimpleNamespace(auto_map={"AutoModelForMaskedLM":"module.Model"}),"pinned-sha")
    assert calls==[("module.Model","model",{"revision":"pinned-sha"})]
    assert RemoteModel.all_tied_weights_keys=={}


def test_failed_steps_are_recorded_with_their_actual_name_and_error(tmp_path,monkeypatch):
    from dataclasses import replace
    import json
    import pytest
    from src import reproduce
    config=PipelineConfig()
    config=replace(config,paths=replace(config.paths,results_dir=tmp_path,manifests_dir=tmp_path/"manifests"))
    monkeypatch.setattr(reproduce.PipelineConfig,"ensure_paths",lambda self:self.paths.manifests_dir.mkdir())
    def fail(*args,**kwargs):
        raise ValueError("deliberate ingestion failure")
    monkeypatch.setattr(reproduce,"ingest_hf_downstream_tasks",fail)
    with pytest.raises(ValueError):
        reproduce.run_pipeline(config)
    manifest=json.loads((tmp_path/"pipeline_run.json").read_text())
    assert manifest["status"]=="failed"
    assert manifest["steps"][-1]["name"]=="ingest_hf_downstream"
    assert manifest["steps"][-1]["status"]=="failed"
    assert "deliberate ingestion failure" in manifest["steps"][-1]["details"]["error"]
    progress=json.loads((tmp_path/"pipeline_run_progress.json").read_text())
    assert progress["status"]=="running"
    assert [step["name"] for step in progress["steps"]]==["write_config"]


def test_load_probe_metrics_treats_blank_confidence_intervals_as_none(tmp_path) -> None:
    metrics_path = tmp_path / "linear_probe_metrics.csv"
    metrics_path.write_text(
        "\n".join(
            [
                "task,layer,auroc,auroc_ci_low,auroc_ci_high,auprc,auprc_ci_low,auprc_ci_high,accuracy,accuracy_ci_low,accuracy_ci_high,train_examples,test_examples,train_positive_rate,test_positive_rate",
                "promoter_tata,11,0.9,,0.95,0.8,,,0.7,0.6,,10,4,0.5,0.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    row = _load_probe_metrics(metrics_path, PipelineConfig())[0]

    assert row["auroc_ci_low"] is None
    assert row["auroc_ci_high"] == 0.95
    assert row["auprc_ci_low"] is None
    assert row["auprc_ci_high"] is None
    assert row["accuracy_ci_low"] == 0.6
    assert row["accuracy_ci_high"] is None
