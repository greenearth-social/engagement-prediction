from __future__ import annotations

import copy
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
import torch

from engagement_prediction.evaluation import production


class _Ranker(torch.nn.Module):
    @torch.jit.export
    def score_candidate_matrix(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        history_time_deltas_hours: torch.Tensor,
        candidate_post_embeddings: torch.Tensor,
        history_author_indices: torch.Tensor,
        candidate_post_author_idx: torch.Tensor,
        history_prior_cumulative_likes: torch.Tensor,
        candidate_prior_cumulative_likes: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros((history_embeddings.shape[0], candidate_post_embeddings.shape[0]))


@pytest.fixture
def deployed(monkeypatch, tmp_path: Path):
    prefix = "gs://models/Experiment/run.task-id"
    manifest_uri = f"{prefix}/artifacts/ranker_serving_manifest/ranker_serving_manifest.json"
    config_uri = f"{prefix}/artifacts/bst_model_config/model_config.json"
    # The deployment's explicit author map is authoritative, even outside its task root.
    author_uri = "gs://deployed-maps/production/ranker_author_idx.parquet"
    weights_uri = f"{prefix}/models/ranker.pt"
    manifest = {
        "ranker_uri": weights_uri,
        "ranker_clearml_model_id": "model-id",
        "clearml_task_id": "task-id",
    }
    config = {
        "model_type": "bst-ranker",
        "embedding_model": "fixture-model",
        "max_history_len": 32,
        "author_pad_idx": 0,
        "author_unk_idx": 1,
        "constructor_args": {
            "post_embedding_dim": 2,
            "author_table_num_rows": 3,
            "num_transformer_layers": 1,
            "use_post_liker_feature": False,
        },
    }
    parquet_path = tmp_path / "authors.parquet"
    pl.DataFrame({
        "author_did": ["did:author"],
        "author_idx": pl.Series([2], dtype=pl.UInt32),
    }).write_parquet(parquet_path)
    weights_path = tmp_path / "weights.pt"
    torch.jit.script(_Ranker()).save(str(weights_path))
    payloads = {
        manifest_uri: json.dumps(manifest).encode(),
        config_uri: json.dumps(config).encode(),
        author_uri: parquet_path.read_bytes(),
        weights_uri: weights_path.read_bytes(),
    }
    service = {
        "metadata": {"generation": 4},
        "status": {
            "observedGeneration": 4,
            "conditions": [{"type": "Ready", "status": "True"}],
            "latestReadyRevisionName": "new-unserved-revision",
            "traffic": [{"revisionName": "serving-revision", "percent": 100}],
        },
    }
    environment = {
        "GE_INFERENCE_MODELS": "user-tower,post-tower,ranker",
        "GE_INFERENCE_RANKER_MANIFEST_URI": manifest_uri,
        "GE_INFERENCE_RANKER_AUTHOR_MAP_URI": author_uri,
        "GE_INFERENCE_RANKER_MAX_HISTORY_LEN": "64",
        "GE_INFERENCE_CONTENT_EMBED_DIM": "2",
        "GE_INFERENCE_API_KEY": "never-write-this-secret",
    }
    commands = []
    downloads = []
    changed_objects = set()

    def subprocess_run(command, *, check, capture_output, text):
        assert check and capture_output and text
        commands.append(command)
        if command[2] == "services":
            document = service
        else:
            assert command[2] == "revisions"
            document = {"spec": {"containers": [{"env": [
                {"name": name, "value": value}
                for name, value in environment.items()
            ]}]}}
        return SimpleNamespace(stdout=json.dumps(document))

    class Blob:
        def __init__(self, uri):
            self.uri = uri
            self.generation = 17
            self.crc32c = "crc32c-value"
            self.md5_hash = "md5-value"

        def reload(self):
            if self.uri not in payloads:
                raise FileNotFoundError(self.uri)

        def download_to_filename(self, filename, *, if_generation_match, checksum):
            assert if_generation_match == 17
            assert checksum == "auto"
            downloads.append(self.uri)
            if self.uri in changed_objects:
                raise RuntimeError("generation precondition failed")
            Path(filename).write_bytes(payloads[self.uri])

    class Client:
        def __init__(self, *, project):
            assert project == "project"

        def bucket(self, name):
            return SimpleNamespace(blob=lambda path: Blob(f"gs://{name}/{path}"))

    monkeypatch.setattr(production.subprocess, "run", subprocess_run)
    monkeypatch.setattr(production.storage, "Client", Client)
    return SimpleNamespace(
        service=service, environment=environment, manifest=manifest, config=config,
        payloads=payloads, manifest_uri=manifest_uri, config_uri=config_uri,
        author_uri=author_uri, weights_uri=weights_uri, commands=commands,
        downloads=downloads, changed_objects=changed_objects,
        output_dir=tmp_path / "snapshot",
    )


def _snapshot(deployed):
    return production.snapshot_production_model(
        project="project", region="region", service="production-service",
        output_dir=deployed.output_dir, logger=logging.getLogger(__name__),
    )


def test_snapshot_uses_serving_revision_and_pins_objects_without_saving_secrets(deployed):
    model, metadata = _snapshot(deployed)

    assert deployed.commands[1][4] == "serving-revision"
    assert metadata["cloud_run"] == {
        "project": "project", "region": "region", "service": "production-service",
        "revision": "serving-revision",
    }
    assert model.name == "production"
    assert model.artifact_format == "deployed_bst"
    assert model.manifest == deployed.manifest
    assert model.training_config == {}
    assert model.to_dict()["stage_run_id"] is None
    assert model.max_history_len == 64
    assert metadata["training_max_history_len"] == 32
    assert metadata["training_window"] is None
    assert model.author_map_stats["author_count"] == 1
    assert metadata["objects"]["author_map"]["uri"] == deployed.author_uri
    assert model.script_sha256["ranker"] == hashlib.sha256(
        deployed.payloads[deployed.weights_uri]
    ).hexdigest()
    for item in metadata["objects"].values():
        assert item["generation"] == "17"
        assert item["crc32c"] == "crc32c-value"
        assert item["sha256"]
    saved_text = (deployed.output_dir / "production_snapshot.json").read_text()
    assert json.loads(saved_text) == metadata
    assert "GE_INFERENCE_API_KEY" not in saved_text
    assert "never-write-this-secret" not in saved_text
    assert not (deployed.output_dir / "manifest.json").exists()


@pytest.mark.parametrize("traffic", [
    [],
    [{"revisionName": "old", "percent": 50}, {"revisionName": "new", "percent": 50}],
    [{"revisionName": "serving-revision", "percent": 90}],
    [{"latestRevision": True, "percent": 100}],
])
def test_snapshot_rejects_ambiguous_or_unresolved_traffic_before_download(deployed, traffic):
    deployed.service["status"]["traffic"] = traffic
    with pytest.raises(ValueError, match="traffic|revision"):
        _snapshot(deployed)
    assert deployed.downloads == []


def test_snapshot_ignores_zero_percent_tagged_revisions(deployed):
    deployed.service["status"]["traffic"].append({
        "revisionName": "preview", "percent": 0, "tag": "preview",
    })
    _, metadata = _snapshot(deployed)
    assert metadata["cloud_run"]["revision"] == "serving-revision"


@pytest.mark.parametrize("state", ["reconciling", "unready", "routes_unready"])
def test_snapshot_rejects_pending_service_updates(deployed, state):
    if state == "reconciling":
        deployed.service["status"]["observedGeneration"] = 3
    elif state == "unready":
        deployed.service["status"]["conditions"][0]["status"] = "Unknown"
    else:
        deployed.service["status"]["conditions"].append({
            "type": "RoutesReady", "status": "False",
        })
    with pytest.raises(ValueError, match="reconciling|not ready"):
        _snapshot(deployed)
    assert deployed.downloads == []


@pytest.mark.parametrize(("name", "value", "error"), [
    ("GE_INFERENCE_MODELS", "user-tower,post-tower", "does not load a ranker"),
    ("GE_INFERENCE_RANKER_MAX_HISTORY_LEN", "0", "positive integer"),
    ("GE_INFERENCE_RANKER_MAX_HISTORY_LEN", "many", "positive integer"),
    ("GE_INFERENCE_CONTENT_EMBED_DIM", "384", "dimensions disagree"),
    ("GE_INFERENCE_RANKER_MANIFEST_URI", "/local/manifest.json", "gs://"),
])
def test_snapshot_rejects_incompatible_deployment_settings_before_weights(
    deployed, name, value, error
):
    deployed.environment[name] = value
    with pytest.raises(ValueError, match=error):
        _snapshot(deployed)
    assert deployed.weights_uri not in deployed.downloads


def test_snapshot_requires_model_settings_to_be_literal_values(deployed):
    deployed.environment["GE_INFERENCE_RANKER_AUTHOR_MAP_URI"] = None
    with pytest.raises(ValueError, match="GE_INFERENCE_RANKER_AUTHOR_MAP_URI"):
        _snapshot(deployed)
    assert deployed.downloads == []


@pytest.mark.parametrize("field", ["use_post_liker_feature", "use_target_user_feature"])
def test_snapshot_rejects_unsupported_features_before_weights(deployed, field):
    config = copy.deepcopy(deployed.config)
    config["constructor_args"][field] = True
    deployed.payloads[deployed.config_uri] = json.dumps(config).encode()
    with pytest.raises(ValueError, match=field):
        _snapshot(deployed)
    assert deployed.weights_uri not in deployed.downloads


def test_snapshot_rejects_unknown_serving_contract_before_config_or_weights(deployed):
    manifest = {**deployed.manifest, "ranker_contract_version": 3}
    deployed.payloads[deployed.manifest_uri] = json.dumps(manifest).encode()
    with pytest.raises(ValueError, match="serving contract"):
        _snapshot(deployed)
    assert deployed.downloads == [deployed.manifest_uri]


def test_snapshot_rejects_invalid_author_mapping_before_weights(deployed, tmp_path):
    invalid_map = tmp_path / "invalid.parquet"
    pl.DataFrame({
        "author_did": ["did:author"],
        "author_idx": pl.Series([5], dtype=pl.UInt32),
    }).write_parquet(invalid_map)
    deployed.payloads[deployed.author_uri] = invalid_map.read_bytes()
    with pytest.raises(ValueError, match="indices must begin at 2"):
        _snapshot(deployed)
    assert deployed.weights_uri not in deployed.downloads


def test_snapshot_rejects_concurrent_object_overwrite_without_completion_marker(deployed):
    deployed.changed_objects.add(deployed.weights_uri)
    with pytest.raises(RuntimeError, match="generation precondition"):
        _snapshot(deployed)
    assert not (deployed.output_dir / "production_snapshot.json").exists()


def test_snapshot_validates_the_downloaded_torchscript_contract(deployed, tmp_path):
    wrong_model_path = tmp_path / "wrong.pt"
    torch.jit.script(torch.nn.Identity()).save(str(wrong_model_path))
    deployed.payloads[deployed.weights_uri] = wrong_model_path.read_bytes()
    with pytest.raises(ValueError, match="score_candidate_matrix"):
        _snapshot(deployed)
    assert not (deployed.output_dir / "production_snapshot.json").exists()
