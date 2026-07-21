import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import nomia.server as server_module


@pytest.fixture(autouse=True)
def _isolated_server_state(tmp_path, monkeypatch):
    """Each test gets its own NOMIA_HOME (fresh config/journal/reports) and a reset in-process
    job registry - server.py's journal and job dict are module-level singletons that would
    otherwise leak state across tests."""
    monkeypatch.setenv("NOMIA_HOME", str(tmp_path / "nomia_home"))
    server_module._journal = None
    with server_module._jobs_lock:
        server_module._jobs.clear()
    yield


@pytest.fixture
def client():
    return TestClient(server_module.app)


def _fake_chat_response(category="receipt", confidence=0.9):
    content = f'{{"category": "{category}", "subcategory": null, "description": "costco-receipt", "reason": "looks like a receipt", "confidence": {confidence}}}'
    return SimpleNamespace(message=SimpleNamespace(content=content))


def _wait_for_job(client, path, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(path)
        assert resp.status_code == 200
        data = resp.json()
        if data["status"] != "running":
            return data
        time.sleep(0.02)
    raise AssertionError(f"job at {path} did not finish within {timeout}s")


def test_health(client, mocker):
    mocker.patch("nomia.classify.check_model_available", return_value=True)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_config_round_trip(client):
    cfg = client.get("/api/config").json()
    cfg["destination_root"] = "/tmp/somewhere"
    cfg["preserve_source"] = True

    put_resp = client.put("/api/config", json=cfg)
    assert put_resp.status_code == 200

    fetched = client.get("/api/config").json()
    assert fetched["destination_root"] == "/tmp/somewhere"
    assert fetched["preserve_source"] is True


def test_validate_path(client, tmp_path):
    resp = client.post("/api/validate-path", json={"path": str(tmp_path)})
    assert resp.json() == {"exists": True, "is_dir": True}

    resp2 = client.post("/api/validate-path", json={"path": str(tmp_path / "nope")})
    assert resp2.json() == {"exists": False, "is_dir": False}


def test_naming_preview_uses_real_template_engine(client):
    resp = client.post("/api/naming/preview", json={"template": "{category}_{yyyy}-{mm}-{dd}_{index}"})
    assert resp.status_code == 200
    example = resp.json()["example_filename"]
    assert example.startswith("receipt_")
    assert example.endswith("_01.pdf")


def test_naming_preview_handles_missing_tokens_gracefully(client):
    resp = client.post("/api/naming/preview", json={"template": "{original}__{category}"})
    assert resp.status_code == 200
    assert resp.json()["example_filename"] == "scan001__receipt.pdf"


def test_scan_requires_source_and_destination(client):
    resp = client.post("/api/scan", json={})
    assert resp.status_code == 400


def test_full_scan_preview_apply_undo_cycle(client, tmp_path, mocker):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    Image.new("RGB", (20, 20)).save(source_dir / "photo.jpg")
    dest_dir = tmp_path / "dest"

    mocker.patch("nomia.classify.check_model_available", return_value=True)
    mocker.patch("ollama.Client.chat", return_value=_fake_chat_response())

    scan_resp = client.post("/api/scan", json={"source_folders": [str(source_dir)], "destination_root": str(dest_dir)})
    assert scan_resp.status_code == 200
    run_id = scan_resp.json()["run_id"]

    job = _wait_for_job(client, f"/api/scan/{run_id}/status")
    assert job["status"] == "done"

    preview = client.get(f"/api/preview/{run_id}")
    assert preview.status_code == 200
    items = preview.json()["items"]
    assert len(items) == 1
    assert items[0]["route"] == "auto"
    item_id = items[0]["item_id"]

    # Edit the proposed name via PATCH before applying.
    patch_resp = client.patch(f"/api/preview/{run_id}/items/{item_id}", json={"name_override": "my-custom-name.jpg"})
    assert patch_resp.status_code == 200
    assert patch_resp.json()["proposed_name"] == "my-custom-name.jpg"

    apply_resp = client.post(f"/api/apply/{run_id}", json={"confirm": True})
    assert apply_resp.status_code == 200

    apply_job = _wait_for_job(client, f"/api/apply/{run_id}/status")
    assert apply_job["status"] == "done"
    assert apply_job["result"]["applied"] == 1
    assert apply_job["result"]["verification"]["hash_mismatches"] == []

    applied_files = list(dest_dir.rglob("my-custom-name.jpg"))
    assert len(applied_files) == 1
    assert not (source_dir / "photo.jpg").exists()

    report_resp = client.get(f"/api/report/{run_id}")
    assert report_resp.status_code == 200
    assert report_resp.headers["content-type"].startswith("application/vnd.openxmlformats")

    last_applied = client.get("/api/runs/last-applied")
    assert last_applied.json()["run_id"] == run_id

    undo_resp = client.post(f"/api/undo/{run_id}")
    assert undo_resp.status_code == 200
    assert undo_resp.json()["undone"] == 1
    assert (source_dir / "photo.jpg").exists()
    assert not applied_files[0].exists()


def test_apply_requires_confirm_true(client, tmp_path, mocker):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    Image.new("RGB", (10, 10)).save(source_dir / "photo.jpg")

    mocker.patch("nomia.classify.check_model_available", return_value=True)
    mocker.patch("ollama.Client.chat", return_value=_fake_chat_response())

    scan_resp = client.post("/api/scan", json={"source_folders": [str(source_dir)], "destination_root": str(tmp_path / "dest")})
    run_id = scan_resp.json()["run_id"]
    _wait_for_job(client, f"/api/scan/{run_id}/status")

    resp = client.post(f"/api/apply/{run_id}", json={"confirm": False})
    assert resp.status_code == 400


def test_bulk_patch_updates_multiple_items(client, tmp_path, mocker):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    Image.new("RGB", (10, 10)).save(source_dir / "a.jpg")
    Image.new("RGB", (10, 10)).save(source_dir / "b.jpg")

    mocker.patch("nomia.classify.check_model_available", return_value=True)
    mocker.patch("ollama.Client.chat", return_value=_fake_chat_response(confidence=0.6))  # -> review route

    scan_resp = client.post("/api/scan", json={"source_folders": [str(source_dir)], "destination_root": str(tmp_path / "dest")})
    run_id = scan_resp.json()["run_id"]
    _wait_for_job(client, f"/api/scan/{run_id}/status")

    items = client.get(f"/api/preview/{run_id}").json()["items"]
    item_ids = [item["item_id"] for item in items]

    bulk_resp = client.patch(f"/api/preview/{run_id}/items:bulk", json={"item_ids": item_ids, "user_decision": "confirmed"})
    assert bulk_resp.json()["updated"] == 2

    updated_items = client.get(f"/api/preview/{run_id}").json()["items"]
    assert all(item["user_decision"] == "confirmed" for item in updated_items)


def test_preview_and_report_404_for_unknown_run(client):
    assert client.get("/api/preview/doesnotexist").status_code == 404
    assert client.get("/api/report/doesnotexist").status_code == 404
    assert client.post("/api/undo/doesnotexist").status_code == 404
