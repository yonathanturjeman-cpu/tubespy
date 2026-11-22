"""Nightly publish glue: embed the owner's item states into the published data (so
the preview, which can't reach the API, hides Not relevant for everyone) and copy
suggestions.json into the published folder (so the Suggestions tab isn't empty)."""
import json
from datetime import datetime, timezone
from pathlib import Path


def _public(tmp_path, suggestions=True):
    public = tmp_path / "public"
    (public / "data").mkdir(parents=True)
    for name in ("index.html", "styles.css", "app.js"):
        (public / name).write_text("x", encoding="utf-8")
    (public / "data" / "library.json").write_text(
        json.dumps({"schema_version": 2, "advertisers": []}), encoding="utf-8")
    if suggestions:
        (public / "data" / "suggestions.json").write_text(
            json.dumps({"suggestions": [{"video_id": "v1", "rank": 1}]}), encoding="utf-8")
    return public


def _patch_config(monkeypatch, tmp_path):
    import mini_config
    monkeypatch.setattr(mini_config, "OUR_ROOT", tmp_path)
    monkeypatch.setattr(mini_config, "PUBLISH_DIR", tmp_path / "preview")
    monkeypatch.setattr(mini_config, "PUBLISH_SLUG", "coolina-library-preview")


MERGED = {"schema_version": 2, "source_summary": {},
          "advertisers": [{"advertiser": {"google_advertiser_id": "AR1", "name": "Brand"}, "videos": []}]}


def test_publish_embeds_item_states_and_copies_suggestions(tmp_path, monkeypatch):
    import run_nightly as R
    _public(tmp_path)
    _patch_config(monkeypatch, tmp_path)
    res = R._publish(MERGED, {"GmrgKvRelpk": "not_relevant", "aaaaaaaaaaa": "used"},
                     datetime(2026, 11, 9, tzinfo=timezone.utc))
    stage = Path(res["target"])
    published = json.loads((stage / "data" / "library.json").read_text(encoding="utf-8"))
    # item 4: owner states snapshotted into the published data (hidden/used for everyone)
    assert published["item_states"] == {"GmrgKvRelpk": "not_relevant", "aaaaaaaaaaa": "used"}
    assert res["item_states_embedded"] == 2
    # item 2: suggestions.json copied into the published folder
    assert (stage / "data" / "suggestions.json").exists()
    assert res["suggestions_copied"] is True


def test_publish_without_suggestions_file_does_not_fail(tmp_path, monkeypatch):
    import run_nightly as R
    _public(tmp_path, suggestions=False)
    _patch_config(monkeypatch, tmp_path)
    res = R._publish(MERGED, {}, datetime(2026, 11, 9, tzinfo=timezone.utc))
    stage = Path(res["target"])
    assert not (stage / "data" / "suggestions.json").exists()
    assert res["suggestions_copied"] is False
    assert json.loads((stage / "data" / "library.json").read_text(encoding="utf-8"))["item_states"] == {}
