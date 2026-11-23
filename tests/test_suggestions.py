"""Suggestions: pre-filter, signal score, product dedup, Haiku pick (stubbed), schedule."""
import json
from datetime import datetime

import pytest

import suggestions as S


def vid(i, **kw):
    v = {"video_id": f"vid{i:06d}ab", "length_seconds": 180, "views": 5_000_000, "age_days": 30,
         "lifetime_views_per_day": 10_000, "creative_count": 5, "product": f"Product {i}",
         "title": f"title {i}", "upload_date": "2026-09-07T00:00:00+00:00", "english": True,
         "embed_url": "e", "thumbnail_url": "th", "youtube_url": "y", "transparency_url": "tr"}
    v.update(kw)
    return v


def ds(videos):
    return {"schema_version": 2, "advertisers": [{"advertiser": {"name": "Brand"}, "videos": videos}]}


def test_prefilter_applies_free_rules():
    videos = [
        vid(1),                                   # keep
        vid(2, length_seconds=60),                # drop: too short
        vid(3, views=500_000),                    # drop: < 1M
        vid(4, age_days=500),                     # drop: older than 12 months
        vid(5),                                   # excluded below
    ]
    kept = S.prefilter(videos, excluded_ids={"vid000005ab"})
    ids = {v["video_id"] for v in kept}
    assert ids == {"vid000001ab"}


def test_signal_score_rewards_vpd_ads_and_recency():
    hi = vid(1, lifetime_views_per_day=20_000, creative_count=15, age_days=10)
    lo = vid(2, lifetime_views_per_day=5_000, creative_count=1, age_days=300)
    assert S.signal_score(hi) > S.signal_score(lo)


def test_top_candidates_one_video_per_product():
    videos = [
        vid(1, product="Knife", lifetime_views_per_day=9000),
        vid(2, product="Knife", lifetime_views_per_day=1000),   # same product, lower -> dropped
        vid(3, product="Vacuum", lifetime_views_per_day=8000),
    ]
    out = S.top_candidates(videos, limit=10)
    products = [v["product"] for v in out]
    assert products == ["Knife", "Vacuum"]                      # one per product, by signal


def test_generate_uses_picker_and_excludes_used_and_history():
    videos = [vid(i) for i in range(1, 6)]
    picker = lambda cands: [{"video_id": c["video_id"], "reason": f"pick {c['video_id']}"} for c in cands]
    rows = S.generate(ds(videos),
                      item_states={"vid000002ab": "used", "vid000003ab": "not_relevant"},
                      history={"vid000004ab"},
                      picker=picker)
    ids = [r["video_id"] for r in rows]
    assert "vid000002ab" not in ids and "vid000003ab" not in ids and "vid000004ab" not in ids
    assert rows[0]["rank"] == 1 and rows[0]["reason"].startswith("pick ")
    assert all({"video_id", "reason", "rank", "views_per_day", "embed_url"} <= set(r) for r in rows)


def test_generate_excludes_our_own_brands():
    dataset = {"schema_version": 2, "advertisers": [
        {"advertiser": {"name": "Coolina"}, "videos": [vid(1)]},       # our own brand
        {"advertiser": {"name": "Super Brand Tools"}, "videos": [vid(2)]},
    ]}
    picker = lambda cands: [{"video_id": c["video_id"], "reason": "r"} for c in cands]
    rows = S.generate(dataset, item_states={}, history=set(), picker=picker)
    channels = {r["channel"] for r in rows}
    assert "Coolina" not in channels
    assert "Super Brand Tools" in channels


def test_generate_excludes_non_english_videos():
    dataset = ds([vid(1, english=True), vid(2, english=False), vid(3, english=None)])
    picker = lambda cands: [{"video_id": c["video_id"], "reason": "r"} for c in cands]
    ids = [r["video_id"] for r in S.generate(dataset, {}, set(), picker=picker)]
    assert ids == ["vid000001ab"]  # only the confirmed-English video is eligible


def test_refresh_is_twice_a_week_never_sunday():
    assert S.is_refresh_day(datetime(2026, 11, 9)) is True    # Monday
    assert S.is_refresh_day(datetime(2026, 11, 12)) is True   # Thursday
    assert S.is_refresh_day(datetime(2026, 11, 10)) is False  # Tuesday
    assert S.is_refresh_day(datetime(2026, 11, 8)) is False   # Sunday


# ---------- robust parsing (the crash Henry hit) ----------

def test_first_json_array_reads_only_the_first_list_amid_two_lists_and_prose():
    # The reply Henry saw: prose, then the real picks array, then a SECOND list and
    # more text. The old greedy "[.*]" regex spanned from the first "[" to the last
    # "]" across BOTH lists -> invalid JSON -> json.loads crashed the whole night.
    reply = (
        'Here are my picks for Coolina:\n'
        '[{"video_id": "vid000001ab", "reason": "great knife ad"}, '
        '{"video_id": "vid000002ab", "reason": "strong DR gadget"}]\n\n'
        'For reference the raw scores were [9.1, 8.4, 7.7] and that is all.'
    )
    picks = S._first_json_array(reply)
    assert picks == [
        {"video_id": "vid000001ab", "reason": "great knife ad"},
        {"video_id": "vid000002ab", "reason": "strong DR gadget"},
    ]


def test_first_json_array_returns_none_when_no_list():
    assert S._first_json_array("no array at all, sorry") is None
    assert S._first_json_array('broken [not, valid json') is None


def test_picks_from_response_prefers_the_structured_tool_call():
    class Block:
        def __init__(self, **kw): self.__dict__.update(kw)
    # a tool_use block carries the picks as structured input -> used directly
    resp = type("R", (), {"content": [
        Block(type="text", text="ignored prose [1,2,3]"),
        Block(type="tool_use", input={"picks": [{"video_id": "x", "reason": "r"}]}),
    ]})
    assert S._picks_from_response(resp) == [{"video_id": "x", "reason": "r"}]


def each_own_adv(videos):
    return {"schema_version": 2,
            "advertisers": [{"advertiser": {"name": f"Adv{i}"}, "videos": [v]}
                            for i, v in enumerate(videos)]}


# ---------- always return 10: relax ladder (never English) ----------

def test_eligible_pool_does_not_relax_when_base_already_has_ten():
    fresh = [vid(i, age_days=30, product=f"P{i}", advertiser_name=f"Adv{i}") for i in range(12)]
    pool, step = S.eligible_pool(fresh, set(), target=10)
    assert step == 0 and len(pool) >= 10          # nothing relaxed


def test_eligible_pool_relaxes_age_12_to_24_months_to_reach_ten():
    # 10 distinct videos, each 500 days old (>12mo, <=24mo), 1M+ views: none pass at
    # the 12-month base, all pass once age relaxes to 24 months (step 1).
    old = [vid(i, age_days=500, product=f"P{i}", advertiser_name=f"Adv{i}") for i in range(10)]
    pool, step = S.eligible_pool(old, set(), target=10)
    assert step == 1 and len(pool) == 10


def test_eligible_pool_relaxes_to_three_per_advertiser_before_views():
    # 4 advertisers x 3 fresh 1M+ videos. Base cap (2/adv) yields 8; only raising the
    # per-advertiser cap to 3 (step 2) reaches 10 -- views never had to drop.
    vids = [vid(a * 10 + p, age_days=30, product=f"P{a}-{p}", advertiser_name=f"Adv{a}")
            for a in range(4) for p in range(3)]
    pool, step = S.eligible_pool(vids, set(), target=10)
    assert step == 2 and len(pool) >= 10


def test_generate_never_relaxes_english_to_backfill():
    # 4 English + 20 non-English, all otherwise eligible, each its own advertiser.
    # The pool must stay English-only -- non-English is never used to reach 10.
    eng = [vid(i, english=True) for i in range(4)]
    non = [vid(100 + i, english=False) for i in range(20)]
    picker = lambda cands: [{"video_id": c["video_id"], "reason": "r"} for c in cands]
    rows = S.generate(each_own_adv(eng + non), {}, set(), picker=picker)
    assert len(rows) == 4
    assert {r["video_id"] for r in rows} == {v["video_id"] for v in eng}


# ---------- de-dupe picks (the 8vHa5bvRoUA duplicate) ----------

def test_generate_dedupes_a_repeated_pick():
    picker = lambda cands: [
        {"video_id": "vid000001ab", "reason": "a"},
        {"video_id": "vid000001ab", "reason": "dup"},   # same video again (what Haiku did)
        {"video_id": "vid000002ab", "reason": "b"},
    ]
    rows = S.generate(ds([vid(1), vid(2)]), {}, set(), picker=picker)
    assert [r["video_id"] for r in rows] == ["vid000001ab", "vid000002ab"]  # no duplicate
    assert [r["rank"] for r in rows] == [1, 2]                              # ranks stay contiguous


def test_generate_tops_up_to_ten_when_ai_returns_fewer():
    # 14 eligible videos (each its own advertiser/product); the AI picker returns only
    # 3. generate must top up to 10 from the remaining ranked candidates -- no repeats,
    # contiguous ranks -- so the user always gets 10 when the data allows.
    vids = [vid(i) for i in range(14)]
    picker = lambda cands: [{"video_id": c["video_id"], "reason": "ai"} for c in cands[:3]]
    rows = S.generate(each_own_adv(vids), {}, set(), picker=picker)
    ids = [r["video_id"] for r in rows]
    assert len(rows) == 10 and len(set(ids)) == 10                 # 10 distinct, topped up
    assert [r["rank"] for r in rows] == list(range(1, 11))         # contiguous ranks
    assert [r["reason"] for r in rows[:3]] == ["ai", "ai", "ai"]   # AI picks keep their reason
    assert all("reach 10" in r["reason"] for r in rows[3:])        # fillers carry the top-up reason


def test_generate_does_not_invent_picks_when_pool_is_small():
    # Only 4 eligible videos and the AI returned 2: top-up can only reach what the pool
    # has (4), never fabricates beyond the eligible candidates.
    vids = [vid(i) for i in range(4)]
    picker = lambda cands: [{"video_id": c["video_id"], "reason": "ai"} for c in cands[:2]]
    rows = S.generate(each_own_adv(vids), {}, set(), picker=picker)
    assert len(rows) == 4 and len({r["video_id"] for r in rows}) == 4


# ---------- durable store-backed history ----------

def test_run_refresh_excludes_store_history_and_records_new_picks(tmp_path):
    from add_brand import Store
    store = Store(tmp_path / "q.sqlite")
    store.add_suggested(["vid000001ab"])            # already suggested before
    lib = tmp_path / "library.json"
    lib.write_text(json.dumps(ds([vid(1), vid(2), vid(3)])), encoding="utf-8")
    out = tmp_path / "suggestions.json"
    picker = lambda cands: [{"video_id": c["video_id"], "reason": "r"} for c in cands]
    S.run_refresh(lib, {}, out, when=datetime(2026, 11, 9), picker=picker,
                  history_get=store.suggested_history, history_add=store.add_suggested)
    ids = [r["video_id"] for r in json.loads(out.read_text(encoding="utf-8"))["suggestions"]]
    assert "vid000001ab" not in ids                 # excluded by durable history
    assert set(ids) == {"vid000002ab", "vid000003ab"}
    # the new picks are recorded, so the next refresh won't repeat them
    assert store.suggested_history() == {"vid000001ab", "vid000002ab", "vid000003ab"}


def test_run_refresh_uses_the_provided_full_dataset_not_the_path(tmp_path):
    # library_path holds only OUR advertisers (here: none). The FULL all-tab dataset
    # (our + Liel's shared raw) is passed in memory and must be what Suggestions picks
    # from -- this is how Liel's brands become suggestable.
    lib = tmp_path / "library.json"
    lib.write_text(json.dumps(ds([])), encoding="utf-8")        # our library: no videos
    out = tmp_path / "suggestions.json"
    full = ds([vid(1), vid(2)])                                  # stands in for our + shared
    picker = lambda cands: [{"video_id": c["video_id"], "reason": "r"} for c in cands]
    seen = set()
    S.run_refresh(lib, {}, out, when=datetime(2026, 11, 9), picker=picker, dataset=full,
                  history_get=lambda: seen, history_add=seen.update)  # isolate from the real build/ file
    ids = [r["video_id"] for r in json.loads(out.read_text(encoding="utf-8"))["suggestions"]]
    assert set(ids) == {"vid000001ab", "vid000002ab"}           # from the dataset, not the empty file


def test_seed_store_history_from_suggestions_json(tmp_path):
    from add_brand import Store
    store = Store(tmp_path / "q.sqlite")
    sj = tmp_path / "suggestions.json"
    sj.write_text(json.dumps({"suggestions": [{"video_id": "old1"}, {"video_id": "old2"}]}), encoding="utf-8")
    assert S.seed_store_history(store, sj) == {"old1", "old2"}
    assert store.suggested_history() == {"old1", "old2"}


def test_run_refresh_keeps_previous_file_when_parsing_fails(tmp_path):
    # Simulate an unparseable reply: the picker raises the parse error that the live
    # path raises. The night must NOT crash and must keep the previous suggestions.json.
    lib = tmp_path / "library.json"
    lib.write_text(json.dumps(ds([vid(1)])), encoding="utf-8")
    out = tmp_path / "suggestions.json"
    previous = '{"generated_at": "yesterday", "suggestions": [{"video_id": "old"}]}\n'
    out.write_text(previous, encoding="utf-8")

    def boom(_cands):
        raise S.SuggestionsParseError("two lists, no clean parse")

    result = S.run_refresh(lib, {}, out, when=datetime(2026, 11, 9), picker=boom)
    assert result == {"refreshed": False, "reason": "parse-failed", "kept_previous": True}
    assert out.read_text(encoding="utf-8") == previous   # previous file untouched
