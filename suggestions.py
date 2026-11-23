#!/usr/bin/env python3
"""Weekly "Suggestions": 10 AI-picked swipes for Coolina.

Pipeline (cheap first, one paid call at the end):
  1. Free pre-filter: English (the library is already English-only), 2-5 min, 1M+
     views, uploaded in the last 12 months, and NOT Used / Not relevant / already
     suggested.
  2. Signal score (views/day, number of ads using the video, recency); take the
     top 30, at most one video per product.
  3. ONE batched Claude Haiku call: send the 30 with their stats + a fit guide and
     have it pick & rank the best 10, one-line reason each.

History of everything ever suggested is kept so nothing repeats. Refreshed twice a
week (never Sunday) by the caller; generate() is pure given its inputs (the Claude
picker is injectable so tests stay offline).
"""
from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import own_brands

log = logging.getLogger("suggestions")

ROOT = Path(__file__).resolve().parent
HISTORY_PATH = ROOT / "build" / "suggested_history.json"
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-5-5")


class SuggestionsParseError(Exception):
    """The Claude reply could not be parsed into a picks array. Raised so the caller
    can skip Suggestions for the night and keep the previous suggestions.json, instead
    of crashing the whole run."""

MIN_SECONDS, MAX_SECONDS, MIN_VIEWS, MAX_AGE_DAYS = 120, 300, 1_000_000, 365
CANDIDATE_POOL, PICK = 30, 10

FIT_GUIDE = (
    "Coolina sells knives and kitchen/outdoor gear to a mostly-male audience. "
    "GOOD fits: tools, gadgets, cars / scratch repair, work gear, knives / cooking, "
    "backyard / outdoor gear — things men like. BAD fits: apps, hospitals, "
    "supplements, giant brands, and ads that open on a pain Coolina doesn't have. "
    "STRONG signals: hardcore direct-response (not branding), an advertiser with a "
    "gadget niche for this audience, an advertiser that also sells knives, and ads "
    "that look like they ran on TV."
)


# ---------- free pre-filter + signal score ----------

def flatten(dataset: dict) -> list[dict]:
    advertisers = dataset.get("advertisers", []) if dataset.get("schema_version") == 2 else [dataset]
    out = []
    for a in advertisers:
        name = a.get("advertiser", {}).get("name")
        for v in a.get("videos", []):
            out.append({**v, "advertiser_name": name})
    return out


def signal_score(video: dict) -> float:
    vpd = video.get("lifetime_views_per_day") or 0
    ads = video.get("creative_count") or 1
    age = video.get("age_days") or MAX_AGE_DAYS
    recency = max(0.3, 1 - age / MAX_AGE_DAYS)        # newer scores higher, floor 0.3
    ad_boost = 1 + 0.15 * min(ads, 20)                # more ads = stronger DR signal
    return vpd * ad_boost * recency


def prefilter(videos: list[dict], excluded_ids: set[str],
              max_age_days: int = MAX_AGE_DAYS, min_views: int = MIN_VIEWS) -> list[dict]:
    kept = []
    for v in videos:
        if v["video_id"] in excluded_ids:
            continue
        if not (MIN_SECONDS <= (v.get("length_seconds") or 0) <= MAX_SECONDS):
            continue
        if (v.get("views") or 0) < min_views:
            continue
        if (v.get("age_days") or 10**9) > max_age_days:
            continue
        kept.append(v)
    return kept


MAX_PER_ADVERTISER = 2  # at most 2 picks from any one advertiser


def top_candidates(videos: list[dict], limit: int = CANDIDATE_POOL,
                   max_per_advertiser: int = MAX_PER_ADVERTISER) -> list[dict]:
    """Highest signal score, at most one video per product and at most
    `max_per_advertiser` videos per advertiser."""
    ranked = sorted(videos, key=signal_score, reverse=True)
    seen_products, per_advertiser, out = set(), {}, []
    for v in ranked:
        key = (v.get("product") or v["video_id"]).strip().lower()
        if key in seen_products:
            continue
        adv = (v.get("advertiser_name") or v.get("author") or "").strip().lower()
        if adv and per_advertiser.get(adv, 0) >= max_per_advertiser:
            continue
        seen_products.add(key)
        per_advertiser[adv] = per_advertiser.get(adv, 0) + 1
        out.append(v)
        if len(out) >= limit:
            break
    return out


# Relax ONLY in this order, and only as far as needed to reach 10 candidates. Each
# step is strictly more permissive than the last, so the eligible pool only grows.
# English-only (applied upstream in generate) is NEVER relaxed.
RELAX_LADDER = [
    {"max_age_days": 365, "max_per_advertiser": 2, "min_views": 1_000_000},  # base
    {"max_age_days": 730, "max_per_advertiser": 2, "min_views": 1_000_000},  # age 12->24 months
    {"max_age_days": 730, "max_per_advertiser": 3, "min_views": 1_000_000},  # then max 3 / advertiser
    {"max_age_days": 730, "max_per_advertiser": 3, "min_views":   500_000},  # then 500K+ views
]


def eligible_pool(videos: list[dict], excluded_ids: set[str], target: int = PICK) -> tuple[list[dict], int]:
    """The candidate pool, relaxing the filters in RELAX_LADDER order only as far as
    needed to reach `target` candidates (so we always offer 10 when the data allows).
    Returns (pool, step_index); step 0 means nothing had to be relaxed."""
    pool: list[dict] = []
    for step, knobs in enumerate(RELAX_LADDER):
        pre = prefilter(videos, excluded_ids, max_age_days=knobs["max_age_days"], min_views=knobs["min_views"])
        pool = top_candidates(pre, max_per_advertiser=knobs["max_per_advertiser"])
        if len(pool) >= target:
            return pool, step
    return pool, len(RELAX_LADDER) - 1   # exhausted: best effort with the most relaxed step


# ---------- the one batched Claude call ----------

def _api_key() -> str | None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    for path in (ROOT / ".env", Path(r"C:\Users\user\youtube-stats\.env"), Path.home() / "youtube-stats" / ".env"):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("ANTHROPIC_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'") or None
        except OSError:
            continue
    return None


def _candidate_line(i: int, v: dict) -> str:
    return (f"{i}. id={v['video_id']} | product={v.get('product') or '?'} | "
            f"channel={v.get('advertiser_name') or v.get('author') or '?'} | "
            f"views={v.get('views')} | views/day={round(v.get('lifetime_views_per_day') or 0)} | "
            f"ads_using={v.get('creative_count')} | age_days={v.get('age_days')} | "
            f"title={(v.get('title') or '')[:70]}")


# Structured output: force the model through this tool so it CANNOT return prose or a
# second list. The schema is the contract; the picks arrive as the tool-call `input`.
PICK_TOOL = {
    "name": "submit_picks",
    "description": "Submit the ranked competitor-ad swipes picked for Coolina, best first.",
    "input_schema": {
        "type": "object",
        "properties": {
            "picks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string", "description": "id of a candidate video"},
                        "reason": {"type": "string", "description": "one short line"},
                    },
                    "required": ["video_id", "reason"],
                },
            },
        },
        "required": ["picks"],
    },
}


def _first_json_array(text: str) -> list | None:
    """The FIRST valid JSON array in `text`, or None. Reads one complete value with
    json.JSONDecoder.raw_decode from each '[' in turn (so extra lists or prose after
    it are ignored), instead of a greedy '[.*]' regex that would span from the first
    '[' to the LAST ']' across two arrays and fail to parse."""
    decoder = json.JSONDecoder()
    start = text.find("[")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(text, start)
        except ValueError:
            start = text.find("[", start + 1)
            continue
        if isinstance(value, list):
            return value
        start = text.find("[", start + 1)
    return None


def _picks_from_response(response) -> list | None:
    """Pull the picks list out of a Claude reply: the forced tool call's `input` first
    (structured output), else the first valid JSON array in any text block (fallback for
    a model/proxy that ignored tool_choice). None when neither yields a list."""
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and isinstance(block.input, dict):
            picks = block.input.get("picks")
            if isinstance(picks, list):
                return picks
    text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), "")
    return _first_json_array(text)


def choose_with_claude(candidates: list[dict], picker=None, count: int = PICK) -> list[dict]:
    """Return [{video_id, reason}] (best first). picker(candidates)->list lets tests
    stay offline; default calls Claude Haiku once with all candidates, forcing a
    structured tool call. Raises SuggestionsParseError if the reply can't be parsed."""
    if picker is not None:
        return picker(candidates)[:count]
    key = _api_key()
    if not key:
        # no key: fall back to the signal-score order with a neutral reason
        return [{"video_id": v["video_id"], "reason": "Top signal score (AI pick unavailable)."}
                for v in candidates[:count]]
    prompt = (
        "You curate a competitor swipe library for Coolina. " + FIT_GUIDE +
        f"\n\nFrom the {len(candidates)} competitor YouTube video ads below, pick and rank the "
        f"best {count} swipes for Coolina (best first). Prefer the GOOD/STRONG fits; drop BAD fits "
        "even if their numbers are high. Call submit_picks with the best "
        f"{count} (video_id + one short reason each).\n\n"
        + "\n".join(_candidate_line(i + 1, v) for i, v in enumerate(candidates))
    )
    import anthropic
    client = anthropic.Anthropic(api_key=key)
    params = {"model": MODEL, "max_tokens": 1024,
              "messages": [{"role": "user", "content": prompt}],
              "tools": [PICK_TOOL],
              "tool_choice": {"type": "tool", "name": "submit_picks"}}
    if MODEL.startswith("claude-haiku"):
        params["thinking"] = {"type": "disabled"}
    picks = _picks_from_response(client.messages.create(**params))
    if picks is None:
        raise SuggestionsParseError("Claude reply had no parseable picks array")
    valid = {v["video_id"] for v in candidates}
    return [{"video_id": p["video_id"], "reason": str(p.get("reason", "")).strip()}
            for p in picks if isinstance(p, dict) and p.get("video_id") in valid][:count]


# ---------- history ----------

def load_history() -> set[str]:
    try:
        return set(json.loads(HISTORY_PATH.read_text(encoding="utf-8")))
    except Exception:
        return set()


def save_history(ids: set[str]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(sorted(ids)), encoding="utf-8")


def ids_ever_suggested(suggestions_path, *extra_json_paths) -> set[str]:
    """Every video id we can find that was ever suggested: the current suggestions.json
    plus any legacy history JSON files. Used to seed the durable store on an instance
    that started with an empty build/ (so old picks are never suggested again)."""
    ids: set[str] = set()
    for p in (suggestions_path, *extra_json_paths):
        try:
            data = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and isinstance(data.get("suggestions"), list):
            ids |= {r.get("video_id") for r in data["suggestions"] if isinstance(r, dict) and r.get("video_id")}
        elif isinstance(data, list):
            ids |= {x for x in data if isinstance(x, str)}
    return ids


def seed_store_history(store, suggestions_path, *extra_json_paths) -> set[str]:
    """One-time, idempotent seed of the store's durable no-repeat history from every
    video ever suggested. Add-only, so it is safe to call on every service startup."""
    ids = ids_ever_suggested(suggestions_path, *extra_json_paths)
    if ids:
        store.add_suggested(ids)
    return ids


# ---------- orchestration ----------

REFRESH_WEEKDAYS = {0, 3}  # Monday & Thursday — twice a week, never Sunday (weekday 6)


def is_refresh_day(when: datetime) -> bool:
    return when.weekday() in REFRESH_WEEKDAYS


def run_refresh(library_path, item_states, out_path, when=None, picker=None,
                *, history_get=None, history_add=None, dataset=None) -> dict:
    """Twice-weekly refresh: REPLACE suggestions.json with 10 NEW picks (never
    suggested before) and record them in the no-repeat history. No-op on non-refresh
    days (and never Sunday).

    history_get() -> ids ever suggested; history_add(ids) records the new picks. They
    default to the local build/ JSON file; the live instance passes store-backed ones
    (store.suggested_history / store.add_suggested) so history survives redeploys.

    dataset overrides reading library_path -- the nightly passes the FULL all-tab
    library (our advertisers + Liel's shared raw) so Suggestions can pick from his
    brands too, not only ours."""
    history_get = history_get or load_history
    history_add = history_add or (lambda ids: save_history(load_history() | set(ids)))
    when = when or datetime.now(timezone.utc)
    if not is_refresh_day(when):
        return {"refreshed": False, "reason": "not a refresh day"}
    if dataset is None:
        dataset = json.loads(Path(library_path).read_text(encoding="utf-8"))
    history = set(history_get())
    try:
        rows = generate(dataset, item_states, history, picker)
    except SuggestionsParseError as exc:
        # Don't crash the night over Suggestions: keep the previous suggestions.json
        # (leave out_path untouched) and let the rest of the run publish.
        log.warning("suggestions: skipping refresh, keeping previous file (%s)", exc)
        return {"refreshed": False, "reason": "parse-failed", "kept_previous": True}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(
        json.dumps({"generated_at": when.isoformat(), "suggestions": rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    history_add({r["video_id"] for r in rows})
    return {"refreshed": True, "count": len(rows)}


def generate(dataset: dict, item_states: dict, history: set[str], picker=None) -> list[dict]:
    """Return up to 10 suggestion rows (self-contained for the UI). Does not mutate
    history — the caller persists it after writing the file."""
    excluded = set(history) | {vid for vid, st in (item_states or {}).items() if st in ("used", "not_relevant")}
    # English-only, and never our own brands (Coolina et al.). Videos carry an
    # `english` flag now that the library also holds non-English ads (the All tab).
    flat = [v for v in flatten(dataset)
            if v.get("english") is True
            and not own_brands.is_own(v.get("advertiser_name"), v.get("landing_domain"))]
    videos = {v["video_id"]: v for v in flat}
    pool, _ = eligible_pool(list(videos.values()), excluded)
    picks = choose_with_claude(pool, picker)
    rows, seen = [], set()
    for pick in picks:
        vid = pick["video_id"]
        v = videos.get(vid)
        if not v or vid in seen:   # drop unknowns and de-dupe (never show a video twice)
            continue
        seen.add(vid)
        rows.append(_suggestion_row(len(rows) + 1, pick["reason"], v))
    # The AI sometimes returns fewer than PICK (it dropped bad fits). Top up from the
    # remaining ranked candidates until we have 10 -- the pool is signal-ranked and
    # already English-only, within the per-advertiser/product caps, and history-
    # excluded, so the top-ups obey every rule the picks did.
    if len(rows) < PICK:
        for v in pool:
            if len(rows) >= PICK:
                break
            if v["video_id"] in seen:
                continue
            seen.add(v["video_id"])
            rows.append(_suggestion_row(len(rows) + 1, "Top signal score (added to reach 10).", v))
    return rows


def _suggestion_row(rank: int, reason: str, v: dict) -> dict:
    return {
        "rank": rank, "reason": reason, "video_id": v["video_id"],
        "product": v.get("product"), "title": v.get("title"),
        "channel": v.get("advertiser_name") or v.get("author"),
        "views": v.get("views"), "views_per_day": round(v.get("lifetime_views_per_day") or 0),
        "length_seconds": v.get("length_seconds"), "upload_date": v.get("upload_date"),
        "embed_url": v.get("embed_url"), "thumbnail_url": v.get("thumbnail_url"),
        "youtube_url": v.get("youtube_url"), "transparency_url": v.get("transparency_url"),
    }
