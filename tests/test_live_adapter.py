"""Live provider adapter contract: no feed is trusted without validation."""
import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from cricket_ai.live import LiveAdapter


def payload(offset_seconds=0.0, match_id="live", **overrides):
    state = {"match_id": match_id, "date": "2027-04-01", "innings": 1, "over": 1, "score": 0, "wickets": 0,
             "striker": "a", "non_striker": "b", "bowler": "c", "venue": "V", "team_batting": "T",
             "team_bowling": "U", "bowler_confirmed": True, "previous_deliveries": []}
    state.update(overrides)
    observed = datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)
    return {"observed_at": observed.isoformat().replace("+00:00", "Z"), "state": state}


def adapter(body, status=200):
    def handler(request):
        return httpx.Response(status, content=json.dumps(body))
    live = LiveAdapter(url="https://provider.example/state", token="secret")
    original = httpx.Client

    class Patched(original):
        def __init__(self, **kwargs):
            kwargs.pop("transport", None)
            super().__init__(transport=httpx.MockTransport(handler), **kwargs)

    httpx.Client = Patched
    try:
        return live.fetch("live")
    finally:
        httpx.Client = original


def test_insecure_provider_url_is_refused():
    with pytest.raises(ValueError, match="HTTPS"):
        LiveAdapter(url="http://provider.example/state", token="t")
    with pytest.raises(ValueError, match="HTTPS"):
        LiveAdapter(url=None, token="t")


def test_fresh_state_is_accepted():
    state = adapter(payload())
    assert state.match_id == "live" and state.over == 1


def test_stale_and_future_observations_are_refused():
    with pytest.raises(ValueError, match="stale"):
        adapter(payload(offset_seconds=-120))
    with pytest.raises(ValueError, match="stale"):
        adapter(payload(offset_seconds=60))


def test_wrong_match_is_refused():
    with pytest.raises(ValueError, match="different match"):
        adapter(payload(match_id="another"))


def test_provider_errors_and_invalid_schemas_propagate():
    with pytest.raises(httpx.HTTPStatusError):
        adapter(payload(), status=503)
    # An unsupported rain-revised state must never reach the feature builder.
    with pytest.raises(Exception):
        adapter(payload(revised_target=True))
    with pytest.raises(Exception):
        adapter(payload(bowler_confirmed=False))
