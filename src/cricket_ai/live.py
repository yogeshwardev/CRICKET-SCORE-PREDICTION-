"""Provider-neutral HTTP adapter. Provider must expose the validated MatchState contract.

No scraping or invented feed: a licensed provider-specific translator is required
where its response format differs. HTTP retries are bounded and timestamps checked.
"""
import os
from datetime import datetime, timezone
import httpx
from .service import MatchState


class LiveAdapter:
    def __init__(self, url=None, token=None):
        self.url = url or os.getenv("LIVE_PROVIDER_URL")
        self.token = token or os.getenv("LIVE_PROVIDER_TOKEN")
        if not self.url or not self.url.startswith("https://"):
            raise ValueError("Configure an HTTPS LIVE_PROVIDER_URL")

    def fetch(self, match_id: str) -> MatchState:
        with httpx.Client(timeout=10, transport=httpx.HTTPTransport(retries=2), follow_redirects=False) as client:
            response = client.get(self.url, params={"match_id": match_id},
                                  headers={"Authorization": "Bearer "+self.token} if self.token else {})
            response.raise_for_status()
            data = response.json()
        timestamp = datetime.fromisoformat(data["observed_at"].replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc)-timestamp).total_seconds()
        if age < -5 or age > 30:
            raise ValueError("Provider state is stale or has an invalid future timestamp")
        state = MatchState.model_validate(data["state"])
        if state.match_id != match_id:
            raise ValueError("Provider returned a different match")
        return state
