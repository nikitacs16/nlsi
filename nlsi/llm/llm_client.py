from dataclasses import dataclass, field

import httpx

import nlsi.llm.limits as limits


@dataclass
class LLMClient:
    engine: str
    _http_client: httpx.AsyncClient = field(init=False)

    def __post_init__(self):
        headers = {"X-ModelType": self.engine}
        self._http_client = httpx.AsyncClient(
            headers=headers,
            http2=False,
            timeout=httpx.Timeout(50.0),
            limits=httpx.Limits(max_connections=500, max_keepalive_connections=500),
        )
        self._make_request = self._limiter(self._make_request)

    async def _make_request(self, args: dict) -> httpx.Response:
        self._update_authorization()
        try:
            response = await self._http_client.post(self.url, json=args)
        except httpx.RequestError as e:
            raise limits.RateLimitExceededError() from e

        if response.status_code != 200:
            print("API status:", response.status_code)
            if response.status_code in (429, 500, 502, 503):
                raise limits.RateLimitExceededError()
            else:
                raise RuntimeError(response.text)

        return response
