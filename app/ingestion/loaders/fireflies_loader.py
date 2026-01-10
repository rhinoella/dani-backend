from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.config import settings


class FirefliesLoader:
    """
    Low-level Fireflies GraphQL loader.
    Uses Transcript as the primary entity (per Fireflies schema).
    """

    def __init__(self) -> None:
        self.base_url = settings.FIREFLIES_BASE_URL
        self.api_key = settings.FIREFLIES_API_KEY

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        self.timeout = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=10.0)
        
        # Connection pooling to avoid TCP handshake overhead
        self.client = httpx.AsyncClient(
            timeout=self.timeout,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(httpx.HTTPStatusError),
        reraise=True,
    )
    async def _execute(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "query": query,
            "variables": variables or {},
        }

        response = await self.client.post(
            self.base_url,
            headers=self.headers,
            json=payload,
        )

        if response.status_code != 200:
            try:
                error_data = response.json()
                # Rate limit detection
                if response.status_code == 429:
                    raise httpx.HTTPStatusError(
                        f"Rate limit exceeded: {error_data}",
                        request=response.request,
                        response=response,
                    )
                raise RuntimeError(
                    f"Fireflies API error (HTTP {response.status_code}): {error_data}"
                )
            except httpx.HTTPStatusError:
                raise
            except Exception:
                response.raise_for_status()

        data = response.json()

        if "errors" in data:
            raise RuntimeError(f"Fireflies GraphQL error: {data['errors']}")

        return data["data"]

    # -----------------------------
    # Connectivity / sanity check
    # -----------------------------

    async def test_connection(self) -> Dict[str, Any]:
        query = """
        query TestConnection {
          user {
            user_id
            email
          }
        }
        """
        data = await self._execute(query)
        return data["user"]

    # -----------------------------
    # Transcripts (LIST)
    # -----------------------------

    async def list_transcripts(
        self,
        limit: int = 10,
        skip: int = 0,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List transcripts (meetings) metadata with pagination and date filtering.
        
        Args:
            limit: Max number of transcripts to return (max 50)
            skip: Number of transcripts to skip for pagination
            from_date: Start date filter (YYYY-MM-DD format)
            to_date: End date filter (YYYY-MM-DD format)
        """
        query = """
        query ListTranscripts($limit: Int!, $skip: Int!) {
          transcripts(limit: $limit, skip: $skip) {
            id
            title
            date
            duration
            organizer_email
          }
        }
        """
        variables = {"limit": limit, "skip": skip}
        data = await self._execute(query, variables)

        transcripts = data.get("transcripts")
        if transcripts is None:
            raise RuntimeError(f"No transcripts returned: {data}")

        # Client-side date filtering (if Fireflies API doesn't support server-side)
        if from_date or to_date:
            transcripts = self._filter_by_date(transcripts, from_date, to_date)

        return transcripts
    
    def _filter_by_date(
        self, 
        transcripts: List[Dict[str, Any]], 
        from_date: Optional[str], 
        to_date: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Filter transcripts by date range."""
        filtered = []
        for t in transcripts:
            transcript_date = t.get("date")
            if not transcript_date:
                continue
            
            # Extract date portion (format might be datetime)
            date_str = transcript_date.split("T")[0] if "T" in transcript_date else transcript_date
            
            if from_date and date_str < from_date:
                continue
            if to_date and date_str > to_date:
                continue
                
            filtered.append(t)
        
        return filtered

    # -----------------------------
    # Transcript detail (SENTENCES)
    # -----------------------------

    async def get_transcript(
        self,
        transcript_id: str,
    ) -> Dict[str, Any]:
        """
        Fetch a single transcript with sentences.
        """
        query = """
        query GetTranscript($transcriptId: String!) {
          transcript(id: $transcriptId) {
            id
            title
            date
            duration
            organizer_email
            sentences {
              index
              text
              speaker_name
              start_time
              end_time
            }
          }
        }
        """
        variables = {"transcriptId": transcript_id}
        data = await self._execute(query, variables)

        transcript = data.get("transcript")
        if transcript is None:
            raise RuntimeError(f"No transcript found for id={transcript_id}")

        return transcript
