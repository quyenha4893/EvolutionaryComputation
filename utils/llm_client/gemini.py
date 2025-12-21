import logging
from typing import Optional
from .base import BaseClient

import requests
import json
from google.oauth2 import service_account
import google.auth.transport.requests

logger = logging.getLogger(__name__)

class GeminiClient(BaseClient):
    def __init__(self, model: str, temperature: float = 1.0, credentials_json_path: Optional[str] = None, location: Optional[str] = None):
        super().__init__(model, temperature)
        if credentials_json_path is None:
            raise ValueError("Please provide Vertex AI service account JSON path via config (credentials_json_path)")
        self.location = location or "us-central1"
        self.model = model
        self.credentials_json_path = credentials_json_path
        self.project_id = self._get_project_id_from_credentials(credentials_json_path)
        self.api_endpoint = f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.location}/publishers/google/models/{self.model}:generateContent"

    def _get_project_id_from_credentials(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data['project_id']

    def _get_access_token(self, credentials_path):
        credentials = service_account.Credentials.from_service_account_file(credentials_path, scopes=["https://www.googleapis.com/auth/cloud-platform"])
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        return credentials.token

    def _chat_completion_api(self, messages: list[dict], temperature: float, n: int = 1):
        prompt = "\n".join([msg["content"] for msg in messages if msg["role"] == "user"])
        token = self._get_access_token(self.credentials_json_path)  # always refresh token
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {"temperature": temperature}
        }
        response = requests.post(self.api_endpoint, headers=headers, json=body)
        if response.status_code != 200:
            logger.error(f"Vertex AI Gemini API error: {response.text}")
            raise RuntimeError(f"Vertex AI Gemini API error: {response.text}")
        result = response.json()
        content = result["candidates"][0]["content"]["parts"][0]["text"]
        class DummyChoice:
            def __init__(self, content):
                self.message = type("msg", (), {"content": content})
        return [DummyChoice(content)]
