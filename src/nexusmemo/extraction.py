"""LLM-based entity, relation, and decision extraction."""

import json
import logging

from openai import OpenAI

from nexusmemo.config import Settings
from nexusmemo.schemas import ExtractionResult

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You are a knowledge extraction engine. Given a piece of text about a software project, extract structured information.

Return a JSON object with these fields:
{
  "entities": [
    {"name": "EntityName", "type": "tool|concept|person|project|language|framework|service|database|pattern", "description": "brief description"}
  ],
  "relations": [
    {"source": "Entity1", "target": "Entity2", "relation": "USES|REPLACED_BY|DEPENDS_ON|CAUSED_BY|RELATED_TO|WORKED_WITH|BUILT_WITH|MIGRATED_TO|CONFIGURED_WITH|DEPLOYED_ON", "context": "why this relation exists", "confidence": 0.9}
  ],
  "decisions": [
    {"what": "what was decided", "why": "reasoning behind it", "alternatives": ["alt1", "alt2"]}
  ],
  "summary": "one-line summary of the text",
  "importance": 0.7
}

Rules:
- Extract ALL meaningful entities (tools, technologies, people, concepts)
- Capture relationships between entities with context
- Identify any decisions or choices made, with reasoning
- importance: 0.0 (trivial) to 1.0 (critical architectural decision)
- Use consistent entity names (e.g. "PostgreSQL" not "postgres" or "Postgres")
- If no decisions/relations found, return empty arrays
- Return ONLY valid JSON, no markdown formatting"""


class ExtractionService:
    """Extract entities, relations, and decisions from text using LLM."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self._settings.openai_api_key)
        return self._client

    def extract(self, text: str) -> ExtractionResult:
        """Extract structured information from text."""
        try:
            response = self.client.chat.completions.create(
                model=self._settings.extraction_model,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=2000,
            )

            content = response.choices[0].message.content
            if not content:
                logger.warning("Empty response from LLM extraction")
                return ExtractionResult(summary=text[:200])

            data = json.loads(content)
            return ExtractionResult(**data)

        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM response as JSON: %s", e)
            return ExtractionResult(summary=text[:200])
        except Exception as e:
            logger.error("Extraction failed: %s", e)
            return ExtractionResult(summary=text[:200])
