"""Optional LLM brain (anthropic), with rule fallback.

Advisory only: may write a one-line terminology/consistency note over a translated
document image. Disabled by default; validates its own output and on any problem the
caller keeps the rule result. Default deployment makes zero paid API calls and is fully
deterministic. **Never rewrites a translation, changes layout, or alters the rendered image.**
"""

from __future__ import annotations

import os
from typing import Optional

from ..config import AgentConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


class LLMBrain:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self._client = None
        self._tried = False

    def available(self) -> bool:
        return bool(self.cfg.llm_fallback_enabled and os.environ.get(self.cfg.llm_api_key_env))

    def _get_client(self):
        if self._tried:
            return self._client
        self._tried = True
        try:
            import anthropic
            key = os.environ.get(self.cfg.llm_api_key_env)
            self._client = anthropic.Anthropic(api_key=key) if key else None
        except Exception as exc:
            logger.info("anthropic client unavailable (%s)", exc)
            self._client = None
        return self._client

    def consistency_note(self, src_lang: str, tgt_lang: str, sample: str) -> Optional[str]:
        """A one-line terminology/consistency note over the translated text. None keeps the rule rationale."""
        if not self.available() or not sample:
            return None
        client = self._get_client()
        if client is None:
            return None
        prompt = (f"You are a localization reviewer. In ONE short sentence, flag any terminology "
                  f"inconsistency or obvious error in this {src_lang}->{tgt_lang} document translation. "
                  f"Do NOT rewrite it.\n\n{sample[:1500]}\n\nReturn ONLY the one-sentence note.")
        try:
            msg = client.messages.create(model=self.cfg.llm_model, max_tokens=120, temperature=0.0,
                                         messages=[{"role": "user", "content": prompt}])
            text = "".join(getattr(b, "text", "") for b in msg.content).strip()
            return text or None
        except Exception as exc:
            logger.info("LLM consistency_note failed (%s)", exc)
            return None


__all__ = ["LLMBrain"]
