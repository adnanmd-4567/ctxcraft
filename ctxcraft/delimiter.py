import re
from typing import List, Dict, Optional

class DelimiterError(ValueError):
    pass

class DelimiterManager:

    _VALID_TAG_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

    def wrap(self, tag: str, content: str, sanitize: bool = True) -> str:
        self._validate_tag(tag)

        if sanitize:
            content = self._sanitize(tag, content)

        return f"<{tag}>\n{content}\n</{tag}>"

    def wrap_many(self, sections: Dict[str, str], sanitize: bool = True) -> str:
        return "\n\n".join(
            self.wrap(tag, content, sanitize=sanitize)
            for tag, content in sections.items()
        )

    def _validate_tag(self, tag: str) -> None:
        if not tag or not self._VALID_TAG_PATTERN.match(tag):
            raise DelimiterError(
                f"Invalid tag name: {tag!r}. Tags must be alphanumeric/underscore, "
                f"starting with a letter or underscore."
            )

    def _sanitize(self, tag: str, content: str) -> str:
        pattern = re.compile(r"</?\s*[a-zA-Z_][a-zA-Z0-9_]*\s*>")

        def _escape_match(match: "re.Match") -> str:
            return match.group(0).replace("<", "&lt;").replace(">", "&gt;")

        return pattern.sub(_escape_match, content)