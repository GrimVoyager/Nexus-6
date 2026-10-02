import re
from typing import Any


class DataSanitizer:
    """
    Detects and anonymizes personally identifiable information (PII).

    The sanitizer uses deterministic regular-expression rules so that it
    does not depend on native NLP extensions. This makes the security layer
    usable in restricted Windows environments.

    Supported entity types:
        PHONE_NUMBER
        EMAIL_ADDRESS
        US_SSN
        IN_PAN
        CREDIT_CARD
        IP_ADDRESS
        PERSON
    """

    ENTITY_PATTERNS = {
        "EMAIL_ADDRESS": re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        ),
        "PHONE_NUMBER": re.compile(
            r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"
        ),
        "IN_PAN": re.compile(
            r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"
        ),
        "US_SSN": re.compile(
            r"\b\d{3}-\d{2}-\d{4}\b"
        ),
        "CREDIT_CARD": re.compile(
            r"(?<!\d)(?:\d{4}[\s-]?){3}\d{4}(?!\d)"
        ),
        "IP_ADDRESS": re.compile(
            r"\b(?:"
            r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\."
            r"){3}"
            r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
        ),
    }

    def __init__(
        self,
        replacement_template: str = "<{entity_type}>",
    ) -> None:
        self.replacement_template = replacement_template

    def sanitize_text(
        self,
        text: Any,
        entities: list[str] | None = None,
    ) -> str:
        """
        Detect and replace supported PII entities in text.

        Parameters
        ----------
        text:
            Input value. Non-string values are converted to strings.

        entities:
            Optional list of entity types to detect. If omitted, all
            supported deterministic entities are detected.

        Returns
        -------
        str
            Sanitized text with detected PII replaced by entity labels.
        """

        if text is None:
            return ""

        text = str(text)

        if not text:
            return text

        selected_entities = (
            list(self.ENTITY_PATTERNS.keys())
            if entities is None
            else entities
        )

        sanitized_text = text

        for entity_type in selected_entities:
            pattern = self.ENTITY_PATTERNS.get(entity_type)

            if pattern is None:
                continue

            replacement = self.replacement_template.format(
                entity_type=entity_type
            )

            sanitized_text = pattern.sub(
                replacement,
                sanitized_text,
            )

        return sanitized_text

    def sanitize_dataframe_headers(
        self,
        dataframe: Any,
    ) -> Any:
        """
        Sanitize dataframe column names.

        Header sanitization is intentionally conservative. Column names
        containing obvious PII indicators are renamed while preserving
        uniqueness.

        The original dataframe is not modified.
        """

        sanitized_dataframe = dataframe.copy()

        renamed_columns: list[str] = []

        for column in sanitized_dataframe.columns:
            original_name = str(column)
            normalized_name = original_name.lower()

            if self._looks_like_pii_header(normalized_name):
                sanitized_name = self._sanitize_header(original_name)
            else:
                sanitized_name = original_name

            renamed_columns.append(sanitized_name)

        sanitized_dataframe.columns = self._make_unique(
            renamed_columns
        )

        return sanitized_dataframe

    @staticmethod
    def _looks_like_pii_header(column_name: str) -> bool:
        """
        Determine whether a column name appears to contain direct PII.

        This is intentionally based on explicit semantic indicators rather
        than attempting broad natural-language interpretation.
        """

        pii_indicators = (
            "email",
            "e-mail",
            "phone",
            "mobile",
            "telephone",
            "ssn",
            "social_security",
            "pan",
            "credit_card",
            "card_number",
            "ip_address",
            "ipaddress",
            "person_name",
            "full_name",
            "first_name",
            "last_name",
        )

        return any(
            indicator in column_name
            for indicator in pii_indicators
        )

    @staticmethod
    def _sanitize_header(column_name: str) -> str:
        """
        Convert a likely PII column name into a neutral schema name.
        """

        normalized = column_name.lower()

        if "email" in normalized or "e-mail" in normalized:
            return "email"

        if (
            "phone" in normalized
            or "mobile" in normalized
            or "telephone" in normalized
        ):
            return "phone_number"

        if "ssn" in normalized or "social_security" in normalized:
            return "ssn"

        if "pan" in normalized:
            return "pan"

        if "credit_card" in normalized or "card_number" in normalized:
            return "card_number"

        if "ip_address" in normalized or "ipaddress" in normalized:
            return "ip_address"

        if (
            "person_name" in normalized
            or "full_name" in normalized
            or "first_name" in normalized
            or "last_name" in normalized
        ):
            return "person_name"

        return "pii"

    @staticmethod
    def _make_unique(columns: list[str]) -> list[str]:
        """
        Ensure renamed dataframe headers remain unique.
        """

        counts: dict[str, int] = {}
        unique_columns: list[str] = []

        for column in columns:
            count = counts.get(column, 0)

            if count == 0:
                unique_columns.append(column)
            else:
                unique_columns.append(f"{column}_{count + 1}")

            counts[column] = count + 1

        return unique_columns