from __future__ import annotations

import json
import re
from datetime import datetime
from enum import Enum
from typing import Any, Iterable

from pydantic import BaseModel


def enum_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, 'value'):
        return getattr(value, 'value')
    return value


def clean_text(value: Any) -> str:
    value = enum_value(value)
    if value is None:
        return ''
    if isinstance(value, BaseModel):
        value = value.model_dump(by_alias=True, exclude_none=True)
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, datetime):
        value = value.isoformat()
    text = str(value).replace('\r', ' ').replace('\n', ' ')
    return re.sub(r'\s+', ' ', text).strip()


def unique_texts(values: Iterable[Any]) -> list[str]:
    unique: list[str] = []
    for value in values:
        cleaned = clean_text(value)
        if cleaned and cleaned not in unique:
            unique.append(cleaned)
    return unique


def slugify(value: Any) -> str:
    text = clean_text(value).lower()
    slug = re.sub(r'[^a-z0-9]+', '_', text).strip('_')
    return slug or 'item'


def amount_to_text(amount: Any) -> str:
    if amount is None:
        return ''
    if hasattr(amount, 'to_string'):
        return clean_text(amount.to_string())
    return clean_text(amount)


def oxford_join(values: Iterable[Any]) -> str:
    cleaned = unique_texts(values)
    if not cleaned:
        return ''
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f'{cleaned[0]} and {cleaned[1]}'
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


def humanize_token(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ''
    if text.isupper():
        return text
    text = re.sub(r'(?<!^)(?=[A-Z])', ' ', text)
    return text.replace('_', ' ').strip().lower()


def shorten_name(value: Any, *, max_words: int = 14, max_len: int = 120) -> str:
    text = clean_text(value)
    if len(text) <= max_len and len(text.split()) <= max_words:
        return text
    words = text.split()
    shortened = ' '.join(words[:max_words])
    if len(shortened) > max_len:
        shortened = shortened[: max_len - 3].rstrip()
    return shortened.rstrip(',;:. ') + '...'


def site_to_text(site: Any) -> str:
    subunit_index = clean_text(getattr(site, 'subunitIndex', None))
    residue_index = clean_text(getattr(site, 'residueIndex', None))
    if subunit_index and residue_index:
        return f'subunit {subunit_index} residue {residue_index}'
    if residue_index:
        return f'residue {residue_index}'
    return subunit_index


def site_list_to_text(sites: Iterable[Any]) -> str:
    return oxford_join(site_to_text(site) for site in sites)
