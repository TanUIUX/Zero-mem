from __future__ import annotations

import re

from .lexical import tokenize
from .ner import EntityExtractor
from .types import QueryProfile


_TEMPORAL = ["before", "after", "then", "next", "previous", "last", "latest", "first", "earlier", "later", "hôm qua", "hôm nay", "trước", "sau", "tiếp theo", "lần đầu", "gần đây"]
_RELATIONAL = ["relationship", "related", "colleague", "manager", "works with", "friend", "who did", "which person", "quan hệ", "đồng nghiệp", "quản lý", "làm cùng", "ai đã"]
_DATE_QUERY_RE = re.compile(r"\bwhen\b|\bdate\b|khi nào|ngày nào", re.IGNORECASE)


class QueryRouter:
    def __init__(self, entity_extractor: EntityExtractor):
        self.entity_extractor = entity_extractor

    def profile(self, query: str, session_id: str | None = None) -> QueryProfile:
        ql = query.lower()
        entities = self.entity_extractor.extract(query)
        temporal = [x for x in _TEMPORAL if x in ql]
        date_signal = _DATE_QUERY_RE.search(ql)
        if date_signal:
            temporal.append(date_signal.group(0).lower())
            answer_type = "date"
        elif re.search(r"\bwho\b|ai\b", ql):
            answer_type = "person"
        elif re.search(r"\bwhere\b|ở đâu|thành phố|địa điểm", ql):
            answer_type = "location"
        elif re.search(r"\bhow many\b|bao nhiêu|\bnumber\b", ql):
            answer_type = "number"
        elif re.search(r"\blist\b|liệt kê", ql):
            answer_type = "list"
        else:
            answer_type = "text"
        relation_signal = any(x in ql for x in _RELATIONAL) or len(entities) >= 2
        route = "relational" if relation_signal and not temporal else "local"
        stop = {"the", "a", "an", "is", "was", "what", "who", "where", "when", "did", "do", "to", "of", "and", "or", "tôi", "là", "gì", "ở", "đâu", "khi", "nào", "đã"}
        keywords = [t for t in tokenize(query) if t not in stop and len(t) > 1]
        return QueryProfile(
            query=query,
            subjects=entities,
            keywords=keywords,
            answer_type=answer_type,
            temporal_cues=temporal,
            session_id=session_id,
            route=route,
        )
