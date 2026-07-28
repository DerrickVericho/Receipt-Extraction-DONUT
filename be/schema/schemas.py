from pydantic import BaseModel
from typing import Any, Optional, Dict, List


class BaseResponse(BaseModel):
    success: bool
    msg: str
    data: Optional[Any] = None


class ExtractionData(BaseModel):
    company: str
    date: str
    total: str
    items: List[Dict[str, str]]
