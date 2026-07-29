from typing import Any

from schema.schemas import BaseResponse


def format_response(success: bool, msg: str, data: Any = None) -> BaseResponse:
    return BaseResponse(success=success, msg=msg, data=data)
