from schema.schemas import BaseResponse

def format_response(success: bool, msg: str, data: any = None) -> BaseResponse:
    return BaseResponse(success=success, msg=msg, data=data)
