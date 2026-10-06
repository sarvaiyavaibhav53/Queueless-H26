from pydantic import BaseModel, ConfigDict

class OfficeResponse(BaseModel):
    id: int
    code: str
    name: str
    counters: int

    model_config = ConfigDict(from_attributes=True)
