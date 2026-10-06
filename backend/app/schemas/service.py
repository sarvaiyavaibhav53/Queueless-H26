from pydantic import BaseModel, ConfigDict

class ServiceResponse(BaseModel):
    id: int
    code: str
    name: str
    mean_minutes: float

    model_config = ConfigDict(from_attributes=True)
