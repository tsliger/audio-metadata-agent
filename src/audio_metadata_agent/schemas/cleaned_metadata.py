from langgraph.pregel.main import BaseModel
from pydantic.fields import Field


class CleanedMetadata(BaseModel):
    title: str = Field(description="The clean, properly capitalized song title")
    artist: str = Field(description="The clean artist or band name")
    album: str = Field(description="The clean album name, if known")
    genre: str = Field(
        description="The musical genre, if identifiable", default="Unknown"
    )
