import os
from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage

from audio_metadata_agent.audio import read_mp3_metadata, write_mp3_metadata
from audio_metadata_agent.schemas.agent_state import AgentState
from audio_metadata_agent.schemas.cleaned_metadata import CleanedMetadata
from audio_metadata_agent.utilities.llm import get_llm


def read_metadata_node(state: AgentState):
    path = state["file_path"]
    filename = os.path.basename(path)

    current_metadata = read_mp3_metadata(path)

    return {
        "current_metadata": current_metadata,
        "messages": [
            f"Processing file: {filename}",
            f"Read existing metadata: {current_metadata}",
        ],
    }


def propose_corrections_node(state: AgentState):
    current = state["current_metadata"]
    file_path = state["file_path"]
    db_match = state.get("db_match")
    filename = os.path.basename(file_path)

    llm = get_llm()
    structured_llm = llm.with_structured_output(CleanedMetadata)

    context_str = (
        f"Similar Track Reference Match: {db_match}"
        if db_match
        else "No similar track reference found."
    )

    prompt = f"""
  You are an expert music archivist and metadata cleaning agent.
  Your goal is to clean and fill out missing or placeholder track metadata fields.

  CRITICAL RULES:
  1. Any tag value equal to "Unknown" or empty MUST be treated as a placeholder and overwritten. Do not leave fields as "Unknown".
  2. Analyze the Filename ({filename}) and the Similar Track Reference Match to intelligently determine the correct Artist, Title, Album, and Genre.
  3. Priority belongs to accurate existing tags, but placeholders like "Unknown" must always be replaced using context clues.
  4. **Album Caution**: Do NOT blindly copy the album name from the Similar Track Reference Match. Tracks often exist as singles, remixes, or appear on multiple projects. Only assign an album if you are confident it belongs to that release; otherwise, keep it clean or treat it as a single and populate the track name.

  File Path: {file_path}
  Current Tags: {current}
  {context_str}
  """

    try:
        result: CleanedMetadata = cast(
            CleanedMetadata,
            structured_llm.invoke(
                [
                    SystemMessage(
                        content="You are a precise metadata cleaner. Output valid structured data only, with no conversational text or preamble."
                    ),
                    HumanMessage(content=prompt),
                ]
            ),
        )

        cleaned = {
            "title": result.title,
            "artist": result.artist,
            "album": result.album,
            "genre": result.genre,
        }
    except Exception as e:
        print(f"LLM correction failed, using fallback: {e}")

        fallback_artist = current.get("artist", "Unknown")
        if fallback_artist == "Unknown" and db_match and db_match.get("artist"):
            fallback_artist = db_match.get("artist")

        cleaned = {
            "title": current.get("title", "Unknown"),
            "artist": fallback_artist,
            "album": current.get(
                "album", db_match.get("album", "Unknown") if db_match else "Unknown"
            ),
            "genre": current.get(
                "genre", db_match.get("genre", "Unknown") if db_match else "Unknown"
            ),
        }

    return {
        "proposed_metadata": cleaned,
        "messages": [f"OpenRouter LLM proposed metadata: {cleaned}"],
    }


def write_metadata_node(state: AgentState):
    path = state["file_path"]
    new_tags = state["proposed_metadata"]

    success = write_mp3_metadata(path, new_tags)
    status = "approved" if success else "error"
    msg = (
        "Successfully wrote metadata to file."
        if success
        else "Failed to write metadata."
    )

    return {"review_status": status, "messages": [msg]}
