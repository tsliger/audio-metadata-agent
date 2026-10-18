import os
from typing import Any

import chromadb
import laion_clap
import librosa

from audio_metadata_agent.schemas.agent_state import AgentState

# move at some point and configs
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma_client.get_or_create_collection(
    name="audio_embeddings", metadata={"hnsw:space": "cosine"}
)

clap_model = laion_clap.CLAP_Module(
    enable_fusion=False,
    amodel='HTSAT-base'
)
ckpt_path = os.path.expanduser("~/.cache/laion_clap/music_audioset_epoch_15_esc_90.14.pt")
clap_model.load_ckpt(ckpt=ckpt_path)


def generate_embedding_node(state: AgentState):
    path = state["file_path"]
    filename = os.path.basename(path)

    try:
        audio_data, sr = librosa.load(path, sr=48000)
        duration = float(len(audio_data) / sr)

        audio_tensor = audio_data.reshape(1, -1)

        embedding_array = clap_model.get_audio_embedding_from_data(
            x=audio_tensor
        )

        embedding = embedding_array[0].tolist()

    except Exception as e:
        print(f"Error generating LAION-CLAP embedding for {filename}: {e}")
        embedding = [0.0] * 512
        duration = state.get("duration", 0.0)

    return {
        "embedding": embedding,
        "duration": duration,
        "messages": [f"Generated acoustic audio embedding for {filename} (Dimensions: {len(embedding)})"],
    }

def vector_db_lookup_node(state: AgentState) -> dict[str, Any]:
    embedding = state.get("embedding")
    if not embedding:
        return {"messages": ["Skipping vector DB lookup: No embedding available."]}

    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["metadatas", "distances"],
        )

        distances = results.get("distances") if results else None
        metadatas = results.get("metadatas") if results else None

        if distances and metadatas and len(distances[0]) > 0:
            best_distance = distances[0][0]
            matched_meta = metadatas[0][0]

            confidence = max(0.0, min(100.0, round((1.0 - best_distance) * 100, 2)))
            artist = str(matched_meta.get("artist", ""))

            if confidence > 70.0 and artist and "unknown" not in artist.lower():
                return {
                    "db_match": matched_meta,
                    "messages": [
                        f"Found probabilistic match in ChromaDB ({confidence}% similarity): {matched_meta}"
                    ],
                }
    except Exception as e:
        print(f"ChromaDB lookup error: {e}")

    return {
        "db_match": None,
        "messages": [
            "No sufficiently similar track found in ChromaDB via vector search."
        ],
    }


def save_to_chroma_node(state: AgentState) -> dict[str, Any]:
    embedding = state.get("embedding")
    metadata = state.get("proposed_metadata")

    if not embedding or not metadata:
        return {"messages": ["Skipping ChromaDB save: Missing embedding or metadata."]}

    artist = str(metadata.get("artist", ""))
    title = str(metadata.get("title", ""))
    if (
        not artist
        or "unknown" in artist.lower()
        or not title
        or "unknown" in title.lower()
    ):
        return {
            "messages": [
                "Skipping ChromaDB save: Metadata contains placeholder/unknown values."
            ]
        }

    track_id = f"{artist}-{title}".lower().replace(" ", "_")

    try:
        collection.upsert(
            ids=[track_id],
            embeddings=[embedding],
            metadatas=[metadata],
        )
        return {
            "messages": ["Successfully saved audio embedding and metadata to ChromaDB."]
        }
    except Exception as e:
        print(f"Error saving to ChromaDB: {e}")
        return {"messages": [f"Failed to save to ChromaDB: {e}"]}


def check_chroma_match(state: AgentState):
    proposed = state.get("proposed_metadata")
    if proposed and proposed.get("title") and proposed.get("artist"):
        return "skip_llm"
    return "run_llm"
