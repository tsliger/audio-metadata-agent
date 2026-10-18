import argparse
import os

from dotenv import load_dotenv

from audio_metadata_agent.agent import AgentState, graph

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Process a file.")
    parser.add_argument("--file", type=str, required=True, help="Path to the file")

    args = parser.parse_args()

    if args.file is None or args.file == "":
        raise ValueError("--file <file_name> needs to be included")

    input_folder = os.getenv("INPUT_FOLDER_PATH")

    if input_folder is None or input_folder == "":
        raise ValueError("INPUT_FOLDER_PATH environment variable is missing")

    path_template = f"{input_folder}/{args.file}"

    print("Hello from audio-metadata-agent!")
    print(f"Running file path: {path_template}")

    initial_state: AgentState = {
        "file_path": (path_template),
        "embedding": None,
        "duration": None,
        "db_match": None,
        "current_metadata": {},
        "proposed_metadata": {},
        "review_status": "pending",
        "messages": [],
    }

    print("Running metadata agent...")
    result = graph.invoke(initial_state)

    print("\n--- Results ---")
    for msg in result.get("messages", []):
        print(msg)


if __name__ == "__main__":
    main()
