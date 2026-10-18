import acoustid
from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3, ID3NoHeaderError


def generate_fingerprint(file_path: str) -> tuple[bytes | None, float | None]:
    """Generates an acoustic fingerprint and duration using pyacoustid/fpcalc."""
    try:
        duration, fingerprint = acoustid.fingerprint_file(file_path)
        return fingerprint, duration
    except Exception as e:
        print(f"Error generating fingerprint for {file_path}: {e}")
        return None, None


def read_mp3_metadata(file_path: str) -> dict:
    """Reads existing ID3 tags from an MP3 file."""
    try:
        audio = EasyID3(file_path)
        return {key: audio[key][0] for key in audio}
    except ID3NoHeaderError:
        return {}
    except Exception as e:
        print(f"Error reading metadata from {file_path}: {e}")
        return {}


def write_mp3_metadata(file_path: str, new_tags: dict) -> bool:
    """Writes cleaned ID3 tags back to the MP3 file, creating headers if missing."""
    try:
        try:
            audio = EasyID3(file_path)
        except ID3NoHeaderError:
            audio_id3 = ID3()
            audio_id3.save(file_path)
            audio = EasyID3(file_path)

        for key, value in new_tags.items():
            audio[key] = value
        audio.save()
        return True
    except Exception as e:
        print(f"Error writing metadata to {file_path}: {e}")
        return False
