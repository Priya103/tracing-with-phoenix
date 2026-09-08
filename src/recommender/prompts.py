from typing import Optional

SYSTEM_PROMPT = """You are a thoughtful movie recommendation assistant.

Given a user's preferences (favorite genres, movies they enjoyed, mood,
things to avoid), suggest exactly 3 movies they are likely to enjoy.

Rules:
- Recommend real, released films only.
- Do not repeat any movie the user already listed as a favorite.
- Respect any "avoid" constraints (genres, ratings, themes).
- Justify each pick in one sentence tying it back to the user's stated
  preferences.

Return your answer as a numbered list. Format each entry as:
  1. <Title> (<Year>) — <one-sentence reason>
"""


def build_user_prompt(
    favorite_genres: Optional[list[str]] = None,
    liked_movies: Optional[list[str]] = None,
    mood: Optional[str] = None,
    avoid: Optional[list[str]] = None,
) -> str:
    parts: list[str] = []
    if favorite_genres:
        parts.append(f"Favorite genres: {', '.join(favorite_genres)}")
    if liked_movies:
        parts.append(f"Movies I enjoyed: {', '.join(liked_movies)}")
    if mood:
        parts.append(f"Current mood: {mood}")
    if avoid:
        parts.append(f"Please avoid: {', '.join(avoid)}")

    if not parts:
        parts.append("No specific preferences — surprise me with crowd-pleasers.")

    parts.append("Give me 3 recommendations.")
    return "\n".join(parts)
