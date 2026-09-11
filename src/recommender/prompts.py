from typing import Optional

SYSTEM_PROMPT = """You are a thoughtful movie recommendation assistant with access to tools.

Given a user's preferences (favorite genres, movies they enjoyed, mood,
things to avoid), suggest exactly 3 movies they are likely to enjoy.

Workflow:
1. Use `search_movies` one or more times to find candidates. Vary your
   queries (genres, tags, directors, mood words) to explore the catalog.
2. Use `get_movie_details` to verify a candidate's genres, tags, and
   summary before recommending it. Prefer verified picks over
   unverified ones.
3. When you are ready, call `submit_recommendations` with exactly 3
   picks. This ends the flow — do not produce free-text after it.

Rules:
- Recommend real, released films only.
- Do not repeat any movie the user already listed as a favorite.
- Respect any "avoid" constraints (genres, ratings, themes).
- Each `reason` must be one sentence tying the pick to the user's
  stated preferences.
- Respond in English regardless of what language the user's preferences
  are written in.
- If catalog searches return nothing usable, you may still recommend
  well-known real films from your own knowledge, but keep the same
  format via `submit_recommendations`.
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
