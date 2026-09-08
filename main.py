from recommender import MovieRecommender


def main() -> None:
    recommender = MovieRecommender()
    result = recommender.recommend(
        favorite_genres=["science fiction", "thriller"],
        liked_movies=["Arrival", "Ex Machina"],
        mood="cerebral, slow-burn",
    )

    print("=== Prompt ===")
    print(result.input)
    print()
    print(f"=== Recommendations ({result.model}) ===")
    print(result.output)


if __name__ == "__main__":
    main()
