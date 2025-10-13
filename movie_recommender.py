#!/usr/bin/env python3
"""
movie_recommender.py

A simple CLI-based movie recommender and analytics tool.

File formats:
- Movies file (pipe-delimited):  genre|movie_id|movie_name
- Ratings file (pipe-delimited): movie_name|rating|user_id
  * rating is in [0, 5] (integer or float)
  * A user rates a given movie at most once

Features implemented (with menu options):
1) Load input data files (movies and ratings)
2) Top N movies (ranked on average ratings)
3) Top N movies in a genre (ranked on average ratings)
4) Top N genres (ranked on average of average ratings of movies in genre)
5) User's most preferred genre (based on that user's average ratings per genre)
6) Recommend movies for a user: 3 most popular movies from the user's top genre that the user has not yet rated

Python 3.12 compatible.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import sys
import statistics


def _strip_inline_comment(line: str) -> str:
    """Return the part of the line before a '#' comment (inline or full-line)."""
    if '#' in line:
        return line.split('#', 1)[0]
    return line

# -------------------------------
# Data structures
# -------------------------------

@dataclass(frozen=True)
class Movie:
    """Represents a movie loaded from the movies file.
    
    Attributes:
        genre: The single genre to which this movie belongs.
        movie_id: The ID string (not used in calculations but preserved).
        name: The movie's name including year (unique per title-year pair).
    """
    genre: str
    movie_id: str
    name: str


class DataStore:
    """Holds all loaded data and provides lookup dictionaries.
    
    Attributes:
        movies_by_name: Maps movie_name -> Movie
        movies_by_genre: Maps genre -> set of movie_names
        ratings_by_movie: Maps movie_name -> list of float ratings
        user_ratings: Maps user_id -> {movie_name: rating}
    """
    def __init__(self) -> None:
        self.movies_by_name: Dict[str, Movie] = {}
        self.movies_by_genre: Dict[str, set[str]] = {}
        self.ratings_by_movie: Dict[str, List[float]] = {}
        self.user_ratings: Dict[str, Dict[str, float]] = {}

    def clear(self) -> None:
        """Clears all loaded data."""
        self.__init__()


# -------------------------------
# Parsing helpers
# -------------------------------


def load_movies_file(path: str, store: DataStore) -> int:
    """Load a movies file into the DataStore.
    
    Args:
        path: Path to the movies file (pipe-delimited with 'genre|movie_id|movie_name').
        store: The shared DataStore instance to populate.
    
    Returns:
        The number of valid movie rows loaded.
    
    Notes:
        - Lines that do not have exactly 3 fields are skipped.
        - Leading/trailing whitespace around fields is stripped.
        - Inline '#' comments are allowed and ignored.
        - Duplicate movie_name rows will overwrite previous entries (last one wins).
    """
    count = 0
    store.movies_by_name.clear()
    store.movies_by_genre.clear()

    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = _strip_inline_comment(raw).strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 3:
                # malformed line, skip
                continue
            genre, movie_id, movie_name = (p.strip() for p in parts)
            if not genre or not movie_id or not movie_name:
                continue
            m = Movie(genre=genre, movie_id=movie_id, name=movie_name)
            store.movies_by_name[movie_name] = m
            store.movies_by_genre.setdefault(genre, set()).add(movie_name)
            count += 1
    return count


def load_ratings_file(path: str, store: DataStore) -> int:
    """Load a ratings file into the DataStore.
    
    Args:
        path: Path to the ratings file (pipe-delimited with 'movie_name|rating|user_id').
        store: The shared DataStore instance to populate.
    
    Returns:
        The number of unique (user_id, movie_name) rating pairs loaded after applying
        last-write-wins on duplicates.
    
    Notes:
        - Lines that do not have exactly 3 fields are skipped.
        - Ratings must be parseable as float in [0, 5] inclusive; else skipped.
        - Inline comments after a '#' are ignored.
        - We keep ratings for movies not present in the movies file *only* in user_ratings
          (so they count toward the returned total), but we exclude them from
          ratings_by_movie so they do not affect popularity/genre analytics.
        - If a user rates the same movie multiple times, the last one wins.
    """
    store.ratings_by_movie.clear()
    store.user_ratings.clear()

    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = _strip_inline_comment(raw).strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 3:
                continue
            movie_name, rating_str, user_id = (p.strip() for p in parts)
            try:
                rating = float(rating_str)
            except ValueError:
                continue
            if not (0.0 <= rating <= 5.0):
                continue
            # Overwrite if the same user re-rates the same movie (last wins)
            store.user_ratings.setdefault(user_id, {})[movie_name] = rating

    # Build ratings_by_movie from user_ratings (only for movies present in movies file)
    store.ratings_by_movie.clear()
    for user_id, per_user in store.user_ratings.items():
        for movie_name, rating in per_user.items():
            if movie_name in store.movies_by_name:
                store.ratings_by_movie.setdefault(movie_name, []).append(rating)

    # Return number of unique (user, movie) pairs across *all* ratings (including unknown movies)
    unique_pairs = sum(len(per_user) for per_user in store.user_ratings.values())
    return unique_pairs


# -------------------------------
# Analytics helpers
# -------------------------------

def movie_average_rating(store: DataStore, movie_name: str) -> Optional[float]:
    """Compute the average rating for a single movie, or None if unrated."""
    ratings = store.ratings_by_movie.get(movie_name, [])
    if not ratings:
        return None
    return statistics.fmean(ratings)


def rank_movies_by_average(
    store: DataStore,
    movie_names: Optional[List[str]] = None,
    n: Optional[int] = None
) -> List[Tuple[str, float, int]]:
    """Return movies ranked by average rating.
    
    Args:
        store: DataStore with ratings.
        movie_names: If provided, restrict ranking to this list; if None, consider all
                     movies that have at least one rating.
        n: If provided, return only the top n.
    
    Returns:
        A list of tuples: (movie_name, average_rating, num_ratings), sorted by:
            - average_rating desc
            - num_ratings desc
            - movie_name asc (for tie-breaking)
    """
    items: List[Tuple[str, float, int]] = []
    # determine candidate set
    candidates = movie_names if movie_names is not None else list(store.ratings_by_movie.keys())
    for name in candidates:
        ratings = store.ratings_by_movie.get(name, [])
        if not ratings:
            continue
        avg = statistics.fmean(ratings)
        items.append((name, avg, len(ratings)))

    items.sort(key=lambda t: (-t[1], -t[2], t[0]))
    return items[:n] if n is not None else items


def rank_movies_in_genre(store: DataStore, genre: str, n: Optional[int] = None) -> List[Tuple[str, float, int]]:
    """Top movies in a given genre ranked on average rating.
    
    Args:
        store: DataStore with movies and ratings
        genre: The target genre
        n: Optional top-N limit
    
    Returns:
        A ranked list of (movie_name, avg_rating, num_ratings). Unrated movies are excluded.
    """
    genre_movies = list(store.movies_by_genre.get(genre, []))
    if not genre_movies:
        return []
    return rank_movies_by_average(store, movie_names=genre_movies, n=n)


def rank_genres_by_popularity(store: DataStore, n: Optional[int] = None) -> List[Tuple[str, float, int]]:
    """Rank genres by the average of movie-average-ratings within the genre.
    
    For each genre:
        - Compute the average rating for each rated movie in the genre.
        - Take the mean of those movie averages to represent the genre's popularity.
        - Genres with no rated movies are excluded.
    
    Args:
        store: DataStore with movies and ratings
        n: Optional top-N limit
    
    Returns:
        A ranked list of (genre, genre_average, num_rated_movies_in_genre).
        Sorted by genre_average desc, then num_rated_movies desc, then genre asc.
    """
    results: List[Tuple[str, float, int]] = []
    for genre, movie_names in store.movies_by_genre.items():
        avgs = []
        for name in movie_names:
            avg = movie_average_rating(store, name)
            if avg is not None:
                avgs.append(avg)
        if not avgs:
            continue
        genre_avg = statistics.fmean(avgs)
        results.append((genre, genre_avg, len(avgs)))

    results.sort(key=lambda t: (-t[1], -t[2], t[0]))
    return results[:n] if n is not None else results


def user_top_genre(store: DataStore, user_id: str) -> Optional[Tuple[str, float, int]]:
    """Determine the user's most preferred genre, using their own ratings.
    
    For each genre, compute the average of the ratings that THIS user has given to 
    movies in that genre. Return the genre with the highest average.
    
    Tie-breakers:
      1) More movies rated in that genre (desc)
      2) Genre name ascending
    
    Args:
        store: DataStore with movies and ratings
        user_id: The user's ID to evaluate
    
    Returns:
        (genre, user_avg_for_genre, count_movies_rated_in_genre) or None if user has no ratings.
    """
    per_user = store.user_ratings.get(user_id)
    if not per_user:
        return None

    by_genre: Dict[str, List[float]] = {}
    for name, rating in per_user.items():
        movie = store.movies_by_name.get(name)
        if movie is None:
            # Movie not present in movies list -> can't infer genre for this one
            continue
        by_genre.setdefault(movie.genre, []).append(rating)

    if not by_genre:
        return None

    rows: List[Tuple[str, float, int]] = []
    for genre, ratings in by_genre.items():
        rows.append((genre, statistics.fmean(ratings), len(ratings)))

    rows.sort(key=lambda t: (-t[1], -t[2], t[0]))
    return rows[0]


def recommend_for_user(store: DataStore, user_id: str, k: int = 3) -> List[Tuple[str, float, int, str]]:
    """Recommend up to k movies for the user.
    
    Strategy (per spec):
      - Find the user's top genre (by their own average ratings).
      - Within that genre, rank movies by overall popularity (average rating).
      - Recommend the top k that the user has NOT yet rated.
    
    Args:
        store: DataStore
        user_id: The target user
        k: Number of recommendations to return (default 3)
    
    Returns:
        A list of tuples (movie_name, avg_rating, num_ratings, genre).
        Returns an empty list if user has no ratings or if no eligible recommendations exist.
    """
    top = user_top_genre(store, user_id)
    if not top:
        return []
    genre, _, _ = top

    # Movies user already rated
    rated = set(store.user_ratings.get(user_id, {}).keys())

    ranked = rank_movies_in_genre(store, genre=genre, n=None)
    recs: List[Tuple[str, float, int, str]] = []
    for name, avg, cnt in ranked:
        if name in rated:
            continue
        recs.append((name, avg, cnt, genre))
        if len(recs) >= k:
            break
    return recs


# -------------------------------
# CLI utilities
# -------------------------------

def prompt_int(prompt: str, default: Optional[int] = None, min_value: Optional[int] = None) -> int:
    """Prompt the user for an integer with optional default and min-value enforcement."""
    while True:
        raw = input(f"{prompt} " + (f"[default {default}]: " if default is not None else ": ")).strip()
        if not raw and default is not None:
            return default
        try:
            val = int(raw)
            if min_value is not None and val < min_value:
                print(f"Please enter an integer >= {min_value}.")
                continue
            return val
        except ValueError:
            print("Please enter a valid integer.")


def require_loaded(store: DataStore) -> bool:
    """Check that both movies and ratings have been loaded before running analytics."""
    if not store.movies_by_name:
        print("! Load movies file first (option 1).")
        return False
    if not store.user_ratings:
        print("! Load ratings file first (option 2).")
        return False
    return True


def print_ranked_movies(rows: List[Tuple[str, float, int]], header: str) -> None:
    """Pretty-print a list of ranked movies."""
    print(f"\n{header}")
    if not rows:
        print("(no results)")
        return
    print(f"{'Rank':>4}  {'Movie':<50} {'Avg':>6}  {'#Ratings':>9}")
    print("-" * 76)
    for i, (name, avg, cnt) in enumerate(rows, start=1):
        print(f"{i:>4}  {name:<50.50} {avg:>6.2f}  {cnt:>9d}")


def print_ranked_genres(rows: List[Tuple[str, float, int]], header: str) -> None:
    """Pretty-print a list of ranked genres."""
    print(f"\n{header}")
    if not rows:
        print("(no results)")
        return
    print(f"{'Rank':>4}  {'Genre':<25} {'Genre Avg':>10}  {'Rated Movies':>13}")
    print("-" * 58)
    for i, (genre, avg, cnt) in enumerate(rows, start=1):
        print(f"{i:>4}  {genre:<25.25} {avg:>10.2f}  {cnt:>13d}")


def print_recommendations(rows: List[Tuple[str, float, int, str]], user_id: str) -> None:
    """Pretty-print recommendation list."""
    print(f"\nRecommendations for user {user_id}:")
    if not rows:
        print("(no recommendations available)")
        return
    print(f"{'Rank':>4}  {'Movie':<50} {'Avg':>6}  {'#Ratings':>9}  {'Genre':<20}")
    print("-" * 96)
    for i, (name, avg, cnt, genre) in enumerate(rows, start=1):
        print(f"{i:>4}  {name:<50.50} {avg:>6.2f}  {cnt:>9d}  {genre:<20.20}")


def main() -> None:
    store = DataStore()
    movies_loaded = False
    ratings_loaded = False

    print("=== Movie Recommender (CLI) ===")
    print("Python", sys.version.split()[0])
    while True:
        print("\nMenu:")
        print(" 1) Load movies file")
        print(" 2) Load ratings file")
        print(" 3) Top N movies (by average rating)")
        print(" 4) Top N movies in a genre")
        print(" 5) Top N genres (by average of movie averages)")
        print(" 6) User's most preferred genre")
        print(" 7) Recommend movies for a user (top 3)")
        print(" 8) Clear loaded data")
        print(" 9) Exit")
        choice = input("Choose an option [1-9]: ").strip()

        if choice == "1":
            path = input("Enter path to movies file: ").strip()
            try:
                n = load_movies_file(path, store)
                movies_loaded = True
                print(f"Loaded {n} movies from '{path}'.")
            except FileNotFoundError:
                print(f"File not found: {path}")
            except Exception as e:
                print(f"Error loading movies: {e}")

        elif choice == "2":
            path = input("Enter path to ratings file: ").strip()
            try:
                n = load_ratings_file(path, store)
                ratings_loaded = True
                print(f"Loaded {n} ratings from '{path}'.")
            except FileNotFoundError:
                print(f"File not found: {path}")
            except Exception as e:
                print(f"Error loading ratings: {e}")

        elif choice == "3":
            if not require_loaded(store):
                continue
            n = prompt_int("How many movies (N)?", default=10, min_value=1)
            rows = rank_movies_by_average(store, n=n)
            print_ranked_movies(rows, header=f"Top {n} Movies by Average Rating")

        elif choice == "4":
            if not require_loaded(store):
                continue
            genre = input("Enter genre: ").strip()
            n = prompt_int("How many movies (N)?", default=10, min_value=1)
            rows = rank_movies_in_genre(store, genre, n=n)
            print_ranked_movies(rows, header=f"Top {n} Movies in Genre '{genre}'")

        elif choice == "5":
            if not require_loaded(store):
                continue
            n = prompt_int("How many genres (N)?", default=10, min_value=1)
            rows = rank_genres_by_popularity(store, n=n)
            print_ranked_genres(rows, header=f"Top {n} Genres by Average of Movie Averages")

        elif choice == "6":
            if not require_loaded(store):
                continue
            user_id = input("Enter user id: ").strip()
            top = user_top_genre(store, user_id)
            if not top:
                print(f"User '{user_id}' has no ratings or no genre-mapped ratings.")
            else:
                genre, avg, cnt = top
                print(f"User '{user_id}' top genre: {genre} (avg={avg:.2f} over {cnt} movie(s))")

        elif choice == "7":
            if not require_loaded(store):
                continue
            user_id = input("Enter user id: ").strip()
            recs = recommend_for_user(store, user_id, k=3)
            print_recommendations(recs, user_id=user_id)

        elif choice == "8":
            store.clear()
            movies_loaded = ratings_loaded = False
            print("Cleared all loaded data.")

        elif choice == "9":
            print("Goodbye!")
            break

        else:
            print("Invalid option. Please choose a number 1-9.")


if __name__ == "__main__":
    main()
