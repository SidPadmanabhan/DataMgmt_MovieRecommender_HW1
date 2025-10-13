#!/usr/bin/env python3
"""
test_movie_recommender.py

Automated tests for movie_recommender.py (Python 3.12).

Run:
    python test_movie_recommender.py

What it does:
- Writes small fixture files (good, malformed, empty) into a temp folder.
- Loads them with movie_recommender.load_* functions.
- Calls every feature function and compares results against expected values.
- Exercises edge cases: malformed rows, empty files, duplicate ratings, non-numeric ratings, out-of-range ratings, tie sorting rules, ratings for movies not in movies file.

Output:
- Human-readable PASS/FAIL lines plus a summary.
- Exits with code 0 on all-pass; 1 if any test fails.
"""

from __future__ import annotations
import sys
import math
from pathlib import Path
from typing import List, Tuple

# Allow importing the implementation placed in the same directory
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import movie_recommender as mr  # type: ignore


TMP = HERE / "_tmp_fixtures"
TMP.mkdir(exist_ok=True)


def write(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def approx(x: float, y: float, tol: float = 1e-6) -> bool:
    return abs(x - y) <= tol


class TestRunner:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def check(self, name: str, cond: bool, detail: str = "") -> None:
        if cond:
            print(f"[PASS] {name}")
            self.passed += 1
        else:
            print(f"[FAIL] {name} :: {detail}")
            self.failed += 1

    def summary(self) -> int:
        total = self.passed + self.failed
        print("\n=== Test Summary ===")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Total : {total}")
        return 0 if self.failed == 0 else 1


def build_good_fixtures() -> tuple[Path, Path]:
    """Create a 'good' movies and ratings set with controlled expectations."""
    movies = TMP / "movies_good.txt"
    ratings = TMP / "ratings_good.txt"

    write(
        movies,
        """
        # genre|movie_id|movie_name
        Action|m1|Alpha (2001)
        Action|m2|Bravo (1999)
        Drama|m3|Charlie (2010)
        Drama|m4|Delta (2011)
        Comedy|m5|Echo (2005)
        Comedy|m6|Foxtrot (2007)
        """
    )

    # Notes:
    # - Duplicate rating: u1 rates Bravo twice; last wins (4.0).
    # - Nonexistent movie in movies file: Ghost (2012) -> should not affect genre-based calcs.
    # - Non-numeric and out-of-range ratings appear in malformed fixtures, not here.
    write(
        ratings,
        """
        # movie_name|rating|user_id
        Alpha (2001)|5|u1
        Alpha (2001)|4|u2
        Bravo (1999)|2|u1
        Bravo (1999)|4|u1   # overwrite u1's rating to 4 (last wins)
        Bravo (1999)|4|u3
        Charlie (2010)|5|u1
        Charlie (2010)|5|u2
        Charlie (2010)|1|u3
        Delta (2011)|3|u2
        Foxtrot (2007)|4|u2
        Foxtrot (2007)|5|u3
        Ghost (2012)|5|u1   # not in movies file
        """
    )
    return movies, ratings


def build_malformed_fixtures() -> tuple[Path, Path]:
    """Create malformed/edge-case files."""
    movies = TMP / "movies_bad.txt"
    ratings = TMP / "ratings_bad.txt"

    write(
        movies,
        """
        # Missing fields
        OnlyTwo|Fields
        # Empty / comment lines should be ignored
        # Good row:
        Action|mx|Xray (2000)
        # Another good row:
        Drama|my|Yankee (2001)
        """
    )

    write(
        ratings,
        """
        # Bad rows:
        NotEnough|Fields
        Xray (2000)|not_a_number|u1
        Xray (2000)|6|u2            # out of range (>5)
        # Good rows:
        Xray (2000)|4|u1
        Xray (2000)|5|u3
        Yankee (2001)|3|u2
        """
    )
    return movies, ratings


def build_empty_fixtures() -> tuple[Path, Path]:
    movies = TMP / "movies_empty.txt"
    ratings = TMP / "ratings_empty.txt"
    write(movies, "")
    write(ratings, "")
    return movies, ratings


def tuple3_str(rows: List[Tuple[str, float, int]]) -> List[Tuple[str, float, int]]:
    """Round floats for stable comparison/printing."""
    return [(name, round(avg, 3), cnt) for (name, avg, cnt) in rows]


def tuple4_str(rows: List[Tuple[str, float, int, str]]) -> List[Tuple[str, float, int, str]]:
    return [(name, round(avg, 3), cnt, g) for (name, avg, cnt, g) in rows]


def run_tests() -> int:
    T = TestRunner()

    # ---------- Good fixtures ----------
    mfile, rfile = build_good_fixtures()
    store = mr.DataStore()
    n_movies = mr.load_movies_file(str(mfile), store)
    n_ratings = mr.load_ratings_file(str(rfile), store)
    T.check("good: loaded movie rows", n_movies == 6, f"got {n_movies}")
    T.check("good: loaded rating rows (unique per user/movie)", n_ratings == 11, f"got {n_ratings}")

    # Movie averages (computed by hand):
    # Alpha: (5 + 4)/2 = 4.5 (2)
    # Bravo: (4 from u1 overwrite, +4 from u3)/2 = 4.0 (2)
    # Charlie: (5 + 5 + 1) / 3 = 3.6667 (3)
    # Delta: (3) (1)
    # Foxtrot: (4 + 5)/2 = 4.5 (2)
    # Echo: no ratings
    top_all = mr.rank_movies_by_average(store, n=None)
    expect_top_all = [
        ("Alpha (2001)", 4.5, 2),
        ("Foxtrot (2007)", 4.5, 2),
        ("Bravo (1999)", 4.0, 2),
        ("Charlie (2010)", 3.6666666667, 3),
        ("Delta (2011)", 3.0, 1),
    ]
    # Tie rule: avg desc, then count desc, then name asc -> Alpha before Foxtrot
    T.check(
        "feature: Top movies overall order & values",
        tuple3_str(top_all) == tuple3_str(expect_top_all),
        f"expected {tuple3_str(expect_top_all)}, got {tuple3_str(top_all)}"
    )

    # Top N in genre: Action has Alpha (4.5), Bravo (4.0)
    top_action = mr.rank_movies_in_genre(store, "Action", n=None)
    T.check(
        "feature: Top movies in genre=Action",
        tuple3_str(top_action) == tuple3_str([("Alpha (2001)", 4.5, 2), ("Bravo (1999)", 4.0, 2)]),
        f"got {tuple3_str(top_action)}"
    )

    # Genre popularity:
    # Comedy: Foxtrot only (Echo unrated) -> 4.5
    # Action: mean(4.5, 4.0) = 4.25
    # Drama: mean(3.6667, 3.0) = 3.3333
    top_genres = mr.rank_genres_by_popularity(store, n=None)
    expect_genres = [
        ("Comedy", 4.5, 1),
        ("Action", 4.25, 2),
        ("Drama", (3.6666666667 + 3.0)/2, 2),
    ]
    T.check(
        "feature: Top genres order & values",
        tuple3_str(top_genres) == tuple3_str(expect_genres),
        f"expected {tuple3_str(expect_genres)}, got {tuple3_str(top_genres)}"
    )

    # User preference (top genre) for u1:
    # u1 in Action: [5,4] -> 4.5; Drama: [5] -> 5.0 => expect Drama
    u1_top = mr.user_top_genre(store, "u1")
    T.check(
        "feature: User top genre for u1",
        u1_top is not None and u1_top[0] == "Drama" and approx(u1_top[1], 5.0) and u1_top[2] == 1,
        f"got {u1_top}"
    )

    # Recommendations for u1 in Drama: ranked overall are Charlie(3.667), Delta(3.0)
    # u1 already rated Charlie, not Delta -> expect only Delta
    u1_recs = mr.recommend_for_user(store, "u1", k=3)
    T.check(
        "feature: Recommendations for u1 (from top genre)",
        tuple4_str(u1_recs) == tuple4_str([("Delta (2011)", 3.0, 1, "Drama")]),
        f"got {tuple4_str(u1_recs)}"
    )

    # User with tie across genres: u2
    # u2 ratings: Alpha(4), Charlie(5), Delta(3), Foxtrot(4)
    # Per-genre: Action=4.0 (1 movie), Drama=4.0 (2 movies), Comedy=4.0 (1 movie)
    # Tie-breaker -> more movies rated: Drama wins
    u2_top = mr.user_top_genre(store, "u2")
    T.check(
        "feature: User top genre tie-breakers (u2)",
        u2_top is not None and u2_top[0] == "Drama" and approx(u2_top[1], 4.0) and u2_top[2] == 2,
        f"got {u2_top}"
    )

    # Recommendations for u2: In Drama, u2 has rated both -> expect empty list
    u2_recs = mr.recommend_for_user(store, "u2", k=3)
    T.check(
        "feature: Recommendations empty when user rated all in top genre (u2)",
        u2_recs == [],
        f"got {u2_recs}"
    )

    # ---------- Malformed/edge fixtures ----------
    mbad, rbad = build_malformed_fixtures()
    store2 = mr.DataStore()
    n_movies2 = mr.load_movies_file(str(mbad), store2)
    n_ratings2 = mr.load_ratings_file(str(rbad), store2)
    # Only 2 valid movie rows should load
    T.check("edge: malformed movies rows skipped", n_movies2 == 2, f"got {n_movies2}")
    # Ratings: only 3 valid rows (two for Xray from u1 & u3, one for Yankee from u2)
    T.check("edge: bad rating rows skipped", n_ratings2 == 3, f"got {n_ratings2}")

    # Duplicate user/movie rating overwrite already exercised in good fixtures (Bravo/u1)
    # Verify averages for Xray and Yankee
    top_all2 = mr.rank_movies_by_average(store2, n=None)
    # Xray: (4 + 5)/2 = 4.5 ; Yankee: 3.0
    T.check(
        "edge: computed averages with valid rows only",
        tuple3_str(top_all2) == tuple3_str([("Xray (2000)", 4.5, 2), ("Yankee (2001)", 3.0, 1)]),
        f"got {tuple3_str(top_all2)}"
    )

    # ---------- Empty files ----------
    mempty, rempty = build_empty_fixtures()
    store3 = mr.DataStore()
    n_movies3 = mr.load_movies_file(str(mempty), store3)
    n_ratings3 = mr.load_ratings_file(str(rempty), store3)
    T.check("edge: empty movies file loads 0", n_movies3 == 0, f"got {n_movies3}")
    T.check("edge: empty ratings file loads 0", n_ratings3 == 0, f"got {n_ratings3}")
    # No results anywhere
    T.check("edge: no top movies when no data", mr.rank_movies_by_average(store3) == [], "expected empty")
    T.check("edge: no top genres when no data", mr.rank_genres_by_popularity(store3) == [], "expected empty")
    T.check("edge: user_top_genre None if no ratings", mr.user_top_genre(store3, "who") is None, "expected None")
    T.check("edge: recommend empty if no user ratings", mr.recommend_for_user(store3, "who") == [], "expected empty")

    return T.summary()


if __name__ == "__main__":
    raise SystemExit(run_tests())
