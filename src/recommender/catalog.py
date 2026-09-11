"""Tiny in-memory movie catalog that backs the recommender's tools.

Deliberately small and hand-curated — the point is to give the tool
calls a real corpus to search over so Phoenix traces show meaningful
retrieval work, not to be an exhaustive database.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Movie:
    title: str
    year: int
    genres: tuple[str, ...]
    director: str
    tags: tuple[str, ...] = field(default_factory=tuple)
    summary: str = ""


CATALOG: tuple[Movie, ...] = (
    Movie("Arrival", 2016, ("science fiction", "drama"), "Denis Villeneuve",
          ("cerebral", "slow-burn", "linguistics", "first-contact"),
          "A linguist works to communicate with alien visitors as world tensions rise."),
    Movie("Blade Runner 2049", 2017, ("science fiction", "thriller"), "Denis Villeneuve",
          ("atmospheric", "philosophical", "slow-burn", "neo-noir"),
          "A young blade runner's discovery leads him to a former blade runner missing for thirty years."),
    Movie("Ex Machina", 2014, ("science fiction", "thriller"), "Alex Garland",
          ("cerebral", "AI", "slow-burn"),
          "A programmer evaluates the human qualities of a strikingly advanced android."),
    Movie("2001: A Space Odyssey", 1968, ("science fiction",), "Stanley Kubrick",
          ("classic", "slow-burn", "cerebral"),
          "A voyage to Jupiter with the sentient computer HAL after mysterious monoliths appear."),
    Movie("The Matrix", 1999, ("science fiction", "action"), "The Wachowskis",
          ("mind-bending", "cyberpunk"),
          "A hacker discovers reality as he knows it is a simulation."),
    Movie("Children of Men", 2006, ("science fiction", "thriller"), "Alfonso Cuaron",
          ("dystopian", "long-take"),
          "In a world where humans can no longer procreate, a man escorts a miraculous pregnant woman to safety."),
    Movie("Solaris", 1972, ("science fiction", "drama"), "Andrei Tarkovsky",
          ("arthouse", "contemplative", "classic"),
          "A psychologist investigates strange happenings on a space station orbiting a mysterious planet."),
    Movie("Alien", 1979, ("science fiction", "horror"), "Ridley Scott",
          ("classic", "suspense"),
          "The crew of a commercial spacecraft encounter a deadly lifeform after investigating an unknown transmission."),

    Movie("The Godfather", 1972, ("crime", "drama"), "Francis Ford Coppola",
          ("classic", "family-saga"),
          "The aging patriarch of a crime dynasty transfers control to his reluctant son."),
    Movie("The Godfather Part II", 1974, ("crime", "drama"), "Francis Ford Coppola",
          ("classic", "family-saga"),
          "The early life of Vito Corleone in parallel with his son Michael's expansion of the family crime syndicate."),
    Movie("Goodfellas", 1990, ("crime", "drama"), "Martin Scorsese",
          ("gangster", "based-on-true-story"),
          "The rise and fall of a mob associate over three decades."),
    Movie("Heat", 1995, ("crime", "thriller"), "Michael Mann",
          ("heist", "slow-burn"),
          "A career criminal and a determined detective circle each other in Los Angeles."),
    Movie("The Departed", 2006, ("crime", "thriller"), "Martin Scorsese",
          ("undercover", "twisty"),
          "An undercover cop and a mole in the police attempt to identify each other."),

    Movie("Se7en", 1995, ("thriller", "crime"), "David Fincher",
          ("dark", "neo-noir", "serial-killer"),
          "Two detectives hunt a serial killer using the seven deadly sins as his motif."),
    Movie("Zodiac", 2007, ("thriller", "crime"), "David Fincher",
          ("dark", "based-on-true-story", "slow-burn"),
          "A cartoonist becomes obsessed with tracking down the Zodiac Killer."),
    Movie("Prisoners", 2013, ("thriller", "drama"), "Denis Villeneuve",
          ("dark", "slow-burn", "moral-dilemma"),
          "A father takes matters into his own hands after his daughter goes missing."),

    Movie("The Lord of the Rings: The Fellowship of the Ring", 2001, ("fantasy", "adventure"), "Peter Jackson",
          ("epic", "quest"),
          "A hobbit sets out to destroy an evil ring, joined by a fellowship of heroes."),
    Movie("The Lord of the Rings: The Two Towers", 2002, ("fantasy", "adventure"), "Peter Jackson",
          ("epic", "quest"), ""),
    Movie("Pan's Labyrinth", 2006, ("fantasy", "drama"), "Guillermo del Toro",
          ("dark", "fairy-tale", "spanish-language"),
          "In post-Civil War Spain, a young girl escapes into a mythical labyrinth."),

    Movie("Amelie", 2001, ("comedy", "romance"), "Jean-Pierre Jeunet",
          ("whimsical", "warm", "french-language"),
          "A shy waitress in Paris orchestrates the happiness of those around her."),
    Movie("Notting Hill", 1999, ("romance", "comedy"), "Roger Michell",
          ("feel-good",),
          "A London bookseller's life changes when a famous actress walks into his shop."),
    Movie("When Harry Met Sally", 1989, ("romance", "comedy"), "Rob Reiner",
          ("classic", "witty"), ""),
    Movie("Before Sunrise", 1995, ("romance", "drama"), "Richard Linklater",
          ("dialogue-driven", "bittersweet"),
          "A young American man and a French woman meet on a train and spend one night in Vienna."),

    Movie("Spirited Away", 2001, ("animation", "fantasy"), "Hayao Miyazaki",
          ("magical", "family-friendly", "japanese"),
          "A young girl navigates a strange world of spirits to rescue her parents."),
    Movie("Grave of the Fireflies", 1988, ("animation", "drama"), "Isao Takahata",
          ("devastating", "war", "japanese"), ""),
    Movie("WALL-E", 2008, ("animation", "science fiction"), "Andrew Stanton",
          ("family-friendly", "environmental"),
          "In the distant future, a small waste-collecting robot embarks on a journey that will decide humankind's fate."),
    Movie("Persepolis", 2007, ("animation", "drama"), "Marjane Satrapi",
          ("autobiographical", "thoughtful", "adult"),
          "A young girl comes of age against the backdrop of the Iranian Revolution."),
    Movie("Waltz with Bashir", 2008, ("animation", "documentary"), "Ari Folman",
          ("war", "adult", "memoir"), ""),

    Movie("Parasite", 2019, ("thriller", "drama"), "Bong Joon-ho",
          ("social-satire", "korean", "genre-blending"),
          "A poor family schemes to work for a wealthy household, with unexpected consequences."),
    Movie("Shoplifters", 2018, ("drama",), "Hirokazu Kore-eda",
          ("socially-observant", "japanese"),
          "A family of small-time crooks take in a homeless child."),
    Movie("Mad Max: Fury Road", 2015, ("action", "adventure"), "George Miller",
          ("adrenaline", "post-apocalyptic"),
          "In a post-apocalyptic wasteland, a woman rebels against a tyrannical ruler."),

    Movie("Inception", 2010, ("science fiction", "thriller"), "Christopher Nolan",
          ("mind-bending", "heist"),
          "A thief who steals corporate secrets through dream-sharing technology is offered a chance at redemption."),
    Movie("Double Indemnity", 1944, ("film noir", "crime"), "Billy Wilder",
          ("classic", "black-and-white"),
          "An insurance salesman is enticed by a seductive housewife into a scheme to murder her husband."),
    Movie("The Third Man", 1949, ("film noir", "mystery"), "Carol Reed",
          ("classic", "black-and-white", "post-war"),
          "A writer investigates the mysterious death of his friend in post-war Vienna."),

    Movie("The Grand Budapest Hotel", 2014, ("comedy",), "Wes Anderson",
          ("whimsical", "quirky"),
          "The adventures of a legendary concierge at a famous European hotel."),
    Movie("Paddington 2", 2017, ("comedy", "family"), "Paul King",
          ("feel-good", "wholesome"),
          "Paddington picks up a series of odd jobs to buy the perfect present for his aunt."),

    Movie("Unforgiven", 1992, ("western",), "Clint Eastwood",
          ("revisionist",),
          "A retired outlaw takes on one last job."),
    Movie("The Good, the Bad and the Ugly", 1966, ("western",), "Sergio Leone",
          ("classic", "epic"), ""),
    Movie("No Country for Old Men", 2007, ("thriller", "western"), "The Coen Brothers",
          ("neo-western", "dark", "slow-burn"),
          "A hunter's chance discovery of drug money in the Texas desert sets off a violent chain of events."),

    Movie("Manchester by the Sea", 2016, ("drama",), "Kenneth Lonergan",
          ("devastating", "reflective"),
          "A man returns to his hometown to care for his nephew after his brother's death."),
    Movie("Marriage Story", 2019, ("drama",), "Noah Baumbach",
          ("bittersweet", "dialogue-driven"), ""),
    Movie("The Pianist", 2002, ("drama", "war"), "Roman Polanski",
          ("devastating", "based-on-true-story"), ""),
    Movie("Schindler's List", 1993, ("drama", "war"), "Steven Spielberg",
          ("classic", "devastating", "black-and-white"), ""),

    Movie("Man on Wire", 2008, ("documentary",), "James Marsh",
          ("uplifting",), ""),
    Movie("Won't You Be My Neighbor?", 2018, ("documentary",), "Morgan Neville",
          ("warm", "uplifting"), ""),
    Movie("Free Solo", 2018, ("documentary",), "Elizabeth Chai Vasarhelyi",
          ("adrenaline",), ""),

    Movie("Baby Driver", 2017, ("action",), "Edgar Wright",
          ("heist", "kinetic"), ""),
    Movie("John Wick", 2014, ("action",), "Chad Stahelski",
          ("kinetic",), ""),

    Movie("La La Land", 2016, ("musical", "romance"), "Damien Chazelle",
          ("bittersweet",), ""),
)


_CATALOG_BY_TITLE: dict[str, Movie] = {m.title.lower(): m for m in CATALOG}


def _match_score(m: Movie, query: str) -> int:
    q = query.lower().strip()
    if not q:
        return 0
    haystack = " ".join(
        (m.title.lower(), " ".join(m.genres), " ".join(m.tags), m.director.lower(), m.summary.lower())
    )
    return sum(1 for token in q.split() if token in haystack)


def search(query: str, limit: int = 5) -> list[Movie]:
    scored = [(m, _match_score(m, query)) for m in CATALOG]
    scored = [pair for pair in scored if pair[1] > 0]
    scored.sort(key=lambda p: (-p[1], p[0].title))
    return [m for m, _ in scored[:limit]]


def get(title: str) -> Movie | None:
    return _CATALOG_BY_TITLE.get(title.lower())


def as_dict(m: Movie) -> dict:
    return {
        "title": m.title,
        "year": m.year,
        "genres": list(m.genres),
        "director": m.director,
        "tags": list(m.tags),
        "summary": m.summary,
    }
