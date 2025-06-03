from cachetools import TTLCache

song_genre_cache = TTLCache(maxsize=1000, ttl=3600 * 24 * 7)
