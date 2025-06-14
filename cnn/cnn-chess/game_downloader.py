# game_downloader.py
import requests
import zstandard as zstd
from chess.pgn import read_game

LICHESS_ELITE_URL = "https://database.nikonoel.fr/lichess_elite_2023.pgn.zst"

def download_games(url=LICHESS_ELITE_URL, max_games=100000):
    response = requests.get(url, stream=True)
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(response.raw) as reader:
        pgn = reader.read().decode('utf-8')
    games = []
    while len(games) < max_games:
        game = read_game(pgn)
        if game is None: break
        if game.headers["WhiteElo"] > 2400 and game.headers["BlackElo"] > 2400:
            games.append(game)
    return games
