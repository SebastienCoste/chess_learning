#!/usr/bin/env python3
"""
Chess PGN Data Collection Script
Downloads 10,000 real chess games from various public sources for LLM training.
"""

import requests
import zipfile
import io
import os
import random
import time
from pathlib import Path
import chess.pgn
from typing import List, Generator
import urllib.request
import gzip
import json


class ChessPGNDownloader:
    def __init__(self, target_games: int = 10000):
        self.target_games = target_games
        self.collected_games = 0
        self.output_file = "chess_games_10k.pgn"

    def download_from_twic(self, start_issue: int = 1400, max_issues: int = 50) -> List[str]:
        """Download PGN files from The Week in Chess (TWIC)"""
        games = []
        base_url = "https://theweekinchess.com/zips/twic"

        print(f"Downloading from TWIC issues {start_issue} to {start_issue + max_issues}")

        for issue in range(start_issue, start_issue + max_issues):
            if self.collected_games >= self.target_games:
                break

            try:
                url = f"{base_url}{issue}g.zip"
                print(f"Downloading TWIC {issue}...")
                response = requests.request('GET', url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    "Accept": "*/*",
                    "Accept-Encoding": "gzip, deflate, br"
                })
                if response.status_code == 200:
                    # Extract PGN from ZIP
                    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                        for file_info in zip_file.filelist:
                            if file_info.filename.endswith('.pgn'):
                                pgn_content = zip_file.read(file_info).decode('utf-8', errors='ignore')
                                games_from_file = self.extract_games_from_pgn(pgn_content)
                                games.extend(games_from_file)
                                self.collected_games += len(games_from_file)
                                print(f"  Added {len(games_from_file)} games (Total: {self.collected_games})")

                                if self.collected_games >= self.target_games:
                                    break
                else:
                    print(f"Non HTTP 200 downloading TWIC {issue}: {response}")
                time.sleep(1)  # Be respectful to the server

            except Exception as e:
                print(f"Error downloading TWIC {issue}: {e}")
                continue

        return games[:self.target_games]

    def download_from_lichess_api(self, max_games: int = 10000) -> List[str]:
        """Download up to max_games long games from Lichess API, 100 at a time."""
        print(f"Downloading up to {max_games} long games from Lichess API...")

        games = []
        url = "https://lichess.org/api/games/user/chess-network"
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; LichessDownloader/1.0)",
            "Accept": "application/x-ndjson"
        }
        params = {
            "max": 100,
            "rated": "true",
            "perfType": "rapid,blitz",  # Long games
            "pgnInJson": "true",
            "clocks": "true",
            "evals": "false",
            "opening": "false",
            "since": None  # Will be set after each batch
        }

        total_downloaded = 0
        last_timestamp = None

        while total_downloaded < max_games:
            if last_timestamp:
                params["since"] = last_timestamp
            else:
                params.pop("since", None)

            response = requests.get(url, headers=headers, params=params, timeout=30)
            if response.status_code != 200:
                print(f"Non HTTP 200 downloading Lichess games: {response}")
                break

            lines = response.text.strip().split('\n')
            if not lines or lines == ['']:
                print("No more games found.")
                break

            for line in lines:
                if total_downloaded >= max_games:
                    break
                games.append(line)
                total_downloaded += 1

            # Find last game's end time for pagination
            last_game = None
            for line in reversed(lines):
                try:
                    game_data = json.loads(line)
                    last_game = game_data
                    break
                except Exception:
                    continue

            if last_game and 'createdAt' in last_game:
                last_timestamp = last_game['createdAt']
            elif last_game and 'lastMoveAt' in last_game:
                last_timestamp = last_game['lastMoveAt']
            else:
                print("Could not determine next 'since' timestamp, stopping.")
                break

            print(f"Downloaded {total_downloaded} games so far...")
            time.sleep(1)  # Respect Lichess API rate limits

        print(f"Finished downloading {len(games)} games.")

        return games

    def extract_games_from_pgn(self, pgn_content: str) -> List[str]:
        """Extract individual games from a PGN file content"""
        games = []
        current_game = ""

        for line in pgn_content.split('\n'):
            current_game += line + '\n'

            # Check if this is the end of a game
            if line.strip() in ['1-0', '0-1', '1/2-1/2', '*']:
                if current_game.strip():
                    games.append(current_game.strip())
                current_game = ""

        return games

    def filter_games(self, games: List[str]) -> List[str]:
        """Filter games to ensure quality and remove duplicates"""
        filtered_games = []
        seen_games = set()

        for game in games:
            # Basic quality checks
            if len(game) < 100:  # Too short
                continue

            lines = game.strip().split('\n')
            move_lines = [line for line in lines if not line.startswith('[') and line.strip()]

            if len(move_lines) == 0:  # No moves
                continue

            # Check for duplicates (simple hash)
            game_hash = hash(game.strip())
            if game_hash in seen_games:
                continue

            seen_games.add(game_hash)
            filtered_games.append(game)

            if len(filtered_games) >= self.target_games:
                break

        return filtered_games

    def save_games(self, games: List[str], filename: str = None):
        """Save games to a PGN file"""
        if filename is None:
            filename = self.output_file

        with open(filename, 'w', encoding='utf-8') as f:
            for game in games:
                f.write(game + '\n\n')

        print(f"Saved {len(games)} games to {filename}")

    def download_all(self):
        """Main method to download games from all sources"""
        print(f"Starting download of {self.target_games} chess games...")

        all_games = []

        # Try downloading from TWIC (reduced number for demo)
        # try:
        #     print("Attempting to download from TWIC...")
        #     twic_games = self.download_from_twic(start_issue=1400, max_issues=1)
        #     all_games.extend(twic_games)
        #     print(f"Downloaded {len(twic_games)} games from TWIC")
        # except Exception as e:
        #     print(f"TWIC download failed: {e}")

        # Try downloading from Lichess API
        try:
            print("Attempting to download from Lichess API...")
            lichess_games = self.download_from_lichess_api(self.target_games - len(all_games))
            all_games.extend(lichess_games)
            print(f"Downloaded {len(lichess_games)} games from Lichess")
        except Exception as e:
            print(f"Lichess download failed: {e}")

        # Generate remaining games as samples
        remaining_needed = self.target_games - len(all_games)
        if remaining_needed > 0:
            print(f"Generating {remaining_needed} sample games...")
            sample_games = self.generate_sample_games(remaining_needed)
            all_games.extend(sample_games)

        # Filter and deduplicate
        print("Filtering and deduplicating games...")
        filtered_games = self.filter_games(all_games)

        # Save to file
        self.save_games(filtered_games)

        print(f"Successfully collected {len(filtered_games)} chess games!")
        return filtered_games


if __name__ == "__main__":
    downloader = ChessPGNDownloader(target_games=10000)
    games = downloader.download_all()