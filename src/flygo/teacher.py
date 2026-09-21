"""KataGo analysis protocol adapters for reproducible teacher targets."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, TextIO

from flygo.atomic import write_text
from flygo.dataset import GameRecord, sample_id
from flygo.go import komi_for

_GTP_COLUMNS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


def action_to_gtp(action: int, size: int) -> str:
    """Convert a row-major FlyGo action to a GTP vertex."""
    if action == size * size:
        return "pass"
    if not 0 <= action < size * size:
        raise ValueError("Action is outside the board")
    row, column = divmod(action, size)
    return f"{_GTP_COLUMNS[column]}{size - row}"


def gtp_to_action(vertex: str, size: int) -> int:
    """Convert a GTP vertex to a row-major FlyGo action."""
    if vertex.lower() == "pass":
        return size * size
    text = vertex.upper()
    if len(text) < 2 or text[0] not in _GTP_COLUMNS[:size]:
        raise ValueError(f"Invalid GTP vertex {vertex!r}")
    try:
        row_number = int(text[1:])
    except ValueError as error:
        raise ValueError(f"Invalid GTP vertex {vertex!r}") from error
    row = size - row_number
    column = _GTP_COLUMNS.index(text[0])
    if not 0 <= row < size:
        raise ValueError(f"Invalid GTP vertex {vertex!r}")
    return row * size + column


def katago_queries(
    games: Sequence[GameRecord],
    *,
    visits: int,
    rules: str = "tromp-taylor",
    stride: int = 1,
) -> Iterable[dict[str, Any]]:
    """Yield one versionable KataGo JSON analysis query per complete game."""
    if visits < 1:
        raise ValueError("visits must be positive")
    if stride < 1:
        raise ValueError("stride must be positive")
    for game in games:
        moves: list[list[str]] = []
        player = "B"
        for action in game.moves:
            moves.append([player, action_to_gtp(action, game.size)])
            player = "W" if player == "B" else "B"
        yield {
            "id": game.game_id,
            "moves": moves,
            "rules": rules,
            "komi": komi_for(game.size),
            "boardXSize": game.size,
            "boardYSize": game.size,
            "analyzeTurns": list(range(0, len(game.moves), stride)),
            "maxVisits": visits,
            "includePolicy": True,
        }


def write_katago_queries(
    games: Sequence[GameRecord],
    output: Path,
    *,
    visits: int,
    stride: int = 1,
) -> int:
    """Write KataGo JSON Lines input and return the query count."""
    queries = list(katago_queries(games, visits=visits, stride=stride))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(query) + "\n" for query in queries))
    return len(queries)


def import_katago_analysis(
    input_path: Path,
    output: Path,
    games: Sequence[GameRecord],
    *,
    winrate_perspective: str = "black",
) -> int:
    """Convert KataGo JSON Lines output to FlyGo teacher targets atomically."""
    if winrate_perspective not in {"black", "white", "side-to-move"}:
        raise ValueError("winrate perspective must be black, white, or side-to-move")
    sizes = {game.game_id: game.size for game in games}
    targets: dict[str, dict[str, Any]] = {}
    provisional = 0
    for line_number, line in enumerate(input_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            game_id = str(payload["id"])
            turn = int(payload["turnNumber"])
            during_search = payload.get("isDuringSearch", False)
            if not isinstance(during_search, bool):
                raise ValueError("isDuringSearch must be a boolean")
            root = payload["rootInfo"]
            winrate = float(root["winrate"])
            move_infos = payload["moveInfos"]
            size = sizes[game_id]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid KataGo response on line {line_number}") from error
        if during_search:
            provisional += 1
            continue
        identifier = sample_id(game_id, turn)
        if identifier in targets:
            raise ValueError(f"Duplicate final KataGo response for {identifier}")
        visits = [(gtp_to_action(item["move"], size), int(item["visits"])) for item in move_infos]
        visits = [(action, count) for action, count in visits if count > 0]
        if not visits:
            raise ValueError(f"KataGo response for {identifier} has no visited moves")
        value = 2 * winrate - 1
        if (winrate_perspective == "black" and turn % 2 == 1) or (
            winrate_perspective == "white" and turn % 2 == 0
        ):
            value = -value
        targets[identifier] = {
            "sample_id": identifier,
            "policy": visits,
            "value": value,
        }
    if provisional and not targets:
        raise ValueError(
            f"KataGo analysis has only provisional responses ({provisional}); no search finished"
        )

    def write_targets(temporary: TextIO) -> None:
        for identifier in sorted(targets):
            temporary.write(json.dumps(targets[identifier]) + "\n")

    write_text(output, write_targets)
    return len(targets)
