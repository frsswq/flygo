/**
 * Tromp-Taylor Go rules for square boards from 5x5 through 9x9.
 *
 * This mirrors `src/flygo/go.py` exactly: area scoring, positional superko,
 * self-capture allowed, and two consecutive passes ending the game. Both
 * engines replay `shared/rules-conformance.json`, so keep them aligned by
 * regenerating that fixture instead of editing either engine alone.
 */

export type Stone = -1 | 0 | 1;
export type Player = 1 | -1;

export const BOARD_SIZES = [5, 6, 7, 8, 9] as const;
export const MIN_BOARD_SIZE = 5;
export const MAX_BOARD_SIZE = 9;
export const DEFAULT_BOARD_SIZE = 9;
export const RULESET = "Tromp-Taylor, area scoring, positional superko";

/** Komi is 0 on 5x5 so the published solved result of Black +25 holds. */
const KOMI_BY_SIZE: Readonly<Record<number, number>> = {
  5: 0,
  6: 7.5,
  7: 7.5,
  8: 7.5,
  9: 7.5,
};

export interface Position {
  readonly board: readonly Stone[];
  readonly history: ReadonlySet<string>;
  readonly size: number;
  readonly toPlay: Player;
}

export interface GameResult {
  readonly blackArea: number;
  readonly komi: number;
  readonly label: string;
  readonly margin: number;
  readonly whiteArea: number;
  readonly winner: 0 | Player;
}

export class IllegalMoveError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "IllegalMoveError";
  }
}

export const pointsFor = (size: number): number => size * size;

export const passActionFor = (size: number): number => pointsFor(size);

export const emptyBoard = (size: number): Stone[] =>
  Array.from({ length: pointsFor(size) }, () => 0 as Stone);

export const komiFor = (size: number): number => {
  const komi = KOMI_BY_SIZE[size];
  if (komi === undefined) {
    throw new RangeError(`No komi is defined for a ${size}x${size} board`);
  }
  return komi;
};

const symbolOf = (stone: Stone): string => {
  if (stone === 1) {
    return "+";
  }
  if (stone === -1) {
    return "-";
  }
  return "0";
};

export const encodeBoard = (board: readonly Stone[]): string =>
  board.map((stone) => symbolOf(stone)).join("");

export const decodeBoard = (text: string): Stone[] =>
  [...text].map((symbol) => {
    if (symbol === "+") {
      return 1;
    }
    if (symbol === "-") {
      return -1;
    }
    if (symbol === "0") {
      return 0;
    }
    throw new RangeError(`Unknown board symbol '${symbol}'`);
  });

const emptyPosition = (size: number, toPlay: Player = 1): Position => ({
  board: emptyBoard(size),
  history: new Set<string>(),
  size,
  toPlay,
});

const neighborsOf = (point: number, size: number): number[] => {
  const row = Math.floor(point / size);
  const column = point % size;
  const adjacent: number[] = [];
  if (row > 0) {
    adjacent.push(point - size);
  }
  if (row + 1 < size) {
    adjacent.push(point + size);
  }
  if (column > 0) {
    adjacent.push(point - 1);
  }
  if (column + 1 < size) {
    adjacent.push(point + 1);
  }
  return adjacent;
};

const groupOf = (
  board: readonly Stone[],
  start: number,
  size: number
): number[] => {
  const color = board[start];
  const group = [start];
  const seen = new Set<number>([start]);
  const pending = [start];
  while (pending.length > 0) {
    const point = pending.pop();
    if (point === undefined) {
      break;
    }
    for (const neighbor of neighborsOf(point, size)) {
      if (board[neighbor] === color && !seen.has(neighbor)) {
        seen.add(neighbor);
        group.push(neighbor);
        pending.push(neighbor);
      }
    }
  }
  return group;
};

const hasLiberty = (
  board: readonly Stone[],
  group: readonly number[],
  size: number
): boolean => {
  for (const point of group) {
    for (const neighbor of neighborsOf(point, size)) {
      if (board[neighbor] === 0) {
        return true;
      }
    }
  }
  return false;
};

const opponentOf = (player: Player): Player => (player === 1 ? -1 : 1);

/** Apply one action, or throw `IllegalMoveError` when the action is illegal. */
export const applyMove = (position: Position, action: number): Position => {
  const { board, history, size, toPlay } = position;
  const passAction = passActionFor(size);
  if (action === passAction) {
    return { board, history, size, toPlay: opponentOf(toPlay) };
  }
  if (!Number.isInteger(action) || action < 0 || action > passAction) {
    throw new IllegalMoveError("Action must be a board point or pass");
  }
  if (board[action] !== 0) {
    throw new IllegalMoveError("That point is occupied");
  }

  const next = [...board];
  next[action] = toPlay;
  const opponent = opponentOf(toPlay);
  for (const neighbor of neighborsOf(action, size)) {
    if (next[neighbor] === opponent) {
      const group = groupOf(next, neighbor, size);
      if (!hasLiberty(next, group, size)) {
        for (const point of group) {
          next[point] = 0;
        }
      }
    }
  }
  const ownGroup = groupOf(next, action, size);
  if (!hasLiberty(next, ownGroup, size)) {
    for (const point of ownGroup) {
      next[point] = 0;
    }
  }

  const nextKey = encodeBoard(next);
  if (history.has(nextKey) || nextKey === encodeBoard(board)) {
    throw new IllegalMoveError("That move repeats an earlier board position");
  }
  const nextHistory = new Set([...history, encodeBoard(board)]);
  return { board: next, history: nextHistory, size, toPlay: opponent };
};

export const legalActions = (position: Position): number[] => {
  const legal = [passActionFor(position.size)];
  for (let action = 0; action < position.board.length; action += 1) {
    if (position.board[action] !== 0) {
      continue;
    }
    try {
      applyMove(position, action);
    } catch (error: unknown) {
      if (error instanceof IllegalMoveError) {
        continue;
      }
      throw error;
    }
    legal.push(action);
  }
  return legal;
};

/**
 * Area scoring: a player's stones plus every empty point that reaches only that
 * player's stones. Regions that touch both colors score for neither, and dead
 * stones stay on the board.
 */
export const areaScore = (
  board: readonly Stone[],
  size: number
): [number, number] => {
  let black = 0;
  let white = 0;
  for (const stone of board) {
    if (stone === 1) {
      black += 1;
    } else if (stone === -1) {
      white += 1;
    }
  }

  const seen = Array.from({ length: board.length }, () => false);
  for (let start = 0; start < board.length; start += 1) {
    if (board[start] !== 0 || seen[start]) {
      continue;
    }
    let region = 0;
    let touchesBlack = false;
    let touchesWhite = false;
    const pending = [start];
    seen[start] = true;
    while (pending.length > 0) {
      const point = pending.pop();
      if (point === undefined) {
        break;
      }
      region += 1;
      for (const neighbor of neighborsOf(point, size)) {
        const value = board[neighbor];
        if (value === 0) {
          if (!seen[neighbor]) {
            seen[neighbor] = true;
            pending.push(neighbor);
          }
        } else if (value === 1) {
          touchesBlack = true;
        } else {
          touchesWhite = true;
        }
      }
    }
    if (touchesBlack && !touchesWhite) {
      black += region;
    } else if (touchesWhite && !touchesBlack) {
      white += region;
    }
  }
  return [black, white];
};

const labelFor = (winner: 0 | Player, margin: number): string => {
  if (winner === 0) {
    return "Draw";
  }
  return `${winner === 1 ? "Black" : "White"} by ${margin}`;
};

export const resultOf = (position: Position): GameResult => {
  const [blackArea, whiteArea] = areaScore(position.board, position.size);
  const komi = komiFor(position.size);
  const difference = blackArea - whiteArea - komi;
  let winner: 0 | Player = 0;
  if (difference > 0) {
    winner = 1;
  } else if (difference < 0) {
    winner = -1;
  }
  const margin = Math.abs(difference);
  return {
    blackArea,
    komi,
    label: labelFor(winner, margin),
    margin,
    whiteArea,
    winner,
  };
};

export const replayMoves = (
  size: number,
  moves: readonly number[]
): Position => {
  let position = emptyPosition(size);
  for (const action of moves) {
    position = applyMove(position, action);
  }
  return position;
};

export const trailingPasses = (
  moves: readonly number[],
  size: number
): number => {
  const passAction = passActionFor(size);
  let count = 0;
  for (let index = moves.length - 1; index >= 0; index -= 1) {
    if (moves[index] !== passAction) {
      break;
    }
    count += 1;
  }
  return count;
};
