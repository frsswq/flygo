import {
  legalActions,
  passActionFor,
  replayMoves,
  resultOf,
  trailingPasses,
} from "@/lib/go-rules";
import type { GameResult, Position, Stone } from "@/lib/go-rules";

/** Whether two consecutive passes have ended the game. */
export const isFinished = (size: number, moves: readonly number[]): boolean =>
  trailingPasses(moves, size) >= 2;

export interface BoardView {
  readonly board: readonly Stone[];
  readonly gameOver: boolean;
  readonly legal: readonly number[];
  readonly passAction: number;
  readonly position: Position;
  readonly result: GameResult;
}

/** Derive everything the board display and the move rules need from the moves. */
export const boardViewOf = (
  size: number,
  moves: readonly number[]
): BoardView => {
  const position = replayMoves(size, moves);
  return {
    board: position.board,
    gameOver: isFinished(size, moves),
    legal: legalActions(position),
    passAction: passActionFor(size),
    position,
    result: resultOf(position),
  };
};

export interface TurnGuard {
  readonly gameOver: boolean;
  readonly legal: readonly number[];
  readonly ready: boolean;
  readonly selfPlay: boolean;
  readonly thinking: boolean;
}

/** A human move is allowed only in a ready, idle, unfinished, legal position. */
export const canPlay = (guard: TurnGuard, action: number): boolean =>
  !(
    guard.selfPlay ||
    !guard.ready ||
    guard.thinking ||
    guard.gameOver ||
    !guard.legal.includes(action)
  );

export interface StatusInput {
  readonly failure: string | null;
  readonly gameOver: boolean;
  readonly moveCount: number;
  readonly ready: boolean;
  readonly result: GameResult;
  readonly searchSummary: string | null;
  readonly selfPlay: boolean;
  readonly thinking: boolean;
}

/**
 * Derive the status line.
 *
 * A failure outranks every other state, a finished game outranks the
 * in-progress text, and an unready or searching worker reads as "Thinking".
 */
export const statusOf = (state: StatusInput): string => {
  if (state.failure !== null) {
    return state.failure;
  }
  if (state.gameOver) {
    return state.result.label;
  }
  if (state.selfPlay) {
    return state.searchSummary
      ? `FlyGo vs FlyGo · move ${state.moveCount} · ${state.searchSummary}`
      : `FlyGo vs FlyGo · move ${state.moveCount + 1}`;
  }
  if (state.searchSummary) {
    return `Your turn · ${state.searchSummary}`;
  }
  if (!state.ready || state.thinking) {
    return "Thinking";
  }
  return "Your turn";
};
