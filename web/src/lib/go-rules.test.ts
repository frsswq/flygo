import { describe, expect, it } from "vitest";
import { z } from "zod";

import type { Player, Position } from "@/lib/go-rules";
import {
  applyMove,
  areaScore,
  BOARD_SIZES,
  DEFAULT_BOARD_SIZE,
  decodeBoard,
  emptyBoard,
  encodeBoard,
  IllegalMoveError,
  komiFor,
  legalActions,
  MAX_BOARD_SIZE,
  MIN_BOARD_SIZE,
  passActionFor,
  pointsFor,
  replayMoves,
  resultOf,
  RULESET,
  trailingPasses,
} from "@/lib/go-rules";

import fixtureJson from "../../../shared/rules-conformance.json";

const caseSchema = z.object({
  black_area: z.number().int(),
  board: z.string(),
  final_board: z.string(),
  final_to_play: z.number().int(),
  history: z.array(z.string()),
  komi: z.number(),
  label: z.string(),
  legal_actions: z.array(z.number().int()),
  margin: z.number(),
  moves: z.array(z.number().int()),
  name: z.string(),
  size: z.number().int().min(MIN_BOARD_SIZE).max(MAX_BOARD_SIZE),
  to_play: z.number().int(),
  white_area: z.number().int(),
  winner: z.number().int(),
});

const illegalSchema = z.object({
  board: z.string(),
  error: z.string(),
  history: z.array(z.string()),
  moves: z.array(z.number().int()),
  name: z.string(),
  size: z.number().int().min(MIN_BOARD_SIZE).max(MAX_BOARD_SIZE),
  to_play: z.number().int(),
});

const fixtureSchema = z.object({
  cases: z.array(caseSchema),
  illegal: z.array(illegalSchema),
  komi: z.record(z.string(), z.number()),
  ruleset: z.string(),
  sizes: z.array(z.number().int()),
});

const fixture = fixtureSchema.parse(fixtureJson);

const playerOf = (value: number): Player => (value === 1 ? 1 : -1);

interface BoardCase {
  board: string;
  history: string[];
  size: number;
  to_play: number;
}

const positionOf = (testCase: BoardCase): Position => ({
  board: decodeBoard(testCase.board),
  history: new Set(testCase.history),
  size: testCase.size,
  toPlay: playerOf(testCase.to_play),
});

const replay = (testCase: BoardCase, moves: number[]): Position => {
  let position = positionOf(testCase);
  for (const move of moves) {
    position = applyMove(position, move);
  }
  return position;
};

const sorted = (actions: number[]): number[] =>
  actions.toSorted((a, b) => a - b);

describe("rules conformance with the Python engine", () => {
  it("uses the same ruleset and komi table", () => {
    expect(fixture.ruleset).toBe(RULESET);
    expect(fixture.sizes).toStrictEqual([...BOARD_SIZES]);
    for (const size of BOARD_SIZES) {
      expect(fixture.komi[String(size)]).toBe(komiFor(size));
    }
  });

  it.each(fixture.cases.map((testCase) => [testCase.name, testCase] as const))(
    "agrees on %s",
    (_name, testCase) => {
      const position = replay(testCase, testCase.moves);

      expect(encodeBoard(position.board)).toBe(testCase.final_board);
      expect(position.toPlay).toBe(testCase.final_to_play);
      expect(sorted(legalActions(position))).toStrictEqual(
        testCase.legal_actions
      );
      expect(resultOf(position)).toStrictEqual({
        blackArea: testCase.black_area,
        komi: testCase.komi,
        label: testCase.label,
        margin: testCase.margin,
        whiteArea: testCase.white_area,
        winner: testCase.winner,
      });
    }
  );

  it.each(
    fixture.illegal.map((testCase) => [testCase.name, testCase] as const)
  )("rejects %s", (_name, testCase) => {
    expect(() => replay(testCase, testCase.moves)).toThrow(testCase.error);
  });
});

describe("Go rules", () => {
  it.each(BOARD_SIZES)("builds an empty %i by %i board", (size) => {
    const board = emptyBoard(size);

    expect(board).toHaveLength(pointsFor(size));
    expect(board.every((stone) => stone === 0)).toBeTruthy();
    expect(passActionFor(size)).toBe(pointsFor(size));
  });

  it("starts a new game on a nineteen by nineteen board", () => {
    expect(DEFAULT_BOARD_SIZE).toBe(19);
    expect(emptyBoard(DEFAULT_BOARD_SIZE)).toHaveLength(361);
  });

  it("defines komi for every supported size", () => {
    expect(komiFor(5)).toBe(0);
    expect(komiFor(19)).toBe(7.5);
    expect(() => komiFor(9)).toThrow(/komi/u);
  });

  it("round trips a board through its encoded form", () => {
    const board = emptyBoard(5);
    board[0] = 1;
    board[6] = -1;

    expect(decodeBoard(encodeBoard(board))).toStrictEqual(board);
    expect(() => decodeBoard("x")).toThrow(/Unknown board symbol/u);
  });

  it("scores an empty board as neutral", () => {
    expect(areaScore(emptyBoard(5), 5)).toStrictEqual([0, 0]);
  });

  it("scores a full board for one color", () => {
    expect(areaScore(emptyBoard(5).fill(1), 5)).toStrictEqual([25, 0]);
  });

  it("splits a board between colors", () => {
    const board = emptyBoard(5);
    for (let row = 0; row < 5; row += 1) {
      board[row * 5] = 1;
      board[row * 5 + 4] = -1;
    }

    expect(areaScore(board, 5)).toStrictEqual([5, 5]);
  });

  it("counts an enclosed empty point for the surrounding color", () => {
    const board = emptyBoard(5);
    for (const index of [2, 6, 8, 12]) {
      board[index] = 1;
    }
    for (const index of [0, 3, 4, 5, 9, 10, 13, 14, 15, 19, 20, 24]) {
      board[index] = -1;
    }

    expect(areaScore(board, 5)).toStrictEqual([5, 12]);
  });

  it("reports a draw on an empty five by five board", () => {
    const result = resultOf(replayMoves(5, []));

    expect(result.winner).toBe(0);
    expect(result.label).toBe("Draw");
  });

  it("gives white the komi on an empty nineteen by nineteen board", () => {
    const result = resultOf(replayMoves(19, []));

    expect(result.winner).toBe(-1);
    expect(result.margin).toBe(7.5);
    expect(result.label).toBe("White by 7.5");
  });

  it("counts trailing passes", () => {
    expect(trailingPasses([], 5)).toBe(0);
    expect(trailingPasses([0], 5)).toBe(0);
    expect(trailingPasses([25], 5)).toBe(1);
    expect(trailingPasses([0, 25, 25], 5)).toBe(2);
  });

  it("rejects an occupied point", () => {
    const position = replayMoves(5, [0]);

    expect(() => applyMove(position, 0)).toThrow(IllegalMoveError);
    expect(() => applyMove(position, 0)).toThrow("That point is occupied");
  });
});
