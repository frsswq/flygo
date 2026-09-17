import { describe, expect, it } from "vitest";

import {
  actionLabel,
  BOARD_SIZES,
  DEFAULT_BOARD_SIZE,
  emptyBoard,
  isBoardSize,
  passActionFor,
  pointLabel,
  pointsFor,
} from "@/lib/board";

describe("Go board presentation", () => {
  it.each(BOARD_SIZES)("creates an empty %i by %i board", (size) => {
    const board = emptyBoard(size);

    expect(board).toHaveLength(pointsFor(size));
    expect(board.every((stone) => stone === 0)).toBeTruthy();
    expect(passActionFor(size)).toBe(pointsFor(size));
  });

  it("starts a new game on a nine by nine board", () => {
    expect(DEFAULT_BOARD_SIZE).toBe(9);
    expect(emptyBoard(DEFAULT_BOARD_SIZE)).toHaveLength(81);
  });

  it("creates a fresh board for each game", () => {
    expect(emptyBoard(5)).not.toBe(emptyBoard(5));
  });

  it("formats board actions and pass", () => {
    expect(actionLabel(0, 5)).toBe("Row 1, column 1");
    expect(actionLabel(24, 5)).toBe("Row 5, column 5");
    expect(actionLabel(25, 5)).toBe("Pass");
    expect(actionLabel(80, 9)).toBe("Row 9, column 9");
    expect(actionLabel(81, 9)).toBe("Pass");
  });

  it("rejects actions outside the board", () => {
    expect(() => actionLabel(26, 5)).toThrow(RangeError);
  });

  it("names points and recognises supported sizes", () => {
    expect(pointLabel(1, 2, 3)).toBe("Row 2, column 3: black");
    expect(pointLabel(-1, 2, 3)).toBe("Row 2, column 3: white");
    expect(pointLabel(0, 2, 3)).toBe("Row 2, column 3: empty");
    expect(isBoardSize(7)).toBeTruthy();
    expect(isBoardSize(4)).toBeFalsy();
  });
});
