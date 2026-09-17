import { describe, expect, it } from "vitest";

import { actionLabel, cyclePoint, emptyBoard } from "@/lib/board";

describe("board position editor", () => {
  it("cycles a point through black, white, and empty", () => {
    const black = cyclePoint(emptyBoard(), 6);
    const white = cyclePoint(black, 6);
    const empty = cyclePoint(white, 6);

    expect(black[6]).toBe(1);
    expect(white[6]).toBe(-1);
    expect(empty[6]).toBe(0);
  });

  it("formats board actions and pass", () => {
    expect(actionLabel(0)).toBe("Row 1, column 1");
    expect(actionLabel(24)).toBe("Row 5, column 5");
    expect(actionLabel(25)).toBe("Pass");
  });

  it("rejects points outside the board", () => {
    expect(() => cyclePoint(emptyBoard(), 25)).toThrow(RangeError);
  });
});
