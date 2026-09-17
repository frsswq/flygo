import { describe, expect, it } from "vitest";

import { actionLabel, emptyBoard } from "@/lib/board";

describe("Go board presentation", () => {
  it("creates an empty 5 by 5 board", () => {
    expect(emptyBoard()).toStrictEqual(Array.from({ length: 25 }, () => 0));
  });

  it("creates a fresh board for each game", () => {
    expect(emptyBoard()).not.toBe(emptyBoard());
  });

  it("formats board actions and pass", () => {
    expect(actionLabel(0)).toBe("Row 1, column 1");
    expect(actionLabel(24)).toBe("Row 5, column 5");
    expect(actionLabel(25)).toBe("Pass");
  });
});
