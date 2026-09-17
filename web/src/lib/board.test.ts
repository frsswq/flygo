import { describe, expect, it } from "vitest";

import { actionLabel, pointLabel } from "@/lib/board";

describe("Go board presentation", () => {
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

  it("names points for assistive technology", () => {
    expect(pointLabel(1, 2, 3)).toBe("Row 2, column 3: black");
    expect(pointLabel(-1, 2, 3)).toBe("Row 2, column 3: white");
    expect(pointLabel(0, 2, 3)).toBe("Row 2, column 3: empty");
  });
});
