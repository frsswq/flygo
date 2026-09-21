import { describe, expect, it } from "vitest";

import { statusOf } from "@/lib/game-view";
import type { GameResult } from "@/lib/go-rules";

const finished: GameResult = {
  blackArea: 20,
  komi: 0,
  label: "Black by 5",
  margin: 5,
  whiteArea: 15,
  winner: 1,
};

const playing: GameResult = {
  blackArea: 0,
  komi: 0,
  label: "Draw",
  margin: 0,
  whiteArea: 0,
  winner: 0,
};

const base = {
  failure: null,
  gameOver: false,
  moveCount: 0,
  ready: true,
  result: playing,
  searchSummary: null,
  selfPlay: false,
  thinking: false,
};

describe("game status", () => {
  it("reports the player's turn when the model is ready and idle", () => {
    expect(statusOf(base)).toBe("Your turn");
  });

  it("reports a loading model as thinking", () => {
    expect(statusOf({ ...base, ready: false })).toBe("Thinking");
    expect(statusOf({ ...base, thinking: true })).toBe("Thinking");
  });

  it("appends the last search summary to the player's turn", () => {
    expect(statusOf({ ...base, searchSummary: "8 simulations in 3 ms" })).toBe(
      "Your turn · 8 simulations in 3 ms"
    );
  });

  it("numbers self-play moves and keeps the summary", () => {
    expect(statusOf({ ...base, ready: false, selfPlay: true })).toBe(
      "FlyGo vs FlyGo · move 1"
    );
    expect(
      statusOf({
        ...base,
        moveCount: 3,
        searchSummary: "8 simulations in 3 ms",
        selfPlay: true,
      })
    ).toBe("FlyGo vs FlyGo · move 3 · 8 simulations in 3 ms");
  });

  it("prefers the game result over in-progress text", () => {
    expect(
      statusOf({
        ...base,
        gameOver: true,
        result: finished,
        searchSummary: "8 simulations in 3 ms",
        selfPlay: true,
      })
    ).toBe("Black by 5");
  });

  it("prefers a failure over every other state", () => {
    expect(
      statusOf({
        ...base,
        failure: "Bundle file graph.bin does not match its manifest sha256",
        gameOver: true,
        ready: false,
        result: finished,
        selfPlay: true,
        thinking: true,
      })
    ).toBe("Bundle file graph.bin does not match its manifest sha256");
  });
});
