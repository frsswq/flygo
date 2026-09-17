import { afterEach, describe, expect, it, vi } from "vitest";

import { playTurn, simulatePosition } from "@/lib/simulation";

const scoredTurn = {
  activity: [],
  board: Array.from({ length: 25 }, () => 0),
  computer_action: null,
  consecutive_passes: 2,
  game_over: true,
  legal_actions: [],
  moves: [25, 25],
  ruleset: "Tromp-Taylor, area scoring, positional superko",
  score: {
    black_area: 0,
    komi: 0,
    label: "Draw",
    margin: 0,
    white_area: 0,
    winner: 0,
  },
  size: 5,
  to_play: -1,
};

type FetchLike = (
  path: string,
  init: RequestInit
) => Promise<globalThis.Response>;

const respondWith = (payload: unknown, ok = true) => {
  const fetchMock = vi.fn<FetchLike>(() =>
    Promise.resolve({
      json: () => Promise.resolve(payload),
      ok,
      status: ok ? 200 : 422,
    } as unknown as globalThis.Response)
  );
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
};

describe("simulation transport", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("sends only the move list and parses a scored turn", async () => {
    const fetchMock = respondWith(scoredTurn);

    const result = await playTurn([25, 25], 5);

    expect(fetchMock).toHaveBeenCalledOnce();
    const [[path, init]] = fetchMock.mock.calls;
    expect(path).toBe("/api/play");
    expect(JSON.parse(String(init.body))).toStrictEqual({
      moves: [25, 25],
      size: 5,
    });
    expect(result.score?.label).toBe("Draw");
    expect(result.moves).toStrictEqual([25, 25]);
  });

  it("rejects a board that does not match the size", async () => {
    respondWith({ ...scoredTurn, board: [0, 0, 0] });

    await expect(playTurn([25, 25], 5)).rejects.toThrow(/Board length/u);
  });

  it("reports a failed request", async () => {
    respondWith({ detail: "That point is occupied" }, false);

    await expect(simulatePosition([0, 0], 5)).rejects.toThrow(
      "Simulation failed with status 422"
    );
  });
});
