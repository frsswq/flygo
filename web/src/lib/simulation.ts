import { z } from "zod";

import { MAX_BOARD_SIZE, MIN_BOARD_SIZE } from "@/lib/board";

const stoneSchema = z.union([z.literal(-1), z.literal(0), z.literal(1)]);
const sizeSchema = z.number().int().min(MIN_BOARD_SIZE).max(MAX_BOARD_SIZE);
const actionSchema = z.number().int().min(0);

const activeNeuronSchema = z.object({
  activity: z.number(),
  body_id: z.number().int(),
  neuron_type: z.string().nullable(),
  superclass: z.string().nullable(),
});

const scoreSchema = z.object({
  black_area: z.number().int(),
  komi: z.number(),
  label: z.string(),
  margin: z.number(),
  white_area: z.number().int(),
  winner: z.number().int(),
});

const simulationResponseSchema = z.object({
  activity: z.array(activeNeuronSchema),
  legal_actions: z.array(actionSchema),
  model_status: z.string(),
  recommended_action: actionSchema,
  ruleset: z.string(),
  size: sizeSchema,
  to_play: z.number().int(),
  topology: z.string(),
});

const turnResponseSchema = z
  .object({
    activity: z.array(activeNeuronSchema),
    board: z.array(stoneSchema),
    computer_action: actionSchema.nullable(),
    consecutive_passes: z.number().int().min(0).max(2),
    game_over: z.boolean(),
    legal_actions: z.array(actionSchema),
    moves: z.array(actionSchema),
    ruleset: z.string(),
    score: scoreSchema.nullable(),
    size: sizeSchema,
    to_play: z.number().int(),
  })
  .refine((value) => value.board.length === value.size * value.size, {
    message: "Board length must match the board size",
  });

export type ActiveNeuron = z.infer<typeof activeNeuronSchema>;
export type Score = z.infer<typeof scoreSchema>;
export type SimulationResponse = z.infer<typeof simulationResponseSchema>;
export type TurnResponse = z.infer<typeof turnResponseSchema>;

const parseResponse = async <Response>(
  response: globalThis.Response,
  schema: z.ZodType<Response>
): Promise<Response> => {
  if (!response.ok) {
    throw new Error(`Simulation failed with status ${response.status}`);
  }

  const payload: unknown = await response.json();
  return schema.parse(payload);
};

const postJson = (path: string, body: unknown, signal?: AbortSignal) =>
  fetch(path, {
    body: JSON.stringify(body),
    headers: { "Content-Type": "application/json" },
    method: "POST",
    signal,
  });

export const simulatePosition = async (
  moves: readonly number[],
  size: number,
  signal?: AbortSignal
): Promise<SimulationResponse> => {
  const response = await postJson("/api/simulate", { moves, size }, signal);
  return parseResponse(response, simulationResponseSchema);
};

export const playTurn = async (
  moves: readonly number[],
  size: number
): Promise<TurnResponse> => {
  const response = await postJson("/api/play", { moves, size });
  return parseResponse(response, turnResponseSchema);
};
