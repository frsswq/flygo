import { z } from "zod";

import { MAX_BOARD_SIZE, MIN_BOARD_SIZE } from "@/lib/board";
import type { Stone } from "@/lib/board";

const stoneSchema = z.union([z.literal(-1), z.literal(0), z.literal(1)]);
const sizeSchema = z.number().int().min(MIN_BOARD_SIZE).max(MAX_BOARD_SIZE);
const actionSchema = z.number().int().min(0);

const activeNeuronSchema = z.object({
  activity: z.number(),
  body_id: z.number().int(),
  neuron_type: z.string().nullable(),
  superclass: z.string().nullable(),
});

const simulationResponseSchema = z.object({
  activity: z.array(activeNeuronSchema),
  legal_actions: z.array(actionSchema),
  model_status: z.string(),
  recommended_action: actionSchema,
  size: sizeSchema,
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
    previous_board: z.array(stoneSchema).nullable(),
    size: sizeSchema,
  })
  .refine((value) => value.board.length === value.size * value.size, {
    message: "Board length must match the board size",
  });

export type ActiveNeuron = z.infer<typeof activeNeuronSchema>;
export type SimulationResponse = z.infer<typeof simulationResponseSchema>;
export type TurnResponse = z.infer<typeof turnResponseSchema>;

interface TurnRequest {
  action: number;
  board: readonly Stone[];
  consecutivePasses: number;
  previousBoard: readonly Stone[] | null;
  size: number;
}

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

export const simulatePosition = async (
  board: readonly Stone[],
  size: number,
  signal?: AbortSignal
): Promise<SimulationResponse> => {
  const response = await fetch("/api/simulate", {
    body: JSON.stringify({ board, size, to_play: 1 }),
    headers: { "Content-Type": "application/json" },
    method: "POST",
    signal,
  });
  return parseResponse(response, simulationResponseSchema);
};

export const playTurn = async ({
  action,
  board,
  consecutivePasses,
  previousBoard,
  size,
}: TurnRequest): Promise<TurnResponse> => {
  const response = await fetch("/api/play", {
    body: JSON.stringify({
      action,
      board,
      consecutive_passes: consecutivePasses,
      previous_board: previousBoard,
      size,
    }),
    headers: { "Content-Type": "application/json" },
    method: "POST",
  });
  return parseResponse(response, turnResponseSchema);
};
