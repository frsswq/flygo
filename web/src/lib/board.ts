export type Stone = -1 | 0 | 1;

export const BOARD_POINTS = 25;
export const PASS_ACTION = BOARD_POINTS;

export const emptyBoard = (): Stone[] =>
  Array.from({ length: BOARD_POINTS }, () => 0);

export const actionLabel = (action: number): string => {
  if (action === PASS_ACTION) {
    return "Pass";
  }
  if (!Number.isInteger(action) || action < 0 || action >= BOARD_POINTS) {
    throw new RangeError("Action must be a board point or pass");
  }
  const row = Math.floor(action / 5) + 1;
  const column = (action % 5) + 1;
  return `Row ${row}, column ${column}`;
};
