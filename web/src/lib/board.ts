export type Stone = -1 | 0 | 1;
export type Player = -1 | 1;

export const BOARD_POINTS = 25;
export const PASS_ACTION = BOARD_POINTS;

export const emptyBoard = (): Stone[] =>
  Array.from({ length: BOARD_POINTS }, () => 0);

export const nextStone = (stone: Stone): Stone => {
  if (stone === 0) {
    return 1;
  }
  if (stone === 1) {
    return -1;
  }
  return 0;
};

export const cyclePoint = (board: readonly Stone[], point: number): Stone[] => {
  if (!Number.isInteger(point) || point < 0 || point >= BOARD_POINTS) {
    throw new RangeError("Board point must be an integer from 0 through 24");
  }
  return board.map((stone, index) =>
    index === point ? nextStone(stone) : stone
  );
};

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
