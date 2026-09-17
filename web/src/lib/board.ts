export type Stone = -1 | 0 | 1;

export const BOARD_SIZES = [5, 6, 7, 8, 9] as const;
export const MIN_BOARD_SIZE = 5;
export const MAX_BOARD_SIZE = 9;
export const DEFAULT_BOARD_SIZE = 9;

export type BoardSize = (typeof BOARD_SIZES)[number];

export const isBoardSize = (value: number): value is BoardSize =>
  (BOARD_SIZES as readonly number[]).includes(value);

export const pointsFor = (size: number): number => size * size;

export const passActionFor = (size: number): number => pointsFor(size);

export const emptyBoard = (size: number): Stone[] =>
  Array.from({ length: pointsFor(size) }, () => 0 as Stone);

export const actionLabel = (action: number, size: number): string => {
  if (action === passActionFor(size)) {
    return "Pass";
  }
  if (!Number.isInteger(action) || action < 0 || action > pointsFor(size)) {
    throw new RangeError("Action must be a board point or pass");
  }
  const row = Math.floor(action / size) + 1;
  const column = (action % size) + 1;
  return `Row ${row}, column ${column}`;
};

const stoneName = (stone: Stone): string => {
  if (stone === 1) {
    return "black";
  }
  if (stone === -1) {
    return "white";
  }
  return "empty";
};

export const pointLabel = (stone: Stone, row: number, column: number): string =>
  `Row ${row}, column ${column}: ${stoneName(stone)}`;
