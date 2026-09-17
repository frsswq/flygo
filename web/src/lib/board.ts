import type { Stone } from "@/lib/go-rules";
import { passActionFor, pointsFor } from "@/lib/go-rules";

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
