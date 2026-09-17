import { RotateCcwIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { cyclePoint } from "@/lib/board";
import type { Player, Stone } from "@/lib/board";
import { cn } from "@/lib/utils";

interface GoBoardProps {
  board: readonly Stone[];
  onBoardChange: (board: Stone[]) => void;
  onReset: () => void;
  onTurnChange: (player: Player) => void;
  recommendedAction: number | null;
  toPlay: Player;
}

const stoneName = (stone: Stone): string => {
  if (stone === 1) {
    return "black";
  }
  if (stone === -1) {
    return "white";
  }
  return "empty";
};

export const GoBoard = ({
  board,
  onBoardChange,
  onReset,
  onTurnChange,
  recommendedAction,
  toPlay,
}: GoBoardProps) => (
  <div className="flex flex-col gap-6">
    <div className="flex items-center justify-between gap-4">
      <p className="text-muted-foreground text-sm">
        Click a point to cycle black, white, and empty.
      </p>
      <Button onClick={onReset} size="sm" type="button" variant="ghost">
        <RotateCcwIcon data-icon="inline-start" />
        Clear
      </Button>
    </div>

    <div className="go-board mx-auto grid aspect-square w-full max-w-96 min-w-0 grid-cols-5">
      {board.map((stone, point) => {
        const row = Math.floor(point / 5) + 1;
        const column = (point % 5) + 1;
        const pointName = stoneName(stone);
        return (
          <button
            aria-label={`Row ${row}, column ${column}: ${pointName}`}
            className={cn("go-point", pointName, {
              recommended: recommendedAction === point,
            })}
            key={`${row}-${column}`}
            onClick={() => onBoardChange(cyclePoint(board, point))}
            type="button"
          />
        );
      })}
    </div>

    <div className="flex flex-wrap items-center justify-between gap-3">
      <span className="text-muted-foreground text-xs font-medium tracking-wide uppercase">
        Player to move
      </span>
      <ToggleGroup
        aria-label="Player to move"
        onValueChange={(value) => {
          const [player] = value;
          if (player === "black") {
            onTurnChange(1);
          }
          if (player === "white") {
            onTurnChange(-1);
          }
        }}
        value={[toPlay === 1 ? "black" : "white"]}
        variant="outline"
      >
        <ToggleGroupItem aria-label="Black to move" value="black">
          <span className="stone-swatch bg-foreground" />
          Black
        </ToggleGroupItem>
        <ToggleGroupItem aria-label="White to move" value="white">
          <span className="stone-swatch bg-background border" />
          White
        </ToggleGroupItem>
      </ToggleGroup>
    </div>
  </div>
);
