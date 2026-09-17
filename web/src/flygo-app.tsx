import { GoBoard } from "@/components/go-board";
import { Button } from "@/components/ui/button";
import { useFlyGoGame } from "@/hooks/use-flygo-game";
import { BOARD_SIZES, passActionFor } from "@/lib/board";
import { cn } from "@/lib/utils";

const App = () => {
  const {
    activity,
    board,
    error,
    gameOver,
    isLoading,
    lastMove,
    legalActions,
    play,
    reset,
    selectSize,
    size,
    status,
  } = useFlyGoGame();

  const passAction = passActionFor(size);
  const canPass = !isLoading && !gameOver && legalActions.includes(passAction);

  return (
    <main className="flex min-h-svh flex-col items-center justify-center gap-4 p-4">
      <GoBoard
        activity={activity}
        board={board}
        disabled={isLoading || gameOver}
        lastMove={lastMove}
        legalActions={legalActions}
        onPlay={play}
        size={size}
      />

      <div className="grid w-full max-w-lg grid-cols-[1fr_minmax(0,auto)_1fr] items-center gap-3">
        <select
          aria-label="Board size"
          className="border-input bg-background h-8 justify-self-start rounded-md border px-2 text-sm"
          onChange={(event) => selectSize(Number(event.target.value))}
          value={size}
        >
          {BOARD_SIZES.map((option) => (
            <option key={option} value={option}>
              {option}x{option}
            </option>
          ))}
        </select>

        <p
          aria-live="polite"
          className={cn("text-muted-foreground truncate text-center text-sm", {
            "text-destructive": error !== null,
          })}
        >
          {status}
        </p>

        <div className="flex items-center gap-2 justify-self-end">
          <Button onClick={reset} size="sm" type="button" variant="ghost">
            New
          </Button>
          <Button
            disabled={!canPass}
            onClick={() => play(passAction)}
            size="sm"
            type="button"
            variant="outline"
          >
            Pass
          </Button>
        </div>
      </div>
    </main>
  );
};

export default App;
