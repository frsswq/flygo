import {
  BrainCircuitIcon,
  ExternalLinkIcon,
  FlaskConicalIcon,
  SparklesIcon,
  TriangleAlertIcon,
} from "lucide-react";
import { useState } from "react";

import { ActivityMap } from "@/components/activity-map";
import { GoBoard } from "@/components/go-board";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Spinner } from "@/components/ui/spinner";
import { actionLabel, emptyBoard } from "@/lib/board";
import type { Player, Stone } from "@/lib/board";
import { simulatePosition } from "@/lib/simulation";
import type { SimulationResponse } from "@/lib/simulation";

const facts = [
  ["Topology", "Official MaleCNS v1.0"],
  ["Demo subgraph", "461 neurons · 605 edges"],
  ["Trainable parts", "Encoder + readout only"],
  ["Current policy", "Untrained demonstration"],
] as const;

const App = () => {
  const [board, setBoard] = useState<Stone[]>(emptyBoard);
  const [toPlay, setToPlay] = useState<Player>(1);
  const [result, setResult] = useState<SimulationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const clearResult = () => {
    setResult(null);
    setError(null);
  };

  const handleBoardChange = (nextBoard: Stone[]) => {
    setBoard(nextBoard);
    clearResult();
  };

  const handleReset = () => {
    setBoard(emptyBoard());
    clearResult();
  };

  const handleSimulation = async () => {
    setIsLoading(true);
    setError(null);
    try {
      setResult(await simulatePosition(board, toPlay));
    } catch (caughtError: unknown) {
      const message =
        caughtError instanceof Error
          ? caughtError.message
          : "Unknown simulation error";
      setError(message);
      setResult(null);
    }
    setIsLoading(false);
  };

  return (
    <div className="bg-background text-foreground min-h-svh">
      <header className="border-b">
        <div className="mx-auto flex h-18 max-w-7xl items-center gap-6 px-5 lg:px-8">
          <span className="text-xl font-semibold tracking-tight">
            FlyGo<span className="text-signal">.</span>
          </span>
          <Badge className="hidden sm:inline-flex" variant="secondary">
            MaleCNS v1.0 · 5×5 Go · frozen topology
          </Badge>
          <a
            className="ml-auto inline-flex items-center gap-1 text-xs font-medium tracking-wide uppercase underline underline-offset-4"
            href="https://male-cns.janelia.org/download/"
            rel="noreferrer"
            target="_blank"
          >
            Official data
            <ExternalLinkIcon data-icon="inline-end" />
          </a>
        </div>
      </header>

      <main className="mx-auto flex max-w-7xl flex-col gap-12 px-5 py-12 lg:px-8 lg:py-18">
        <section className="grid gap-6 lg:grid-cols-[0.55fr_1.45fr]">
          <p className="text-signal font-mono text-xs tracking-wide uppercase">
            A connectome transfer experiment
          </p>
          <div className="flex flex-col gap-7">
            <h1 className="max-w-4xl text-5xl leading-none font-semibold tracking-tighter sm:text-6xl lg:text-7xl">
              Can biological wiring learn a game it never evolved to play?
            </h1>
            <p className="text-muted-foreground max-w-3xl text-base leading-7 sm:text-lg">
              Edit a Go position, then send it through a frozen sample of the
              official male fruit-fly connectome. The glow shows simulated
              neuron activity, not measured biological activity.
            </p>
          </div>
        </section>

        <section className="grid gap-5 lg:grid-cols-[0.85fr_1.15fr]">
          <Card className="[--card-spacing:--spacing(6)]">
            <CardHeader>
              <Badge variant="outline">01</Badge>
              <CardTitle className="mt-2">Go position</CardTitle>
              <CardDescription>
                Build an artificial sensory input for the frozen graph.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <GoBoard
                board={board}
                onBoardChange={handleBoardChange}
                onReset={handleReset}
                onTurnChange={(player) => {
                  setToPlay(player);
                  clearResult();
                }}
                recommendedAction={result?.recommended_action ?? null}
                toPlay={toPlay}
              />
            </CardContent>
          </Card>

          <Card className="[--card-spacing:--spacing(6)]">
            <CardHeader>
              <Badge variant="outline">02</Badge>
              <CardTitle className="mt-2">Connectome activity</CardTitle>
              <CardDescription>
                Rate-coded activity after eight recurrent simulation steps.
              </CardDescription>
              <CardAction>
                <Badge variant={result ? "secondary" : "outline"}>
                  {result ? "Complete" : "Ready"}
                </Badge>
              </CardAction>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col gap-4">
                <ActivityMap neurons={result?.activity ?? []} />
                <div className="text-muted-foreground flex flex-wrap gap-5 font-mono text-xs">
                  <span className="inline-flex items-center gap-1.5">
                    <i className="bg-chart-1 size-2 rounded-full" /> Positive
                  </span>
                  <span className="inline-flex items-center gap-1.5">
                    <i className="bg-chart-2 size-2 rounded-full" /> Negative
                  </span>
                  <span>Size = |activity|</span>
                </div>
                {error ? (
                  <Alert variant="destructive">
                    <TriangleAlertIcon />
                    <AlertTitle>Simulation failed</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                ) : null}
              </div>
            </CardContent>
            <CardFooter>
              <div className="flex w-full items-end justify-between gap-4">
                <div>
                  <span className="text-muted-foreground block font-mono text-xs tracking-wide uppercase">
                    Suggested action
                  </span>
                  <strong className="text-xl">
                    {result
                      ? actionLabel(result.recommended_action)
                      : "Run the model"}
                  </strong>
                </div>
                <Button
                  disabled={isLoading}
                  onClick={handleSimulation}
                  size="lg"
                  type="button"
                >
                  {isLoading ? (
                    <Spinner data-icon="inline-start" />
                  ) : (
                    <SparklesIcon data-icon="inline-start" />
                  )}
                  {isLoading ? "Simulating" : "Simulate position"}
                </Button>
              </div>
            </CardFooter>
          </Card>
        </section>

        <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {facts.map(([label, value]) => (
            <Card key={label} size="sm">
              <CardHeader>
                <CardDescription>{label}</CardDescription>
                <CardTitle>{value}</CardTitle>
              </CardHeader>
            </Card>
          ))}
        </section>

        <Alert>
          <FlaskConicalIcon />
          <AlertTitle>Scientific boundary</AlertTitle>
          <AlertDescription>
            Janelia provides the wiring map. FlyGo defines the rate-coded
            dynamics and artificial Go inputs. This demo does not claim to
            reproduce biological neural activity.
          </AlertDescription>
        </Alert>

        <Separator />
        <footer className="text-muted-foreground flex flex-wrap justify-between gap-3 font-mono text-xs tracking-wide uppercase">
          <span className="inline-flex items-center gap-2">
            <BrainCircuitIcon /> Research scaffold · 2026
          </span>
          <span>Question first. Result second.</span>
        </footer>
      </main>
    </div>
  );
};

export default App;
