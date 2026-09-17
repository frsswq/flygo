import { BrainCircuitIcon } from "lucide-react";
import type { CSSProperties } from "react";

import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty";
import { Spinner } from "@/components/ui/spinner";
import type { ActiveNeuron } from "@/lib/simulation";
import { cn } from "@/lib/utils";

interface ActivityMapProps {
  isLoading: boolean;
  neurons: readonly ActiveNeuron[];
}

interface DotStyle extends CSSProperties {
  "--dot-left": string;
  "--dot-size": string;
  "--dot-top": string;
}

const GOLDEN_ANGLE = 2.399963;

export const ActivityMap = ({ isLoading, neurons }: ActivityMapProps) => (
  <div className="activity-map bg-foreground relative min-h-80 overflow-hidden rounded-lg">
    {neurons.length === 0 ? (
      <div className="text-background [&_[data-slot=empty-description]]:text-background/60 absolute inset-0 flex">
        <Empty>
          <EmptyHeader>
            <EmptyMedia variant="icon">
              {isLoading ? <Spinner /> : <BrainCircuitIcon />}
            </EmptyMedia>
            <EmptyTitle>
              {isLoading ? "Waking the connectome" : "Activity unavailable"}
            </EmptyTitle>
            <EmptyDescription>
              {isLoading
                ? "Propagating the board through the frozen graph."
                : "Start a new game to try the simulation again."}
            </EmptyDescription>
          </EmptyHeader>
        </Empty>
      </div>
    ) : (
      neurons.map((neuron, index) => {
        const magnitude = Math.abs(neuron.activity);
        const radius = 7 + Math.sqrt(index) * 6.5;
        const angle = index * GOLDEN_ANGLE;
        const size = 5 + magnitude * 18;
        const dotStyle: DotStyle = {
          "--dot-left": `${50 + Math.cos(angle) * radius}%`,
          "--dot-size": `${size}px`,
          "--dot-top": `${50 + Math.sin(angle) * radius * 0.7}%`,
        };
        return (
          <span
            className={cn("activity-dot", {
              negative: neuron.activity < 0,
              positive: neuron.activity >= 0,
            })}
            key={neuron.body_id}
            style={dotStyle}
            title={`${neuron.neuron_type ?? "untyped"} · body ${neuron.body_id}`}
          />
        );
      })
    )}
  </div>
);
