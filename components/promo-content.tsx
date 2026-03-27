import React from "react";
import { cn } from "@/lib/utils";

interface PromoContentProps {
  variant?: "desktop" | "mobile";
  className?: string;
}

export function PromoContent({
  variant = "desktop",
  className,
}: PromoContentProps) {
  if (variant === "mobile") {
    return (
      <div className={cn("border-t border-border bg-muted/20 p-3", className)}>
        <div className="flex items-center gap-3">
          <div className="flex-1 min-w-0">
            <p className="text-xs font-medium text-foreground/90 truncate">
              Hanzo AI
            </p>
            <p className="text-xs text-muted-foreground truncate">
              Build with frontier AI
            </p>
          </div>
          <a
            href="https://hanzo.ai"
            className="text-xs text-primary hover:text-primary/80 font-medium"
            onClick={(e) => e.stopPropagation()}
          >
            Learn more
          </a>
        </div>
      </div>
    );
  }

  return (
    <div
      className={cn("border border-border rounded-lg p-4 bg-card", className)}
    >
      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-1">
          <h3 className="text-lg font-semibold tracking-tighter">
            Hanzo AI
          </h3>
          <p className="text-sm text-muted-foreground">
            Build with frontier AI models, infrastructure, and tools.
            Explore Zen models, MCP tools, and more.
          </p>
          <a
            href="https://hanzo.ai"
            className="text-sm text-primary hover:text-primary/80 font-medium mt-2"
          >
            Visit hanzo.ai
          </a>
        </div>
      </div>
    </div>
  );
}
