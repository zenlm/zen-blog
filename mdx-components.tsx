import defaultMdxComponents from "fumadocs-ui/mdx";
import type { MDXComponents } from "mdx/types";
import React from "react";
import {
  MediaViewer,
  ImageViewer,
  VideoViewer,
} from "@/components/media-viewer";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { AuthorCard } from "@/components/author-card";
import { getAuthor, type AuthorKey } from "@/lib/authors";
import { CopyHeader } from "@/components/copy-header";

const createHeading = (level: number) => {
  const Heading = ({
    children,
    ...props
  }: React.HTMLAttributes<HTMLHeadingElement>) => {
    return <CopyHeader level={level} {...props}>{children}</CopyHeader>;
  };

  Heading.displayName = `Heading${level}`;
  return Heading;
};

interface AuthorProps {
  id: AuthorKey;
}

function Author({ id }: AuthorProps) {
  const author = getAuthor(id);
  return <AuthorCard author={author} className="my-8" />;
}

// Custom components for converted Hugo shortcodes

function Figure({ src, alt, width, caption }: { src: string; alt?: string; width?: string; caption?: string }) {
  const cleanSrc = src?.replace(/#center$/, "") ?? "";
  return (
    <figure className="my-6">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={cleanSrc}
        alt={alt || caption || ""}
        style={{ width: width || "100%", maxWidth: "100%", height: "auto" }}
        className="rounded-lg mx-auto block"
        loading="lazy"
      />
      {caption && (
        <figcaption className="text-center text-sm text-muted-foreground mt-2">
          {caption}
        </figcaption>
      )}
    </figure>
  );
}

function LinkButton({ href, label, external }: { href: string; label: string; external?: boolean }) {
  return (
    <a
      href={href}
      target={external ? "_blank" : undefined}
      rel={external ? "noopener noreferrer" : undefined}
      className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg border border-border bg-muted hover:bg-accent text-sm font-medium text-foreground transition-colors mr-2 mb-2 no-underline"
    >
      {label}
    </a>
  );
}

function Video({ src, alt, width, autoplay, loop, controls, muted, playsinline }: {
  src: string;
  alt?: string;
  width?: string;
  autoplay?: boolean;
  loop?: boolean;
  controls?: boolean;
  muted?: boolean;
  playsinline?: boolean;
}) {
  return (
    <div className="my-6">
      <video
        src={src}
        style={{ width: width || "100%", maxWidth: "100%" }}
        className="rounded-lg mx-auto block"
        autoPlay={autoplay}
        loop={loop}
        controls={controls !== false}
        muted={muted}
        playsInline={playsinline}
      >
        {alt || "Your browser does not support the video tag."}
      </video>
    </div>
  );
}

function Fullwidth({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`w-full my-6 ${className || ""}`}>
      {children}
    </div>
  );
}

export function getMDXComponents(components?: MDXComponents): MDXComponents {
  return {
    ...defaultMdxComponents,
    MediaViewer,
    ImageViewer,
    VideoViewer,
    Accordion,
    AccordionContent,
    AccordionItem,
    AccordionTrigger,
    Author,
    Figure,
    LinkButton,
    Video,
    Fullwidth,
    h1: createHeading(1),
    h2: createHeading(2),
    h3: createHeading(3),
    h4: createHeading(4),
    h5: createHeading(5),
    h6: createHeading(6),
    ...components,
  };
}

export const useMDXComponents = getMDXComponents;
