# zen-blog -- AI Knowledge Base

**Project**: zen-blog
**Organization**: zenlm
**Repository**: https://github.com/zenlm/zen-blog
**Domain**: https://blog.zenlm.org
**Last Updated**: 2026-03-26

## Overview

zen-blog is the Zen LM blog, built with fumadocs + Next.js (static export). Zen LM is the open model family of Zoo Labs Foundation, a 501(c)(3) non-profit.

## Architecture

- **Framework**: Next.js 15 + fumadocs-mdx 11 + fumadocs-core 15
- **Content**: MDX and Markdown files in `content/`
- **Output**: Static export (`output: "export"`)
- **Styling**: Tailwind CSS 4 + shadcn/ui components
- **Math**: remark-math + rehype-katex (KaTeX rendering)
- **Code highlighting**: Disabled (rehypeCodeOptions: false) due to Shiki/math incompatibility

## Content Format

Posts are stored as `content/{slug}.mdx` or `content/{slug}.md`:

```yaml
---
title: "Post Title"
date: "YYYY-MM-DD"
description: "Post description"
author: "Author Name"
tags: ["tag1", "tag2"]
upstream: true   # only on a post whose subject is another lab's release
---
```

44 posts came from the QwenLM blog this repository forked. They carry
`upstream: true`: their page gets `<meta name="data-upstream">` and their
article and cards carry `data-upstream`. They name the releases they are about,
by the names those releases had. A find-and-replace in b3b6eb0 once rewrote
"Qwen2"/"Qwen2.5" to "zen" in them (Qwen2-57B-A14B read "zen7B-A14B"); every
such token was restored on 2026-09-30. Never rename another lab's model in them.

MDX posts can use custom components: `<Figure />`, `<LinkButton />`, `<Video />`, `<Fullwidth>`.
Markdown posts use standard markdown only (no JSX).

Posts with bare `<` in text or complex math use `.md` extension (not `.mdx`) to avoid JSX parsing issues.

## Commands

```bash
npm run dev    # Development server
npm run build  # fumadocs-mdx → next build → scripts/check-zen.mjs out
npm start      # Serve built site
```

## Zen copy names Zen

`scripts/check-zen.mjs` reads every built page and fails the build on a name
from `zen.upstream` outside a `data-upstream` element. Upstream names belong in
two marked places only: an Architecture value (a loader string) and a License &
attribution section — plus the upstream-subject posts above.

Remote images are not probed at build time (`remarkImageOptions: { external:
false }`): one image host answering 403 used to fail the whole build.

## Deploy

blog.zenlm.org is served by the edge's `staticFiles` middleware from
`s3://hanzo-sites/zen/blog` (universe `infra/aws/routes/sites.yaml`, route
`blog-zenlm-org`), a copy of the Cloudflare Pages deployment it replaced. A build
reaches the site only when `out/` is published into that prefix. The
`gh-pages` branch and `.github/workflows/gh-pages.yml` (a Hugo build) are
leftovers from before the migration and serve nothing.

## Migration History

- **2026-03-26**: Migrated from Hugo (PaperMod theme) to fumadocs + Next.js
  - Converted 71 blog posts from Hugo format
  - Hugo shortcodes (figure, button, video, example, fullwidth) converted to MDX components or markdown
  - Chinese (.zh.md) translations not migrated (English-only for now)
  - JSON case files moved to `public/cases/`

## Rules for AI Assistants

1. **ALWAYS** update LLM.md with significant discoveries
2. **NEVER** commit model weights (*.safetensors, *.bin, *.gguf, *.pt)
3. **NEVER** commit symlinked files (CLAUDE.md, AGENTS.md, GEMINI.md, QWEN.md)
4. **NEVER** create random summary files -- update THIS file only

## Context

This file (`LLM.md`) is symlinked as CLAUDE.md, AGENTS.md, GEMINI.md, QWEN.md.

---

*Part of the Zen AI family -- Clarity Through Intelligence*
