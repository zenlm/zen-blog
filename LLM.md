# zen-blog -- AI Knowledge Base

**Project**: zen-blog
**Organization**: zenlm
**Repository**: https://github.com/zenlm/zen-blog
**Domain**: https://blog.zenlm.org
**Last Updated**: 2026-03-26

## Overview

zen-blog is the Zen LM blog, built with fumadocs + Next.js (static export). Deployed to Cloudflare Pages.

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
---
```

MDX posts can use custom components: `<Figure />`, `<LinkButton />`, `<Video />`, `<Fullwidth>`.
Markdown posts use standard markdown only (no JSX).

Posts with bare `<` in text or complex math use `.md` extension (not `.mdx`) to avoid JSX parsing issues.

## Commands

```bash
npm run dev    # Development server
npm run build  # Production build (static export to out/)
npm start      # Serve built site
```

## Deploy

```bash
wrangler pages deploy out --project-name zen-blog --branch main
```

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
5. Zen models are based on **Qwen3** architecture

## Context

This file (`LLM.md`) is symlinked as CLAUDE.md, AGENTS.md, GEMINI.md, QWEN.md.

---

*Part of the Zen AI family -- Clarity Through Intelligence*
