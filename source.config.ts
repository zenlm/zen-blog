import {
  defineConfig,
  defineDocs,
  frontmatterSchema,
} from "fumadocs-mdx/config";
import { z } from "zod";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

export default defineConfig({
  mdxOptions: {
    providerImportSource: "@/mdx-components",
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    rehypeCodeOptions: false,
    // Remote images render as they are. Probing each one for its size at build
    // time made the build depend on every image host a post links, and one
    // that answers 403 failed it.
    remarkImageOptions: { external: false },
  },
});

// eslint-disable-next-line @typescript-eslint/no-unused-vars
const _deps = { frontmatterSchema };

export const { docs, meta } = defineDocs({
  dir: "content",
  docs: {
    schema: frontmatterSchema.extend({
      date: z.string(),
      tags: z.array(z.string()).optional(),
      featured: z.boolean().optional().default(false),
      readTime: z.string().optional(),
      author: z.string().optional(),
      thumbnail: z.string().optional(),
      // The post's subject is another lab's release; its page and card carry data-upstream.
      upstream: z.boolean().optional().default(false),
    }),
  },
});
