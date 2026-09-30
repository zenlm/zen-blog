import { docs, meta } from "@/.source";
import { loader } from "fumadocs-core/source";
import { createMDXSource } from "fumadocs-mdx";
import { Suspense } from "react";
import { BlogList } from "@/components/blog-list";

const blogSource = loader({
  baseUrl: "/blog",
  source: createMDXSource(docs, meta),
});

const formatDate = (date: Date): string => {
  return date.toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
};

export default function HomePage() {
  const allPages = blogSource.getPages();
  const sortedBlogs = allPages
    .map((page) => {
      const d = page.data as {
        title: string;
        description: string;
        date: string;
        tags?: string[];
        thumbnail?: string;
        upstream?: boolean;
      };
      return {
        url: page.url,
        title: d.title,
        description: d.description,
        date: d.date,
        tags: d.tags ?? [],
        thumbnail: d.thumbnail ?? null,
        upstream: d.upstream ?? false,
        formattedDate: formatDate(new Date(d.date)),
      };
    })
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

  const allTags = [
    "All",
    ...Array.from(new Set(sortedBlogs.flatMap((b) => b.tags))).sort(),
  ];

  const tagCounts = allTags.reduce((acc, tag) => {
    acc[tag] =
      tag === "All"
        ? sortedBlogs.length
        : sortedBlogs.filter((b) => b.tags.includes(tag)).length;
    return acc;
  }, {} as Record<string, number>);

  return (
    <div className="min-h-screen bg-background">
      {/* Hero */}
      <section className="px-6 pt-16 pb-10 border-b border-border/50">
        <div className="max-w-5xl mx-auto">
          <h1 className="text-4xl sm:text-5xl font-bold tracking-tight text-foreground mb-3">
            Zen LM Blog
          </h1>
          <p className="text-lg text-muted-foreground max-w-2xl leading-relaxed">
            Research notes and releases from Zen LM, the open model family of Zoo Labs Foundation, a 501(c)(3)
            non-profit, chosen for two jobs: agentic coding that runs on your own machine, and marketing work.
          </p>
          <p className="text-base text-muted-foreground max-w-2xl leading-relaxed mt-3">
            Zen 6 and Zen 6 Flash are available now, as open weights and as <code>zen6</code> and{" "}
            <code>zen6-flash</code> on api.hanzo.ai. Zen 7 is in research preview.
          </p>
          <div className="flex flex-wrap gap-3 mt-5 text-sm">
            <a href="https://zenlm.org/models/#zen6" className="px-3 py-1.5 rounded-full border border-border text-foreground hover:bg-accent transition-all font-medium">
              Zen 6
            </a>
            <a href="https://hanzo.ai/research-access" className="px-3 py-1.5 rounded-full border border-border text-foreground hover:bg-accent transition-all font-medium">
              Request access
            </a>
          </div>
        </div>
      </section>

      {/* Posts */}
      <Suspense>
        <BlogList blogs={sortedBlogs} allTags={allTags} tagCounts={tagCounts} />
      </Suspense>
    </div>
  );
}
