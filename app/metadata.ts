import { Metadata } from "next";
import { siteConfig } from "@/lib/site";

export const metadataKeywords = [
    "Blog",
    "Zen LM",
    "Zen Models",
    "Open Source AI",
    "Language Models",
    "Machine Learning",
    "AI Research",
    "Qwen",
    "Zoo Labs Foundation",
    "Hanzo AI",
    "Open Weights",
    "Frontier Models",
    "MoE",
    "Multimodal AI",
]

export const metadata: Metadata = {
    title: siteConfig.name,
    description: siteConfig.description,
    keywords: metadataKeywords,
    authors: [
        {
            name: "Zen LM",
            url: "https://zenlm.org",
        },
    ],
    creator: "Zen LM",
    openGraph: {
        type: "website",
        locale: "en_US",
        url: siteConfig.url,
        title: siteConfig.name,
        description: siteConfig.description,
        siteName: siteConfig.name,
    },
    twitter: {
        card: "summary_large_image",
        title: siteConfig.name,
        description: siteConfig.description,
        creator: "@zenlm",
    },
    robots: {
        index: true,
        follow: true,
        googleBot: {
            index: true,
            follow: true,
            "max-video-preview": -1,
            "max-image-preview": "large",
            "max-snippet": -1,
        },
    },
};
