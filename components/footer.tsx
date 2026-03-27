import Image from "next/image";

export default function Footer() {
  return (
    <footer className="border-t border-border/50 px-6 py-6 mt-auto">
      <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-muted-foreground">
        <div className="flex items-center gap-2">
          <Image
            src="/zen-logo.png"
            alt="Zen LM"
            width={16}
            height={16}
            className="opacity-50"
          />
          <span>&copy; 2025 Zoo Labs Foundation &amp; Hanzo AI</span>
        </div>
        <div className="flex items-center gap-4">
          <a href="https://zenlm.org" className="hover:text-foreground transition-colors">zenlm.org</a>
          <a href="https://hanzo.ai" className="hover:text-foreground transition-colors hidden sm:block">hanzo.ai</a>
          <a href="https://zoo.ngo" className="hover:text-foreground transition-colors hidden sm:block">zoo.ngo</a>
          <a
            href="https://github.com/zenlm"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-foreground transition-colors"
          >
            GitHub
          </a>
        </div>
      </div>
    </footer>
  );
}
