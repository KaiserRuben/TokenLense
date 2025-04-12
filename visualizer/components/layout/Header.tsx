"use client"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import ResetOnboarding from "./ResetOnboarding"
import { Github } from "lucide-react"

const navItems = [
  { href: "/", label: "Models" },
  { href: "/performance", label: "Performance" },
  { href: "/compare-selector", label: "Compare" },
  { href: "/token-importance", label: "Token Importance" },
]

export default function Header() {
  const pathname = usePathname()

  // Check if current path matches a nav item (including subpaths)
  const isActivePath = (path: string) => {
    if (path === "/") {
      return pathname === "/"
    }
    return pathname.startsWith(path)
  }

  return (
      <header className="border-b border-border sticky top-0 bg-background z-10">
        <div className="container mx-auto px-4 flex h-16 items-center justify-between">
          <div className="flex gap-6 items-center">
            <Link href="/" className="flex items-center gap-2">
              <span className="font-semibold text-xl">TokenLense</span>
            </Link>

            <nav className="hidden md:flex gap-6">
              {navItems.map((item) => (
                  <Link
                      key={item.href}
                      href={item.href}
                      className={cn(
                          "text-sm transition-colors hover:text-foreground/80",
                          isActivePath(item.href)
                              ? "text-foreground font-medium"
                              : "text-foreground/60"
                      )}
                  >
                    {item.label}
                  </Link>
              ))}
            </nav>
          </div>

          <div className="flex items-center gap-4">
            <ResetOnboarding />

            <a
                href="https://github.com/KaiserRuben/TokenLense"
                target="_blank"
                rel="noopener noreferrer"
                className="transition-colors hover:text-foreground/80 flex items-center gap-1 text-foreground/60"
            >
              <Github size={18} />
              <span className="hidden sm:inline">GitHub</span>
            </a>
          </div>
        </div>
      </header>
  )
}