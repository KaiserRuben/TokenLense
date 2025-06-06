"use client"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { useState } from "react"
import { cn } from "@/lib/utils"
import ResetOnboarding from "./ResetOnboarding"
import { Github, Menu, X } from "lucide-react"
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet"

const navItems = [
  { href: "/", label: "Models" },
  { href: "/performance", label: "Performance" },
  { href: "/compare-selector", label: "Compare" },
  { href: "/token-importance", label: "Token Importance" },
]

export default function Header() {
  const pathname = usePathname()
  const [open, setOpen] = useState(false)

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
          <div className="flex gap-4 items-center">
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

          <div className="flex items-center gap-2 sm:gap-4">
            {/* Hide Reset Onboarding on mobile */}
            <div className="hidden sm:block">
              <ResetOnboarding />
            </div>

            <a
                href="https://github.com/KaiserRuben/TokenLense"
                target="_blank"
                rel="noopener noreferrer"
                className="transition-colors hover:text-foreground/80 flex items-center gap-1 text-foreground/60"
            >
              <Github size={18} />
              <span className="hidden sm:inline">GitHub</span>
            </a>

            <Sheet open={open} onOpenChange={setOpen}>
              <SheetTrigger asChild className="md:hidden">
                <button className="p-2">
                  <Menu size={20} />
                </button>
              </SheetTrigger>
              <SheetContent side="right" className="w-[250px] sm:w-[300px]">
                <div className="flex flex-col gap-6 p-6">
                  <span className="font-semibold text-lg">Menu</span>

                  <nav className="flex flex-col gap-4">
                    {navItems.map((item) => (
                        <Link
                            key={item.href}
                            href={item.href}
                            onClick={() => setOpen(false)}
                            className={cn(
                                "px-4 py-2 rounded-md transition-colors hover:bg-muted",
                                isActivePath(item.href)
                                    ? "bg-muted font-medium"
                                    : ""
                            )}
                        >
                          {item.label}
                        </Link>
                    ))}
                  </nav>

                  {/* Add Reset Onboarding inside the mobile menu */}
                  <div className="mt-4 py-2 px-4">
                    <ResetOnboarding />
                  </div>
                </div>
              </SheetContent>
            </Sheet>
          </div>
        </div>
      </header>
  )
}