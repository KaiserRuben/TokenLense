"use client"

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Eye, BarChart2, Code } from 'lucide-react';

export default function Header() {
  const pathname = usePathname();
  
  const isActive = (path: string) => {
    return pathname.startsWith(path);
  };
  
  return (
    <header className="border-b">
      <div className="container mx-auto px-4 flex items-center justify-between h-16">
        <div className="flex items-center gap-8">
          <Link href="/" className="flex items-center space-x-2">
            <Eye className="h-6 w-6" />
            <span className="text-xl font-bold">TokenLense</span>
          </Link>
          
          <nav className="hidden md:flex items-center space-x-8">
            <Link 
              href="/" 
              className={`${
                pathname === '/' ? 'text-blue-600 dark:text-blue-400' : 'text-muted-foreground'
              } hover:text-foreground transition-colors`}
            >
              Models
            </Link>
            {/*<Link */}
            {/*  href="/attribution" */}
            {/*  className={`${*/}
            {/*    isActive('/attribution') ? 'text-blue-600 dark:text-blue-400' : 'text-muted-foreground'*/}
            {/*  } hover:text-foreground transition-colors`}*/}
            {/*>*/}
            {/*  Attribution*/}
            {/*</Link>*/}
            <Link 
              href="/performance" 
              className={`${
                isActive('/performance') ? 'text-blue-600 dark:text-blue-400' : 'text-muted-foreground'
              } hover:text-foreground transition-colors flex items-center gap-1`}
            >
              <BarChart2 className="h-4 w-4" />
              <span>Performance</span>
            </Link>
          </nav>
        </div>
        
        <div className="flex items-center space-x-2">
          <Link 
            href="/performance" 
            className="p-2 rounded-full text-muted-foreground hover:text-foreground transition-colors"
            title="Performance Metrics"
          >
            <BarChart2 className="h-5 w-5" />
          </Link>
          <a 
            href="https://github.com/KaiserRuben/TokenLense"
            target="_blank"
            rel="noopener noreferrer"
            className="p-2 rounded-full text-muted-foreground hover:text-foreground transition-colors"
            title="View on GitHub"
          >
            <Code className="h-5 w-5" />
          </a>
        </div>
      </div>
    </header>
  );
}