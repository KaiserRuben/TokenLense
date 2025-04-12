import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import Header from "@/components/layout/Header";
import OnboardingWrapper from "@/components/layout/OnboardingWrapper";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "TokenLense - Language Model Attribution Visualization",
  description: "Advanced data visualization and analysis for language model token attribution",
  icons: {
    icon: "/favicon.ico",
  },
};

export default function RootLayout({
                                   children,
                                   }: Readonly<{
  children: React.ReactNode;
}>) {
  return (
      <html lang="en" suppressHydrationWarning>
      <body
          className={`${geistSans.variable} ${geistMono.variable} antialiased min-h-screen flex flex-col`}
      >
      <Header />
      <main className="flex-1 container mx-auto px-4 py-8">
        {children}
      </main>
      <OnboardingWrapper />
      {/*<footer className="border-t border-border py-6 mt-12">*/}
      {/*  <div className="container mx-auto px-4">*/}
      {/*    <div className="flex flex-col md:flex-row justify-between items-center gap-4">*/}
      {/*      <div className="text-sm text-muted-foreground">*/}
      {/*        &copy; {new Date().getFullYear()} TokenLense. All rights reserved.*/}
      {/*      </div>*/}
      {/*      <div className="flex gap-6">*/}
      {/*        <a href="#" className="text-sm text-muted-foreground hover:text-foreground">Documentation</a>*/}
      {/*        <a href="#" className="text-sm text-muted-foreground hover:text-foreground">GitHub</a>*/}
      {/*        <a href="#" className="text-sm text-muted-foreground hover:text-foreground">Privacy</a>*/}
      {/*      </div>*/}
      {/*    </div>*/}
      {/*  </div>*/}
      {/*</footer>*/}
      </body>
      </html>
  );
}