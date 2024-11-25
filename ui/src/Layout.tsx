import React, { useState, useEffect } from 'react';
import { Sun, Moon } from 'lucide-react';
import {AnalysisResult} from "@/utils/data.ts";

type BackgroundShapesProps = {
    isDark: boolean;
};

type ThemeToggleProps = {
    isDark: boolean;
    onToggle: () => void;
};

type DashboardLayoutProps = {
    children: React.ReactNode;
};

export interface LayoutState {
    view: 'list' | 'detail';
    selectedAnalysis: AnalysisResult | null;
}

export interface WithDarkMode {
    isDark?: boolean;
}

const BackgroundShapes: React.FC<BackgroundShapesProps> = ({ isDark }) => (
    <div className="fixed inset-0 z-0 overflow-hidden opacity-80">
        <div
            className="absolute w-[140vw] h-[140vw] -top-20 left-1/2 -translate-x-1/2"
            style={{
                background: isDark
                    ? 'radial-gradient(circle, rgba(255, 255, 255, 0.03) 0%, transparent 70%)'
                    : 'radial-gradient(circle, rgba(0, 0, 0, 0.02) 0%, transparent 70%)',
                borderRadius: '50%',
                filter: 'blur(180px)',
                mixBlendMode: 'soft-light',
            }}
        />
        <div
            className="absolute w-screen h-screen -bottom-30 -right-20"
            style={{
                background: isDark
                    ? 'radial-gradient(circle, rgba(255, 255, 255, 0.02) 0%, transparent 70%)'
                    : 'radial-gradient(circle, rgba(0, 0, 0, 0.015) 0%, transparent 70%)',
                borderRadius: '50%',
                filter: 'blur(180px)',
                mixBlendMode: 'soft-light',
            }}
        />
    </div>
);

const ThemeToggle: React.FC<ThemeToggleProps> = ({ isDark, onToggle }) => (
    <button
        onClick={onToggle}
        className="fixed top-6 right-6 z-20 p-3 rounded-xl transition-all hover:scale-110 backdrop-blur-md
                 dark:bg-surface-dark dark:text-content-primary-dark dark:hover:bg-opacity-80
                 bg-surface-light text-content-primary-light hover:bg-opacity-80"
    >
        {isDark ? <Sun size={20} /> : <Moon size={20} />}
    </button>
);

const MyLayout: React.FC<DashboardLayoutProps> = ({ children }) => {
    const [isDark, setIsDark] = useState(true);

    useEffect(() => {
        if (isDark) {
            document.documentElement.classList.add('dark');
        } else {
            document.documentElement.classList.remove('dark');
        }
    }, [isDark]);

    const childAcceptsDarkMode = (child: React.ReactElement): child is React.ReactElement<WithDarkMode> => {
        return React.isValidElement(child);
    };

    return (
        <div className="min-h-screen transition-colors duration-300
                      dark:bg-background-dark bg-background-light">
            <BackgroundShapes isDark={isDark} />

            <ThemeToggle
                isDark={isDark}
                onToggle={() => setIsDark(!isDark)}
            />

            <main className="relative pt-24 px-6 max-w-6xl mx-auto">
                {React.Children.map(children, child => {
                    if (childAcceptsDarkMode(child as React.ReactElement)) {
                        return React.cloneElement(child as React.ReactElement, { isDark });
                    }
                    return child;
                })}
            </main>
        </div>
    );
};

export default MyLayout;