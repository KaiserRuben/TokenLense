import React from 'react';
import type { WithDarkMode } from '../Layout';

interface MetricCardProps extends WithDarkMode {
    title: string;
    value?: string | number;
    change?: string;
    className?: string;
    children?: React.ReactNode;
}

const MetricCard: React.FC<MetricCardProps> = ({
                                                   title,
                                                   value,
                                                   change,
                                                   className = '',
                                                   children
                                               }) => {
    const isPositive = change?.startsWith('+');

    return (
        <div className={`card ${className}`}>
            <div className="card-gradient" />

            <div className="relative z-10 space-y-2">
                <div className="text-sm font-medium mb-3 text-content-secondary-light dark:text-content-secondary-dark">
                    {title}
                </div>

                {value && (
                    <div className="text-4xl font-light tracking-tight mb-2 text-content-primary-light dark:text-content-primary-dark">
                        {value}
                    </div>
                )}

                {change && (
                    <div className="text-sm flex items-center gap-2 text-content-muted-light dark:text-content-muted-dark">
                        <div className="w-full h-1 rounded-full bg-gray-800 overflow-hidden">
                            <div
                                className={`h-full rounded-full transition-all duration-500 ${
                                    isPositive
                                        ? 'bg-accent-positive-subtle'
                                        : 'bg-accent-negative-subtle'
                                }`}
                                style={{
                                    width: `${Math.min(Math.abs(parseFloat(change)) * 10, 100)}%`
                                }}
                            />
                        </div>
                        <span className={`font-medium min-w-[3.5rem] text-right ${
                            isPositive
                                ? 'text-accent-positive-primary'
                                : 'text-accent-negative-primary'
                        }`}>
              {change}
            </span>
                    </div>
                )}

                {children}
            </div>
        </div>
    );
};

export default MetricCard;