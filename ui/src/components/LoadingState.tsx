// src/components/LoadingState.tsx
import React from 'react';
import { LoaderCircle } from 'lucide-react';

interface LoadingStateProps {
    message?: string;
    className?: string;
}

/**
 * LoadingState component displays a loading spinner with an optional message
 * @param message - Optional loading message to display
 * @param className - Optional additional CSS classes
 */
const LoadingState: React.FC<LoadingStateProps> = ({
                                                       message = 'Loading...',
                                                       className = ''
                                                   }) => {
    return (
        <div
            className={`
                flex flex-col items-center justify-center
                min-h-[400px] w-full
                dark:text-gray-400 text-gray-600
                ${className}
            `}
        >
            <LoaderCircle className="w-8 h-8 animate-spin mb-4" />
            <p className="text-lg">{message}</p>
        </div>
    );
};

/**
 * Inline variant for smaller loading indicators
 */
export const InlineLoadingState: React.FC<LoadingStateProps> = ({
                                                                    message = 'Loading...',
                                                                    className = ''
                                                                }) => {
    return (
        <div
            className={`
                flex items-center gap-3
                dark:text-gray-400 text-gray-600
                ${className}
            `}
        >
            <LoaderCircle className="w-4 h-4 animate-spin" />
            <span className="text-sm">{message}</span>
        </div>
    );
};

export default LoadingState;