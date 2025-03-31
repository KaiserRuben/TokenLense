// src/components/ErrorState.tsx
import React from 'react';
import { AlertCircle, ArrowLeft, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';

export interface ErrorStateProps {
    error: string;
    className?: string;
    onRetry?: () => void;
    onBack?: () => void;
    title?: string;
    subtitle?: string;
}

const ErrorState: React.FC<ErrorStateProps> = ({
                                                   error,
                                                   className = '',
                                                   onRetry,
                                                   onBack,
                                                   title = 'Something went wrong',
                                                   subtitle = 'Analysis not found',
                                               }) => {
    return (
        <div className={`relative min-h-[90vh] flex items-center justify-center p-6 ${className}`}>
            {/* Animated background shapes */}
            <div className="absolute inset-0 overflow-hidden">
                <div className="absolute w-[140vw] h-[140vw] -top-[20vh] left-1/2 -translate-x-1/2 rounded-full"
                     style={{
                         background: 'radial-gradient(circle, rgba(239, 68, 68, 0.05) 0%, transparent 70%)',
                         filter: 'blur(80px)',
                         animation: 'pulse 8s infinite'
                     }} />
                <div className="absolute w-full h-full -bottom-[30vh] -right-[20vw] rounded-full"
                     style={{
                         background: 'radial-gradient(circle, rgba(239, 68, 68, 0.03) 0%, transparent 70%)',
                         filter: 'blur(100px)',
                         animation: 'float 10s infinite'
                     }} />
            </div>

            {/* Content container */}
            <div className="relative w-full max-w-xl mx-auto">
                {/* Glass card effect */}
                <div className="backdrop-blur-xl bg-white/[0.01] dark:bg-black/[0.01] border border-white/[0.05] dark:border-white/[0.05] rounded-3xl overflow-hidden">
                    <div className="relative p-8 sm:p-12">
                        {/* Dynamic gradient overlay */}
                        <div className="absolute inset-0 bg-gradient-to-br from-red-500/[0.03] via-transparent to-red-500/[0.03] dark:from-red-500/[0.05] dark:to-red-500/[0.05]" />

                        {/* Content */}
                        <div className="relative flex flex-col items-center text-center space-y-6">
                            {/* Animated icon container */}
                            <div className="relative">
                                <div className="absolute inset-0 bg-red-500/20 dark:bg-red-500/30 blur-2xl rounded-full" />
                                <div className="relative p-4 rounded-full bg-gradient-to-br from-red-500/10 to-red-500/5 dark:from-red-500/20 dark:to-red-500/10">
                                    <AlertCircle className="w-12 h-12 text-red-500 dark:text-red-400 animate-pulse" />
                                </div>
                            </div>

                            {/* Text content */}
                            <div className="space-y-2">
                                <h2 className="text-3xl font-light tracking-tight text-gray-900 dark:text-white">
                                    {title}
                                </h2>
                                <p className="text-gray-500 dark:text-gray-400">
                                    {subtitle}
                                </p>
                                <p className="text-sm text-gray-600 dark:text-gray-300 max-w-md">
                                    {error}
                                </p>
                            </div>

                            {/* Action buttons with hover effects */}
                            <div className="flex flex-col sm:flex-row gap-4 w-full sm:w-auto mt-4">
                                {onBack && (
                                    <Button
                                        variant="ghost"
                                        onClick={onBack}
                                        className="group relative overflow-hidden backdrop-blur-sm bg-white/5 dark:bg-white/5 border border-white/10 dark:border-white/10 hover:bg-white/10 dark:hover:bg-white/10 transition-all duration-300"
                                    >
                                        <ArrowLeft className="w-5 h-5 mr-2 transition-transform duration-300 group-hover:-translate-x-1" />
                                        Go Back
                                    </Button>
                                )}
                                {onRetry && (
                                    <Button
                                        onClick={onRetry}
                                        className="group relative overflow-hidden bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 text-white transition-all duration-300"
                                    >
                                        <RefreshCw className="w-5 h-5 mr-2 transition-transform duration-500 group-hover:rotate-180" />
                                        Try Again
                                    </Button>
                                )}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Decorative elements */}
                <div className="absolute -z-10 inset-0 blur-3xl opacity-30">
                    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-32 h-32 bg-red-500/20 rounded-full mix-blend-multiply filter blur-xl animate-blob" />
                    <div className="absolute top-1/2 left-1/2 -translate-x-full -translate-y-full w-32 h-32 bg-orange-500/20 rounded-full mix-blend-multiply filter blur-xl animate-blob animation-delay-2000" />
                    <div className="absolute top-1/2 right-1/2 translate-y-full w-32 h-32 bg-pink-500/20 rounded-full mix-blend-multiply filter blur-xl animate-blob animation-delay-4000" />
                </div>
            </div>

            {/* Add required keyframes to your global CSS */}
            <style>{`
                @keyframes float {
                    0%, 100% { transform: translateY(0px); }
                    50% { transform: translateY(-20px); }
                }
                @keyframes pulse {
                    0%, 100% { opacity: 0.5; transform: scale(1); }
                    50% { opacity: 0.8; transform: scale(1.1); }
                }
                @keyframes blob {
                    0%, 100% { transform: translate(0, 0) scale(1); }
                    25% { transform: translate(20px, -20px) scale(1.1); }
                    50% { transform: translate(0, 20px) scale(1); }
                    75% { transform: translate(-20px, -20px) scale(0.9); }
                }
                .animation-delay-2000 {
                    animation-delay: 2s;
                }
                .animation-delay-4000 {
                    animation-delay: 4s;
                }
            `}</style>
        </div>
    );
};

export default ErrorState;