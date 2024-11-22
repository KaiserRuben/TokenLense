import React, { useState, useEffect } from 'react';
import { Brain, Cpu, CalendarDays, LoaderCircle } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import {AnalysisResult, getAnalysisResults} from "@/utils/data.ts";
// Props interface for AnalysisCard
interface AnalysisCardProps {
    analysis: AnalysisResult;
}

const AnalysisCard: React.FC<AnalysisCardProps> = ({ analysis }) => {
    const date = new Date(analysis.metadata.timestamp);
    const formattedDate = new Intl.DateTimeFormat('en-US', {
        day: 'numeric',
        month: 'short',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
    }).format(date);

    const inputTokenCount = analysis.data.input_tokens.length;
    const outputTokenCount = analysis.data.output_tokens.length;

    return (
        <Card className="group relative overflow-hidden backdrop-blur-md
                    dark:bg-[rgba(20,20,25,0.9)] dark:border-white/8 dark:hover:border-white/15
                    bg-white/70 border-black/5 hover:border-black/10
                    transition-all duration-300">
            <div className="absolute inset-0 bg-gradient-to-br from-transparent via-white/5 to-transparent
                    opacity-0 group-hover:opacity-80 transition-opacity duration-500" />

            <CardContent className="relative space-y-6 p-6">
                {/* Query Preview */}
                <div className="space-y-4">
                    <p className="text-xl font-light tracking-tight
                      dark:text-white text-gray-900">
                        {analysis.data.input_preview}
                    </p>

                    {/* Metrics */}
                    <div className="flex items-center gap-8">
                        {/* Input Tokens */}
                        <div className="space-y-1.5">
                            <div className="flex items-center gap-2">
                                <Brain className="w-4 h-4 dark:text-gray-400 text-gray-600" />
                                <span className="text-sm font-medium dark:text-gray-400 text-gray-600">
                  Input Tokens
                </span>
                            </div>
                            <p className="text-2xl font-light pl-6 dark:text-white text-gray-900">
                                {inputTokenCount.toLocaleString()}
                            </p>
                        </div>

                        {/* Output Tokens */}
                        <div className="space-y-1.5">
                            <div className="flex items-center gap-2">
                                <Cpu className="w-4 h-4 dark:text-gray-400 text-gray-600" />
                                <span className="text-sm font-medium dark:text-gray-400 text-gray-600">
                  Output Tokens
                </span>
                            </div>
                            <p className="text-2xl font-light pl-6 dark:text-white text-gray-900">
                                {outputTokenCount.toLocaleString()}
                            </p>
                        </div>
                    </div>
                </div>

                {/* Metadata */}
                <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-4">
            <span className="font-medium dark:text-gray-300 text-gray-700">
              {analysis.metadata.llm_id}
            </span>
                        <div className="flex items-center gap-1.5 dark:text-gray-400 text-gray-600">
                            <CalendarDays className="w-4 h-4" />
                            <span>{formattedDate}</span>
                        </div>
                    </div>
                    <div className="px-2 py-0.5 rounded-full text-xs font-medium
                        dark:bg-blue-500/20 dark:text-blue-400
                        bg-blue-100 text-blue-700">
                        v{analysis.metadata.version}
                    </div>
                </div>
            </CardContent>
        </Card>
    );
};

// Props interface for AnalysisOverview
interface AnalysisOverviewProps {
    className?: string;
}

const AnalysisOverview: React.FC<AnalysisOverviewProps> = ({ className = '' }) => {
    const [analyses, setAnalyses] = useState<AnalysisResult[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchAnalyses = async () => {
            try {
                const results = await getAnalysisResults();
                setAnalyses(results);
            } catch (err) {
                setError(err instanceof Error ? err.message : 'Failed to load analyses');
            } finally {
                setLoading(false);
            }
        };

        fetchAnalyses();
    }, []);

    if (loading) {
        return (
            <div className="flex flex-col items-center justify-center py-20 dark:text-gray-400 text-gray-600">
                <LoaderCircle className="w-8 h-8 animate-spin mb-4" />
                <p className="text-lg">Loading analyses...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex flex-col items-center justify-center py-20">
                <p className="text-lg text-red-500 mb-2">Error loading analyses</p>
                <p className="text-sm opacity-80">{error}</p>
            </div>
        );
    }

    return (
        <div className={`max-w-3xl mx-auto space-y-8 ${className}`}>
            <div>
                <h1 className="text-3xl font-light mb-2 dark:text-white text-gray-900">
                    Analysis Overview
                </h1>
                <p className="dark:text-gray-400 text-gray-600">
                    Showing {analyses.length} analysis results
                </p>
            </div>

            <div className="space-y-6">
                {analyses.map((analysis, index) => (
                    <AnalysisCard
                        key={`${analysis.metadata.timestamp}-${index}`}
                        analysis={analysis}
                    />
                ))}
            </div>
        </div>
    );
};

export type { AnalysisResult, AnalysisCardProps, AnalysisOverviewProps };
export default AnalysisOverview;