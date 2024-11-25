// src/components/Overview.tsx
import React from 'react';
import { LoaderCircle } from 'lucide-react';
import { useNavigate } from 'react-router';
import { useAnalysis } from '@/contexts/AnalysisContext';
import AnalysisCard from '@/components/AnalysisCard';
import { AnalysisResult } from '@/utils/data';

interface OverviewProps {
    className?: string;
}

const Overview: React.FC<OverviewProps> = ({ className = '' }) => {
    const { analyses, loading, error, setSelectedAnalysis } = useAnalysis();
    const navigate = useNavigate();

    const handleAnalysisSelect = (analysis: AnalysisResult) => {
        setSelectedAnalysis(analysis);
        const analysisId = encodeURIComponent(analysis.metadata.timestamp);
        navigate(`/analysis/${analysisId}`);
    };

    // Enhanced loading state with skeleton animation
    if (loading) {
        return (
            <div className="flex flex-col items-center justify-center py-20 dark:text-gray-400 text-gray-600">
                <LoaderCircle className="w-8 h-8 animate-spin mb-4" />
                <p className="text-lg animate-pulse">Loading analyses...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex flex-col items-center justify-center py-20">
                <p className="text-lg text-red-500 mb-2">Error loading analyses</p>
                <p className="text-sm opacity-80">{error}</p>
                <button
                    onClick={() => window.location.reload()}
                    className="mt-4 px-4 py-2 bg-red-500 text-white rounded-md hover:bg-red-600 transition-colors"
                >
                    Try Again
                </button>
            </div>
        );
    }

    return (
        <div className="min-h-screen">
            <div className={`max-w-3xl mx-auto space-y-8 ${className}`}>
                <div className="animate-fadeIn">
                    <h1 className="text-3xl font-light mb-2 dark:text-white text-gray-900">
                        Analysis Overview
                    </h1>
                    <p className="dark:text-gray-400 text-gray-600">
                        Showing {analyses.length} analysis results
                    </p>
                </div>

                <div className="space-y-6">
                    {analyses.map((analysis, index) => (
                        <div
                            key={`${analysis.metadata.timestamp}-${index}`}
                            className="animate-slideIn"
                            style={{ animationDelay: `${index * 100}ms` }}
                        >
                            <AnalysisCard
                                analysis={analysis}
                                onClick={() => handleAnalysisSelect(analysis)}
                            />
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export type { AnalysisResult, OverviewProps };
export default Overview;