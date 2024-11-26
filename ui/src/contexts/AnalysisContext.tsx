import React, { createContext, useContext, useState, useEffect, useMemo } from 'react';
import { AnalysisResult, getAnalysisResults } from '@/utils/data';

interface AnalysisContextState {
    analyses: AnalysisResult[];
    loading: boolean;
    error: string | null;
    selectedAnalysis: AnalysisResult | null;
    setSelectedAnalysis: (analysis: AnalysisResult | null) => void;
    refreshAnalyses: () => Promise<void>;
}

const AnalysisContext = createContext<AnalysisContextState | undefined>(undefined);

interface AnalysisProviderProps {
    children: React.ReactNode;
}

export const AnalysisProvider: React.FC<AnalysisProviderProps> = ({ children }) => {
    // State management
    const [analyses, setAnalyses] = useState<AnalysisResult[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [selectedAnalysis, setSelectedAnalysis] = useState<AnalysisResult | null>(null);

    // Fetch analyses data
    const refreshAnalyses = async () => {
        try {
            setLoading(true);
            const results = await getAnalysisResults({ offset: 0, limit: 100 });
            setAnalyses(results);
            setError(null);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load analyses');
        } finally {
            setLoading(false);
        }
    };

    // Initial load
    useEffect(() => {
        refreshAnalyses();
    }, []);

    // Memoize context value to prevent unnecessary re-renders
    const contextValue = useMemo(
        () => ({
            analyses,
            loading,
            error,
            selectedAnalysis,
            setSelectedAnalysis,
            refreshAnalyses,
        }),
        [analyses, loading, error, selectedAnalysis]
    );

    return (
        <AnalysisContext.Provider value={contextValue}>
            {children}
        </AnalysisContext.Provider>
    );
};

// Custom hook for using the analysis context
export const useAnalysis = () => {
    const context = useContext(AnalysisContext);
    if (context === undefined) {
        throw new Error('useAnalysis must be used within an AnalysisProvider');
    }
    return context;
};