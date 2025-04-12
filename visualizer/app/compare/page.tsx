"use client"

import { useState, useEffect, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import { getAttribution } from '@/lib/api';
import { convertAttributionToAnalysisResult } from '@/lib/utils';
import { AttributionResponse, AnalysisResult, AggregationMethod } from '@/lib/types';
import ComparisonControls from '@/components/comparison/ComparisonControls';
import SplitView from '@/components/comparison/SplitView';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';

// Loading component for the Suspense boundary
function ComparisonLoading() {
    return (
        <div className="space-y-8">
            <div className="flex items-center gap-2">
                <div className="flex items-center text-sm text-muted-foreground">
                    <div className="h-4 w-4 mr-1 bg-gray-200 rounded animate-pulse" />
                    <div className="h-4 w-40 bg-gray-200 rounded animate-pulse" />
                </div>
            </div>

            <div>
                <div className="h-8 w-64 bg-gray-200 rounded animate-pulse mb-2" />
                <div className="h-4 w-80 bg-gray-200 rounded animate-pulse" />
            </div>

            <div className="h-12 w-full bg-gray-200 rounded animate-pulse" />

            <div className="flex items-center justify-center h-64">
                <div className="animate-pulse text-muted-foreground">Loading comparison data...</div>
            </div>
        </div>
    );
}

// Content component that uses the searchParams
function ComparisonContent() {
    const searchParams = useSearchParams();

    // Parse parameters from URL
    const comparisonType = searchParams.get('type') || 'models'; // 'models' or 'methods' or 'both'
    const fileId = parseInt(searchParams.get('fileId') || '0', 10);

    // For model comparison
    const leftModel = searchParams.get('leftModel') || '';
    const rightModel = searchParams.get('rightModel') || '';
    const sharedMethod = searchParams.get('method') || '';

    // For method comparison
    const leftMethod = searchParams.get('leftMethod') || '';
    const rightMethod = searchParams.get('rightMethod') || '';
    const sharedModel = searchParams.get('model') || '';

    // Aggregation methods (can be different for each side)
    const leftAggregation = (searchParams.get('leftAggregation') as AggregationMethod) || 'sum';
    const rightAggregation = (searchParams.get('rightAggregation') as AggregationMethod) || 'sum';

    // State for attribution data
    const [leftAttribution, setLeftAttribution] = useState<AttributionResponse | null>(null);
    const [rightAttribution, setRightAttribution] = useState<AttributionResponse | null>(null);

    // State for processed analysis results
    const [leftAnalysis, setLeftAnalysis] = useState<AnalysisResult | null>(null);
    const [rightAnalysis, setRightAnalysis] = useState<AnalysisResult | null>(null);

    // Loading and error states
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // Shared hover state
    const [hoveredTokenIndex, setHoveredTokenIndex] = useState<number | null>(null);
    const [lockedTokenIndex, setLockedTokenIndex] = useState<number | null>(null);

    // Visualization settings for each side
    const [leftSettings, setLeftSettings] = useState({
        maxConnections: 3,
        useRelativeStrength: true,
        showBackground: true,
        showConnections: false,
    });

    const [rightSettings, setRightSettings] = useState({
        maxConnections: 3,
        useRelativeStrength: true,
        showBackground: true,
        showConnections: false,
    });

    // Load attribution data
    useEffect(() => {
        async function loadAttributionData() {
            try {
                setLoading(true);
                setError(null);

                // Determine what to fetch based on comparison type
                let leftModelToFetch = '';
                let rightModelToFetch = '';
                let leftMethodToFetch = '';
                let rightMethodToFetch = '';

                if (comparisonType === 'models') {
                    leftModelToFetch = leftModel || '';
                    rightModelToFetch = rightModel || '';
                    leftMethodToFetch = sharedMethod || '';
                    rightMethodToFetch = sharedMethod || '';
                } else if (comparisonType === 'methods') {
                    leftModelToFetch = sharedModel || '';
                    rightModelToFetch = sharedModel || '';
                    leftMethodToFetch = leftMethod || '';
                    rightMethodToFetch = rightMethod || '';
                } else if (comparisonType === 'both') {
                    leftModelToFetch = leftModel || '';
                    rightModelToFetch = rightModel || '';
                    leftMethodToFetch = leftMethod || '';
                    rightMethodToFetch = rightMethod || '';
                }

                // Verify all required parameters are non-empty strings
                if (!leftModelToFetch || !rightModelToFetch || !leftMethodToFetch || !rightMethodToFetch) {
                    setError('Missing required parameters for comparison.');
                    setLoading(false);
                    return;
                }

                // Fetch data for both sides
                const [leftData, rightData] = await Promise.all([
                    getAttribution(leftModelToFetch, leftMethodToFetch, fileId, leftAggregation),
                    getAttribution(rightModelToFetch, rightMethodToFetch, fileId, rightAggregation)
                ]);

                setLeftAttribution(leftData);
                setRightAttribution(rightData);

                // Convert to analysis results
                setLeftAnalysis(convertAttributionToAnalysisResult(leftData));
                setRightAnalysis(convertAttributionToAnalysisResult(rightData));
            } catch (err) {
                console.error('Error loading comparison data:', err);
                setError('Failed to load comparison data. Please check your parameters and try again.');
            } finally {
                setLoading(false);
            }
        }

        // Only fetch if we have the necessary params
        const canFetch =
            (comparisonType === 'models' && leftModel && rightModel && sharedMethod) ||
            (comparisonType === 'methods' && leftMethod && rightMethod && sharedModel) ||
            (comparisonType === 'both' && leftModel && rightModel && leftMethod && rightMethod);

        if (canFetch && !isNaN(fileId)) {
            loadAttributionData();
        } else {
            setError('Missing required parameters for comparison.');
            setLoading(false);
        }
    }, [
        comparisonType, fileId,
        leftModel, rightModel, sharedMethod,
        leftMethod, rightMethod, sharedModel,
        leftAggregation, rightAggregation
    ]);

    // Handle token hover/lock (shared between both sides)
    const handleTokenHover = (index: number) => {
        if (lockedTokenIndex === null) {
            setHoveredTokenIndex(index);
        }
    };

    const handleTokenClick = (index: number) => {
        if (lockedTokenIndex === index) {
            setLockedTokenIndex(null);
            setHoveredTokenIndex(null);
        } else {
            setLockedTokenIndex(index);
            setHoveredTokenIndex(index);
        }
    };

    const handleTokenLeave = () => {
        if (lockedTokenIndex === null) {
            setHoveredTokenIndex(null);
        }
    };

    return (
        <div className="space-y-8">
            <div className="flex items-center gap-2">
                <Link
                    href="/compare-selector"
                    className="flex items-center text-sm text-muted-foreground hover:text-foreground transition-colors"
                >
                    <ArrowLeft className="mr-1 h-4 w-4" />
                    Back to compare selector
                </Link>
            </div>

            <div>
                <h1 className="text-3xl font-bold">Attribution Comparison</h1>
                <p className="mt-2 text-muted-foreground">
                    Comparing {comparisonType === 'models' ? 'models' : 'methods'} for the same prompt
                </p>
            </div>

            <ComparisonControls
                comparisonType={comparisonType}
                leftSettings={leftSettings}
                rightSettings={rightSettings}
                setLeftSettings={setLeftSettings}
                setRightSettings={setRightSettings}
            />

            {loading && (
                <div className="flex items-center justify-center h-64">
                    <div className="animate-pulse text-muted-foreground">Loading comparison data...</div>
                </div>
            )}

            {error && (
                <div className="p-4 border border-red-200 bg-red-50 text-red-600 rounded-md">
                    {error}
                </div>
            )}

            {!loading && !error && leftAnalysis && rightAnalysis && (
                <SplitView
                    leftAnalysis={leftAnalysis}
                    rightAnalysis={rightAnalysis}
                    leftSettings={leftSettings}
                    rightSettings={rightSettings}
                    hoveredTokenIndex={hoveredTokenIndex}
                    lockedTokenIndex={lockedTokenIndex}
                    onTokenHover={handleTokenHover}
                    onTokenClick={handleTokenClick}
                    onTokenLeave={handleTokenLeave}
                    leftAttribution={leftAttribution}
                    rightAttribution={rightAttribution}
                    comparisonType={comparisonType}
                />
            )}
        </div>
    );
}

// Main component with Suspense boundary
export default function ComparisonPage() {
    return (
        <Suspense fallback={<ComparisonLoading />}>
            <ComparisonContent />
        </Suspense>
    );
}