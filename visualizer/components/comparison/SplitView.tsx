"use client"

import React from 'react';
import { AttributionResponse, AnalysisResult } from '@/lib/types';
import ComparisonTokenCloud from './ComparisonTokenCloud';

interface SplitViewProps {
    leftAnalysis: AnalysisResult;
    rightAnalysis: AnalysisResult;
    leftAttribution: AttributionResponse | null;
    rightAttribution: AttributionResponse | null;
    leftSettings: {
        maxConnections: number;
        useRelativeStrength: boolean;
        showBackground: boolean;
        showConnections: boolean;
    };
    rightSettings: {
        maxConnections: number;
        useRelativeStrength: boolean;
        showBackground: boolean;
        showConnections: boolean;
    };
    hoveredTokenIndex: number | null;
    lockedTokenIndex: number | null;
    onTokenHover: (index: number) => void;
    onTokenClick: (index: number) => void;
    onTokenLeave: () => void;
    comparisonType: string;
}

const SplitView: React.FC<SplitViewProps> = ({
                                                 leftAnalysis,
                                                 rightAnalysis,
                                                 leftAttribution,
                                                 rightAttribution,
                                                 leftSettings,
                                                 rightSettings,
                                                 hoveredTokenIndex,
                                                 lockedTokenIndex,
                                                 onTokenHover,
                                                 onTokenClick,
                                                 onTokenLeave,
                                                 comparisonType
                                             }) => {
    // Get labels for what we're comparing
    const getLeftLabel = () => {
        if (comparisonType === 'models' || comparisonType === 'both') {
            return `${leftAnalysis.metadata.llm_id}${comparisonType === 'both' ? ` (${leftAttribution?.method.replace(/_/g, ' ')})` : ''}`;
        } else {
            return leftAttribution?.method.replace(/_/g, ' ') || 'Left Method';
        }
    };

    const getRightLabel = () => {
        if (comparisonType === 'models' || comparisonType === 'both') {
            return `${rightAnalysis.metadata.llm_id}${comparisonType === 'both' ? ` (${rightAttribution?.method.replace(/_/g, ' ')})` : ''}`;
        } else {
            return rightAttribution?.method.replace(/_/g, ' ') || 'Right Method';
        }
    };

    return (
        <div className="space-y-6">
            {/* Prompt and generation - same for both sides */}
            <div className="space-y-4">
                <div className="p-4 border rounded-lg">
                    <h3 className="text-sm font-medium text-muted-foreground mb-2">Prompt</h3>
                    <p className="whitespace-pre-wrap">{leftAttribution?.prompt || ''}</p>
                </div>

                <div className="p-4 border rounded-lg">
                    <h3 className="text-sm font-medium text-muted-foreground mb-2">Generation</h3>
                    <p className="whitespace-pre-wrap">{leftAttribution?.generation || ''}</p>
                </div>
            </div>

            {/* Split view comparison */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="border rounded-lg p-4">
                    <h3 className="text-lg font-medium mb-4">{getLeftLabel()}</h3>
                    <ComparisonTokenCloud
                        analysis={leftAnalysis}
                        settings={leftSettings}
                        hoveredTokenIndex={hoveredTokenIndex}
                        lockedTokenIndex={lockedTokenIndex}
                        onTokenHover={onTokenHover}
                        onTokenClick={onTokenClick}
                        onTokenLeave={onTokenLeave}
                    />
                </div>

                <div className="border rounded-lg p-4">
                    <h3 className="text-lg font-medium mb-4">{getRightLabel()}</h3>
                    <ComparisonTokenCloud
                        analysis={rightAnalysis}
                        settings={rightSettings}
                        hoveredTokenIndex={hoveredTokenIndex}
                        lockedTokenIndex={lockedTokenIndex}
                        onTokenHover={onTokenHover}
                        onTokenClick={onTokenClick}
                        onTokenLeave={onTokenLeave}
                    />
                </div>
            </div>

            {/* Attribution info */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="p-4 border rounded-lg">
                    <h3 className="text-sm font-medium mb-2">Left Side Attribution</h3>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                        <div>
                            <p className="text-muted-foreground">Aggregation</p>
                            <p>{leftAttribution?.aggregation}</p>
                        </div>
                        <div>
                            <p className="text-muted-foreground">Matrix Size</p>
                            <p>
                                {leftAttribution?.attribution_matrix.length || 0} ×
                                {leftAttribution?.attribution_matrix[0]?.length || 0}
                            </p>
                        </div>
                    </div>
                </div>

                <div className="p-4 border rounded-lg">
                    <h3 className="text-sm font-medium mb-2">Right Side Attribution</h3>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                        <div>
                            <p className="text-muted-foreground">Aggregation</p>
                            <p>{rightAttribution?.aggregation}</p>
                        </div>
                        <div>
                            <p className="text-muted-foreground">Matrix Size</p>
                            <p>
                                {rightAttribution?.attribution_matrix.length || 0} ×
                                {rightAttribution?.attribution_matrix[0]?.length || 0}
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default SplitView;