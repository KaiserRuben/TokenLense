"use client"

import React, { useRef, useState, useEffect } from 'react';
import { calculateTokenImportance, findTopInfluences } from '@/lib/utils';
import { AnalysisResult } from '@/lib/types';

interface ComparisonTokenCloudProps {
    analysis: AnalysisResult;
    settings: {
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
}

interface Connection {
    path: string;
    opacity: number;
    strokeWidth: number;
}

const ComparisonTokenCloud: React.FC<ComparisonTokenCloudProps> = ({
                                                                       analysis,
                                                                       settings,
                                                                       hoveredTokenIndex,
                                                                       lockedTokenIndex,
                                                                       onTokenHover,
                                                                       onTokenClick,
                                                                       onTokenLeave,
                                                                   }) => {
    const { maxConnections, useRelativeStrength, showBackground, showConnections } = settings;
    const [connections, setConnections] = useState<Connection[]>([]);
    const wordRefs = useRef<(HTMLSpanElement | null)[]>([]);
    const containerRef = useRef<HTMLDivElement>(null);
    const [tokenImportance, setTokenImportance] = useState<number[]>([]);

    // Calculate token importance when analysis data changes
    useEffect(() => {
        if (analysis?.data) {
            const importance = calculateTokenImportance(
                analysis.data.input_tokens,
                analysis.data.normalized_association
            );
            setTokenImportance(importance);
        }
    }, [analysis]);

    // Update connections when active token changes (from external state)
    useEffect(() => {
        const activeIndex = lockedTokenIndex !== null ? lockedTokenIndex : hoveredTokenIndex;
        if (activeIndex !== null) {
            updateConnections(activeIndex);
        } else {
            setConnections([]);
        }
    }, [hoveredTokenIndex, lockedTokenIndex, maxConnections, useRelativeStrength, showConnections]);

    // Function to create a visual connection between tokens
    const createConnection = (
        startIndex: number,
        endIndex: number,
        strength: number,
        influences: Array<{index: number, value: number}>
    ): Connection | null => {
        const startEl = wordRefs.current[startIndex];
        const endEl = wordRefs.current[endIndex];
        const container = containerRef.current;
        if (!startEl || !endEl || !container) return null;

        const containerRect = container.getBoundingClientRect();
        const startRect = startEl.getBoundingClientRect();
        const endRect = endEl.getBoundingClientRect();

        const x1 = startRect.left + startRect.width / 2 - containerRect.left;
        const y1 = startRect.top + startRect.height / 2 - containerRect.top;
        const x2 = endRect.left + endRect.width / 2 - containerRect.left;
        const y2 = endRect.top + endRect.height / 2 - containerRect.top;

        const cy = (y1 + y2) / 2;
        const cp1y = y1 + (cy - y1) * 0.5;
        const cp2y = y2 - (y2 - cy) * 0.5;

        const path = `M ${x1} ${y1} C ${x1} ${cp1y}, ${x2} ${cp2y}, ${x2} ${y2}`;

        let normalizedStrength = strength;
        if (useRelativeStrength && influences.length > 0) {
            const maxValue = Math.max(...influences.map(inf => inf.value));
            normalizedStrength = strength / maxValue;
        }

        return {
            path,
            opacity: normalizedStrength,
            strokeWidth: Math.max(1, normalizedStrength * 2)
        };
    };

    // Update connections when a token is active
    const updateConnections = (index: number) => {
        if (!analysis?.data || !showConnections) {
            setConnections([]);
            return;
        }

        const newConnections: Connection[] = [];
        const inputLength = analysis.data.input_tokens.length;

        // Find top influences for this token
        const influences = findTopInfluences(
            index,
            inputLength,
            analysis.data.normalized_association,
            maxConnections
        );

        // Create connection for each influence
        for (const {index: influenceIndex, value} of influences) {
            if (value > 0) {
                const startIndex = Math.min(index, influenceIndex);
                const endIndex = Math.max(index, influenceIndex);

                const connection = createConnection(
                    startIndex,
                    endIndex,
                    value,
                    influences
                );

                if (connection) newConnections.push(connection);
            }
        }

        setConnections(newConnections);
    };

    // Calculate background highlight color for tokens
    const getTokenStyle = (index: number): React.CSSProperties => {
        const activeIndex = lockedTokenIndex !== null ? lockedTokenIndex : hoveredTokenIndex;

        // Always highlight the active token, regardless of visualization mode
        if (index === activeIndex) {
            return {
                backgroundColor: 'rgba(234, 88, 12, 0.3)',
                boxShadow: '0 0 5px rgba(234, 88, 12, 0.5)'
            };
        }

        // If a token is active AND we're in background mode, show relationship strength
        if (showBackground && activeIndex !== null && analysis?.data) {
            const inputLength = analysis.data.input_tokens.length;
            const isInputToken = index < inputLength;
            let strength = 0;

            if (activeIndex < inputLength) {
                if (!isInputToken) {
                    const outputIndex = index - inputLength;
                    if (outputIndex < analysis.data.normalized_association.length) {
                        const row = analysis.data.normalized_association[outputIndex];
                        strength = row[activeIndex] || 0;
                    }
                }
            } else {
                const activeOutputIndex = activeIndex - inputLength;
                if (activeOutputIndex < analysis.data.normalized_association.length) {
                    const row = analysis.data.normalized_association[activeOutputIndex];

                    if (isInputToken && index < row.length) {
                        strength = row[index] || 0;
                    } else if (index < activeIndex && index - inputLength < row.length) {
                        strength = row[index - inputLength] || 0;
                    }
                }
            }

            return {
                backgroundColor: `rgba(234, 88, 12, ${strength * 0.7})`
            };
        }
        // If no token is hovered OR we're in connection mode, show the general importance
        else if (tokenImportance[index] !== undefined) {
            const importance = tokenImportance[index];
            return {
                backgroundColor: `rgba(234, 88, 12, ${importance * 0.5})`
            };
        }

        return {};
    };

    // Render a legend for the visualization
    const renderLegend = () => (
        <div className="absolute top-2 right-2 flex items-center gap-2 text-xs">
            <div className="flex items-center gap-1">
                <div className="w-2 h-2 rounded-full bg-orange-500" />
                <span>Input</span>
            </div>
            <div className="flex items-center gap-1">
                <div className="w-2 h-2 rounded-full bg-blue-500" />
                <span>Output</span>
            </div>
        </div>
    );

    // Render a single token with its hover effects
    const renderToken = (token: { clean_token: string, token_id: number }, index: number, globalIndex: number) => (
        <div key={index} className="group">
            <div className="flex flex-col items-center">
                <div className="relative">
                    {/* @ts-expect-error */}
                    <span ref={el => wordRefs.current[globalIndex] = el}
                        className="px-2 py-1 rounded-lg cursor-pointer transition-all duration-300 ease-in-out
                     hover:shadow-lg hover:scale-105 text-sm"
                        style={getTokenStyle(globalIndex)}
                        onMouseEnter={() => onTokenHover(globalIndex)}
                        onMouseLeave={onTokenLeave}
                        onClick={() => onTokenClick(globalIndex)}
                        title={`Token: ${token.clean_token}, ID: ${token.token_id}`}
                    >
            {token.clean_token || '<empty>'}
          </span>
                </div>
            </div>
        </div>
    );

    if (!analysis?.data) {
        return <div className="min-h-[300px] flex items-center justify-center">No analysis data available</div>;
    }

    return (
        <div className="space-y-4">
            <div
                ref={containerRef}
                className="relative min-h-[300px] rounded-xl border backdrop-blur-sm"
            >
                {renderLegend()}

                <div className="relative p-4 border-b">
                    <h3 className="absolute -top-3 left-4 px-2 bg-background text-xs text-orange-500">
                        Input Tokens
                    </h3>
                    <div className="flex flex-wrap gap-2 mt-2">
                        {analysis.data.input_tokens.map((token, index) =>
                            renderToken(token, index, index)
                        )}
                    </div>
                </div>

                {showConnections && (
                    <svg className="absolute inset-0 pointer-events-none overflow-visible">
                        <defs>
                            <linearGradient id="connectionGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                                <stop offset="0%" stopColor="rgb(234, 88, 12)" />
                                <stop offset="100%" stopColor="rgb(59, 130, 246)" />
                            </linearGradient>
                        </defs>
                        <g className="connection-lines">
                            {connections.map((connection, index) => (
                                <path
                                    key={index}
                                    d={connection.path}
                                    fill="none"
                                    stroke="url(#connectionGradient)"
                                    strokeWidth={connection.strokeWidth}
                                    strokeOpacity={connection.opacity}
                                    className="transition-all duration-300"
                                />
                            ))}
                        </g>
                    </svg>
                )}

                <div className="relative p-4">
                    <h3 className="absolute -top-3 left-4 px-2 bg-background text-xs text-blue-500">
                        Output Tokens
                    </h3>
                    <div className="flex flex-wrap gap-2">
                        {analysis.data.output_tokens.map((token, index) =>
                            renderToken(token, index, index + analysis.data.input_tokens.length)
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ComparisonTokenCloud;