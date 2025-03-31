import React, { useRef, useState, useEffect } from 'react';
import { Lock, Info } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import TokenInfoSidebar from './TokenInfoSidebar';
import {
    TokenData,
    AnalysisResult,
    Connection,
    TokenImportanceSettings,
    ProcessedTokens
} from '@/utils/types';
import { processTokens, filterTokens } from '@/utils/tokenUtils';
import { ensureDecompressedAnalysis } from '@/utils/matrixUtils';

interface TokenVisualizationProps {
    analysis: AnalysisResult;
    settings: TokenImportanceSettings;
}

const TokenVisualization: React.FC<TokenVisualizationProps> = ({ analysis, settings }) => {
    const [activeWordIndex, setActiveWordIndex] = useState<number | null>(null);
    const [lockedWordIndex, setLockedWordIndex] = useState<number | null>(null);
    const [connections, setConnections] = useState<Connection[]>([]);
    const [processedTokens, setProcessedTokens] = useState<ProcessedTokens>({
        input_tokens: [],
        output_tokens: []
    });
    const wordRefs = useRef<(HTMLSpanElement | null)[]>([]);
    const containerRef = useRef<HTMLDivElement | null>(null);

    // Process tokens on mount or when settings change
    useEffect(() => {
        // Ensure we're working with decompressed matrices
        const decompressedAnalysis = ensureDecompressedAnalysis(analysis);
        setProcessedTokens(processTokens(decompressedAnalysis, settings));
    }, [analysis, settings]);

    // Connection creation logic
    const createConnection = (startIndex: number, endIndex: number, strength: number): Connection | null => {
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

        return {
            path,
            opacity: strength,
            strokeWidth: Math.max(1, strength * 3)
        };
    };

    // Update connections when a token is selected
    const updateConnections = (index: number) => {
        const newConnections: Connection[] = [];
        const { input_tokens } = processedTokens;
        const decompressedAnalysis = ensureDecompressedAnalysis(analysis);
        const { normalized_association } = decompressedAnalysis.data;
        const isInputToken = index < input_tokens.length;

        if (isInputToken) {
            const influences = normalized_association.map(row => row[index]);
            const topInfluences = influences
                .map((value, idx) => ({value, idx}))
                .sort((a, b) => b.value - a.value)
                .slice(0, settings.maxConnections);

            for (const { value, idx } of topInfluences) {
                if (value > 0) {
                    const connection = createConnection(
                        index,
                        input_tokens.length + idx,
                        value
                    );
                    if (connection) newConnections.push(connection);
                }
            }
        } else {
            const outputIndex = index - input_tokens.length;
            if (outputIndex < 0 || outputIndex >= normalized_association.length) {
                return; // Guard against out-of-bounds access
            }

            const row = normalized_association[outputIndex];
            const contextWidth = input_tokens.length + outputIndex;
            if (!row || contextWidth > row.length) {
                return; // Guard against bad data
            }

            const contextInfluences = row.slice(0, contextWidth)
                .map((value, idx) => ({value, idx}))
                .filter(({value}) => value > 0)
                .sort((a, b) => b.value - a.value)
                .slice(0, settings.maxConnections);

            for (const { value, idx } of contextInfluences) {
                const connection = createConnection(idx, index, value);
                if (connection) newConnections.push(connection);
            }
        }

        setConnections(newConnections);
    };

    // Event handlers
    const handleWordHover = (index: number) => {
        if (lockedWordIndex !== null) return;
        setActiveWordIndex(index);
        updateConnections(index);
    };

    const handleWordClick = (index: number) => {
        if (lockedWordIndex === index) {
            setLockedWordIndex(null);
            setActiveWordIndex(null);
            setConnections([]);
        } else {
            setLockedWordIndex(index);
            setActiveWordIndex(index);
            updateConnections(index);
        }
    };

    const handleWordLeave = () => {
        if (lockedWordIndex !== null) return;
        setActiveWordIndex(null);
        setConnections([]);
    };

    // Get color based on importance
    const getImportanceColor = (importance: number): string => {
        // Only apply if above threshold
        if (importance < settings.highlightThreshold) return "";

        // Use CSS variables for theming
        return `rgba(var(--token-highlight-rgb), ${importance * 0.8})`;
    };

    // Get style for token display
    const getTokenStyle = (token: TokenData, index: number): React.CSSProperties => {
        const baseStyle: React.CSSProperties = {};

        // Apply importance background color when no active selection
        if (activeWordIndex === null) {
            baseStyle.backgroundColor = getImportanceColor(token.calculatedImportance || 0);
        }

        // When a token is selected, show its connections
        if (activeWordIndex !== null) {
            const { input_tokens } = processedTokens;
            const decompressedAnalysis = ensureDecompressedAnalysis(analysis);
            const { normalized_association } = decompressedAnalysis.data;

            // Input token selected
            if (activeWordIndex < input_tokens.length) {
                if (index >= input_tokens.length) {
                    // This is an output token, check how much it's influenced by the selected input
                    const outputIndex = index - input_tokens.length;
                    if (outputIndex < normalized_association.length) {
                        const influence = normalized_association[outputIndex][activeWordIndex];
                        if (influence > 0) {
                            baseStyle.backgroundColor = `rgba(var(--token-highlight-rgb), ${influence * 0.8})`;
                            baseStyle.opacity = 0.5 + (influence * 0.5);
                        }
                    }
                }
            }
            // Output token selected
            else {
                const activeOutputIndex = activeWordIndex - input_tokens.length;
                if (activeOutputIndex >= 0 && activeOutputIndex < normalized_association.length) {
                    const row = normalized_association[activeOutputIndex];

                    if (index < input_tokens.length) {
                        // This is an input token, check how much it influences the selected output
                        const influence = row[index];
                        if (influence > 0) {
                            baseStyle.backgroundColor = `rgba(var(--token-highlight-rgb), ${influence * 0.8})`;
                            baseStyle.opacity = 0.5 + (influence * 0.5);
                        }
                    } else if (index < activeWordIndex) {
                        // This is a previous output token
                        const outputIdx = index - input_tokens.length;
                        if (input_tokens.length + outputIdx < row.length) {
                            const influence = row[input_tokens.length + outputIdx];
                            if (influence > 0) {
                                baseStyle.backgroundColor = `rgba(var(--token-highlight-rgb), ${influence * 0.8})`;
                                baseStyle.opacity = 0.5 + (influence * 0.5);
                            }
                        }
                    }
                }
            }
        }

        // The active token should be highlighted differently
        if (index === activeWordIndex || index === lockedWordIndex) {
            baseStyle.backgroundColor = 'rgba(var(--token-active-rgb), 0.2)';
            baseStyle.boxShadow = '0 0 0 2px rgba(var(--token-active-rgb), 0.5)';
        }

        return baseStyle;
    };

    const filteredTokens = filterTokens(processedTokens, settings.showSystemTokens);
    const activeToken = activeWordIndex !== null ? (
        activeWordIndex < processedTokens.input_tokens.length
            ? processedTokens.input_tokens[activeWordIndex]
            : processedTokens.output_tokens[activeWordIndex - processedTokens.input_tokens.length]
    ) : null;

    return (
        <Card className="relative">
            <CardContent className="p-0">
                <div
                    ref={containerRef}
                    className="token-visualization-container"
                >
                    {/* Legend */}
                    <div className="absolute top-4 right-4 flex items-center gap-4 text-sm z-10">
                        <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-orange-500 dark:bg-orange-400" />
                            <span className="dark:text-gray-300">Input Tokens</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-3 h-3 rounded-full bg-blue-500 dark:bg-blue-400" />
                            <span className="dark:text-gray-300">Output Tokens</span>
                        </div>
                        <TooltipProvider>
                            <Tooltip>
                                <TooltipTrigger>
                                    <Info size={16} className="text-gray-400" />
                                </TooltipTrigger>
                                <TooltipContent>
                                    <p>Token background color shows importance</p>
                                    <p>Hover over tokens to see relationships</p>
                                    <p>Click to lock the selection</p>
                                </TooltipContent>
                            </Tooltip>
                        </TooltipProvider>
                    </div>

                    {/* Input Tokens Section */}
                    <div className="relative p-8 token-section-input">
                        <h3 className="absolute -top-3 left-4 px-2 dark:bg-gray-950 bg-white text-sm token-section-header-input">
                            Input Tokens
                        </h3>
                        <div className="flex flex-wrap gap-4">
                            {filteredTokens.input_tokens.map((token, idx) => {
                                const globalIdx = idx; // For filtered tokens we need to map back
                                const origIdx = processedTokens.input_tokens.findIndex(t => t.token_id === token.token_id);

                                return (
                                    <TooltipProvider key={`input-${idx}`}>
                                        <Tooltip>
                                            <TooltipTrigger>
                                                <div className="relative">
                          <span
                              ref={el => wordRefs.current[globalIdx] = el}
                              className="token token-input"
                              style={getTokenStyle(token, globalIdx)}
                              onMouseEnter={() => handleWordHover(origIdx)}
                              onMouseLeave={handleWordLeave}
                              onClick={() => handleWordClick(origIdx)}
                          >
                            {token.clean_token}
                          </span>
                                                    {lockedWordIndex === origIdx && (
                                                        <div className="token-indicator-input">
                                                            <Lock size={12} />
                                                        </div>
                                                    )}
                                                </div>
                                            </TooltipTrigger>
                                            <TooltipContent>
                                                <p>Importance: {(token.calculatedImportance ? (token.calculatedImportance * 100).toFixed(1) : "0")}%</p>
                                                <p>Token ID: {token.token_id}</p>
                                            </TooltipContent>
                                        </Tooltip>
                                    </TooltipProvider>
                                );
                            })}
                        </div>
                    </div>

                    {/* SVG for Connections */}
                    <svg className="absolute inset-0 pointer-events-none overflow-visible">
                        <defs className="connection-gradient-definitions">
                            <linearGradient id="connectionGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                                <stop offset="0%" stopColor="var(--connection-gradient-start)" />
                                <stop offset="100%" stopColor="var(--connection-gradient-end)" />
                            </linearGradient>
                        </defs>
                        <g className="connection-lines">
                            {connections.map((connection, index) => (
                                <path
                                    key={index}
                                    d={connection.path}
                                    className="connection-line"
                                    strokeWidth={connection.strokeWidth}
                                    strokeOpacity={connection.opacity}
                                />
                            ))}
                        </g>
                    </svg>

                    {/* Output Tokens Section */}
                    <div className="relative p-8 token-section-output">
                        <h3 className="absolute -top-3 left-4 px-2 dark:bg-gray-950 bg-white text-sm token-section-header-output">
                            Output Tokens
                        </h3>
                        <div className="flex flex-wrap gap-4">
                            {filteredTokens.output_tokens.map((token, idx) => {
                                const globalIdx = idx + processedTokens.input_tokens.length;
                                const origIdx = processedTokens.output_tokens.findIndex(t => t.token_id === token.token_id) +
                                    processedTokens.input_tokens.length;

                                return (
                                    <TooltipProvider key={`output-${idx}`}>
                                        <Tooltip>
                                            <TooltipTrigger>
                                                <div className="relative">
                          <span
                              ref={el => wordRefs.current[globalIdx] = el}
                              className="token token-output"
                              style={getTokenStyle(token, globalIdx)}
                              onMouseEnter={() => handleWordHover(origIdx)}
                              onMouseLeave={handleWordLeave}
                              onClick={() => handleWordClick(origIdx)}
                          >
                            {token.clean_token}
                          </span>
                                                    {lockedWordIndex === origIdx && (
                                                        <div className="token-indicator-output">
                                                            <Lock size={12} />
                                                        </div>
                                                    )}
                                                </div>
                                            </TooltipTrigger>
                                            <TooltipContent>
                                                <p>Importance: {(token.calculatedImportance ? (token.calculatedImportance * 100).toFixed(1) : "0")}%</p>
                                                <p>Token ID: {token.token_id}</p>
                                            </TooltipContent>
                                        </Tooltip>
                                    </TooltipProvider>
                                );
                            })}
                        </div>
                    </div>
                </div>
            </CardContent>

            {/* Information Sidebar */}
            {activeToken && (
                <TokenInfoSidebar
                    token={activeToken}
                    analysis={analysis}
                    activeIndex={activeWordIndex || 0}
                    isInputToken={activeWordIndex !== null && activeWordIndex < processedTokens.input_tokens.length}
                />
            )}
        </Card>
    );
};

export default TokenVisualization;