import React, { useRef, useState } from 'react';
import { Lock, Info } from 'lucide-react';
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from "@/components/ui/tooltip"

interface TokenData {
    clean_token: string;
    token_id: number;
}

interface AnalysisResult {
    data: {
        input_tokens: TokenData[];
        output_tokens: TokenData[];
        normalized_association: number[][];
    };
}

interface WordCloudProps {
    analysis: AnalysisResult;
    maxConnections: number;
    useRelativeStrength: boolean;
    showBackground: boolean;
    showConnections: boolean;
    showImportanceBars: boolean;
}

interface Connection {
    path: string;
    opacity: number;
    strokeWidth: number;
}

const WordCloud: React.FC<WordCloudProps> = ({
                                                 analysis,
                                                 maxConnections,
                                                 useRelativeStrength,
                                                 showBackground,
                                                 showConnections,
                                                 showImportanceBars
                                             }) => {
    const [activeWordIndex, setActiveWordIndex] = useState<number | null>(null);
    const [lockedWordIndex, setLockedWordIndex] = useState<number | null>(null);
    const [connections, setConnections] = useState<Connection[]>([]);
    const wordRefs = useRef<(HTMLSpanElement | null)[]>([]);
    const containerRef = useRef<HTMLDivElement>(null);

    const calculateImportance = () => {
        const inputImportance = analysis.data.input_tokens.map((_, idx) => {
            return analysis.data.normalized_association
                .map(row => row[idx])
                .reduce((sum, val) => sum + val, 0);
        });

        const outputImportance = analysis.data.normalized_association.map(row => {
            return row.reduce((sum, val) => sum + val, 0);
        });
        const importance_per_token = [...inputImportance, ...outputImportance];
        const max_element = Math.max(...importance_per_token)
        return importance_per_token.map(e => e / max_element)
    };

    const tokenImportance = calculateImportance();

    const createConnection = (
        startIndex: number,
        endIndex: number,
        strength: number,
        influences: Array<{value: number}>
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

    const updateConnections = (index: number) => {
        const newConnections: Connection[] = [];
        const isInputToken = index < analysis.data.input_tokens.length;

        if (isInputToken) {
            const influences = analysis.data.normalized_association.map(row => row[index]);
            const topInfluences = influences
                .map((value, idx) => ({value, idx}))
                .sort((a, b) => b.value - a.value)
                .slice(0, maxConnections);

            for (const {value, idx} of topInfluences) {
                if (value > 0) {
                    const connection = createConnection(
                        index,
                        analysis.data.input_tokens.length + idx,
                        value,
                        topInfluences
                    );
                    if (connection) newConnections.push(connection);
                }
            }
        } else {
            const outputIndex = index - analysis.data.input_tokens.length;
            const row = analysis.data.normalized_association[outputIndex];

            const allInfluences = [
                ...row.slice(0, analysis.data.input_tokens.length)
                    .map((value, idx) => ({
                        value,
                        idx,
                        type: 'input' as const
                    })),
                ...row.slice(
                    analysis.data.input_tokens.length,
                    analysis.data.input_tokens.length + outputIndex
                ).map((value, idx) => ({
                    value,
                    idx: idx + analysis.data.input_tokens.length,
                    type: 'output' as const
                }))
            ].filter(({value}) => value > 0);

            const topInfluences = allInfluences
                .sort((a, b) => b.value - a.value)
                .slice(0, maxConnections);

            for (const {value, idx} of topInfluences) {
                const connection = createConnection(idx, index, value, topInfluences);
                if (connection) newConnections.push(connection);
            }
        }

        setConnections(newConnections);
    };

    const handleWordHover = (index: number) => {
        if (lockedWordIndex !== null) return;
        if (index === activeWordIndex) return;

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

    const getTokenStyle = (index: number): React.CSSProperties => {
        if (!showBackground || activeWordIndex === null) return {};

        const isInputToken = index < analysis.data.input_tokens.length;
        let strength = 0;

        if (activeWordIndex < analysis.data.input_tokens.length) {
            if (!isInputToken) {
                const outputIndex = index - analysis.data.input_tokens.length;
                const row = analysis.data.normalized_association[outputIndex];
                const inputInfluences = row.slice(0, analysis.data.input_tokens.length);
                const maxInputInfluence = Math.max(...inputInfluences);
                strength = row[activeWordIndex] / maxInputInfluence;
            }
        } else {
            const activeOutputIndex = activeWordIndex - analysis.data.input_tokens.length;
            const row = analysis.data.normalized_association[activeOutputIndex];

            if (isInputToken || index < activeWordIndex) {
                const inputInfluences = row.slice(0, analysis.data.input_tokens.length);
                const previousOutputInfluences = row.slice(
                    analysis.data.input_tokens.length,
                    analysis.data.input_tokens.length + activeOutputIndex
                );
                const allInfluences = [...inputInfluences, ...previousOutputInfluences];
                const maxInfluence = Math.max(...allInfluences);

                if (isInputToken) {
                    strength = row[index] / maxInfluence;
                } else {
                    const currentOutputIndex = index - analysis.data.input_tokens.length;
                    strength = row[analysis.data.input_tokens.length + currentOutputIndex] / maxInfluence;
                }
            }
        }

        if (index === activeWordIndex) {
            return {};
        }

        return {
            backgroundColor: `rgba(234, 88, 12, ${strength})`,
            opacity: 0.5 + (strength * 0.5)
        };
    };

    const renderLegend = () => (
        <div className="absolute top-4 right-4 flex items-center gap-4 text-sm">
            <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-orange-500" />
                <span>Input Tokens</span>
            </div>
            <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-blue-500" />
                <span>Output Tokens</span>
            </div>
            <TooltipProvider>
                <Tooltip>
                    <TooltipTrigger>
                        <Info size={16} className="text-gray-400" />
                    </TooltipTrigger>
                    <TooltipContent>
                        <p>Hover over tokens to see relationships.</p>
                        <p>Click to lock the selection.</p>
                        <p>Connection strength shown by line thickness.</p>
                    </TooltipContent>
                </Tooltip>
            </TooltipProvider>
        </div>
    );

    const renderToken = (token: TokenData, index: number, globalIndex: number) => (
        <TooltipProvider key={index}>
            <Tooltip>
                <TooltipTrigger>
                    <div className="flex flex-col items-center gap-1 group">
                        <div className="relative">
                            <span
                                ref={el => wordRefs.current[globalIndex] = el}
                                className="px-3 py-1.5 rounded-lg cursor-pointer transition-all duration-300 ease-in-out
                                    hover:shadow-lg hover:scale-105"
                                style={getTokenStyle(globalIndex)}
                                onMouseEnter={() => handleWordHover(globalIndex)}
                                onMouseLeave={handleWordLeave}
                                onClick={() => handleWordClick(globalIndex)}
                            >
                                {token.clean_token}
                            </span>
                            {lockedWordIndex === globalIndex && (
                                <div className="absolute -top-2 left-1/2 -translate-x-1/2 animate-bounce">
                                    <Lock size={12} className="text-orange-600" />
                                </div>
                            )}
                        </div>
                        {showImportanceBars && (
                            <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-1.5">
                                <div
                                    className="bg-gray-800 dark:bg-gray-200 rounded-full h-1.5 transition-all duration-300"
                                    style={{
                                        width: `${tokenImportance[globalIndex] * 100}%`
                                    }}
                                />
                            </div>
                        )}
                    </div>
                </TooltipTrigger>
                <TooltipContent>
                    <p>Global Importance: {(tokenImportance[globalIndex] * 100).toFixed(1)}%</p>
                    <p>Token ID: {token.token_id}</p>
                </TooltipContent>
            </Tooltip>
        </TooltipProvider>
    );

    return (
        <div className="space-y-4">
            <div
                ref={containerRef}
                className="relative min-h-[400px] rounded-xl border dark:border-gray-800/50 dark:bg-gray-950/50 backdrop-blur-sm"
            >
                {renderLegend()}

                <div className="relative p-8 border-b dark:border-gray-800/50">
                    <h3 className="absolute -top-3 left-4 px-2 bg-gray-200 dark:bg-gray-950 text-sm text-orange-500">
                        Input Tokens
                    </h3>
                    <div className="flex flex-wrap gap-4">
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

                <div className="relative p-8">
                    <h3 className="absolute -top-3 left-4 px-2 bg-gray-200 dark:bg-gray-950 text-sm text-blue-500">
                        Output Tokens
                    </h3>
                    <div className="flex flex-wrap gap-4">
                        {analysis.data.output_tokens.map((token, index) =>
                            renderToken(token, index, index + analysis.data.input_tokens.length)
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default WordCloud;