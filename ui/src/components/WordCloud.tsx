import React, {useRef, useState} from 'react';
import { Lock } from 'lucide-react';
import {AnalysisResult, TokenData} from "@/utils/data.ts";

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

        // Match the background visualization style
        return {
            path,
            opacity: normalizedStrength,
            strokeWidth: Math.max(1, normalizedStrength * 2) // Scale down to 2x to make it less overwhelming
        };
    };

    const updateConnections = (index: number) => {
        const newConnections: Connection[] = [];
        const isInputToken = index < analysis.data.input_tokens.length;

        if (isInputToken) {
            // When hovering an input token, show its influence on output tokens
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
            // When hovering an output token
            const outputIndex = index - analysis.data.input_tokens.length;
            const row = analysis.data.normalized_association[outputIndex];

            // Consider all possible influences (both input and previous output tokens)
            const allInfluences = [
                // Input token influences
                ...row.slice(0, analysis.data.input_tokens.length)
                    .map((value, idx) => ({
                        value,
                        idx,
                        type: 'input' as const
                    })),
                // Previous output token influences
                ...row.slice(
                    analysis.data.input_tokens.length,
                    analysis.data.input_tokens.length + outputIndex
                ).map((value, idx) => ({
                    value,
                    idx: idx + analysis.data.input_tokens.length,
                    type: 'output' as const
                }))
            ].filter(({value}) => value > 0);

            // Sort by influence strength and take top N
            const topInfluences = allInfluences
                .sort((a, b) => b.value - a.value)
                .slice(0, maxConnections);

            // Create connections for all top influences
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
            // Unlock if clicking the locked word
            setLockedWordIndex(null);
            setActiveWordIndex(null);
            setConnections([]);
        } else {
            // Lock new word
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

    const renderToken = (token: TokenData, index: number, globalIndex: number) => (
        <div key={index} className="flex flex-col items-center gap-1">
            <div className="relative">
                <span
                    ref={el => wordRefs.current[globalIndex] = el}
                    className="px-3 py-1.5 rounded-lg cursor-pointer transition-all duration-200"
                    style={getTokenStyle(globalIndex)}
                    onMouseEnter={() => handleWordHover(globalIndex)}
                    onMouseLeave={handleWordLeave}
                    onClick={() => handleWordClick(globalIndex)}
                >
                    {token.clean_token}
                </span>
                {lockedWordIndex === globalIndex && (
                    <div className="absolute -top-2 left-1/2 -translate-x-1/2">
                        <Lock size={12} className="text-orange-600" />
                    </div>
                )}
            </div>
            {showImportanceBars && (
                <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                        className="bg-orange-500 rounded-full h-2 transition-all duration-300"
                        style={{
                            width: `${tokenImportance[globalIndex] * 100}%`
                        }}
                    />
                </div>
            )}
        </div>
    );

    return (
        <div className="space-y-4">
            <div
                ref={containerRef}
                className="relative min-h-[400px] rounded-lg border border-gray-800/50"
            >
                <div className="relative p-6 border-b border-gray-800/50">
                    <div className="flex flex-wrap gap-4">
                        {analysis.data.input_tokens.map((token, index) =>
                            renderToken(token, index, index)
                        )}
                    </div>
                </div>

                {showConnections && (
                    <svg className="absolute inset-0 pointer-events-none overflow-visible">
                        <g className="connection-lines">
                            {connections.map((connection, index) => (
                                <path
                                    key={index}
                                    d={connection.path}
                                    fill="none"
                                    stroke="rgb(234, 88, 12)"
                                    strokeWidth={connection.strokeWidth}
                                    strokeOpacity={connection.opacity}
                                    className="transition-all duration-300"
                                />
                            ))}
                        </g>
                    </svg>
                )}

                <div className="relative p-6">
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