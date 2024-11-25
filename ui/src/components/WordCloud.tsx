import React, {useRef, useState} from 'react';
import type {AnalysisResult} from '@/utils/data';

interface WordCloudProps {
    analysis: AnalysisResult;
    maxConnections: number;
    useRelativeStrength: boolean;
}

interface Connection {
    path: string;
    opacity: number;
    strokeWidth: number;
}

const WordCloud: React.FC<WordCloudProps> = ({analysis, maxConnections, useRelativeStrength}) => {
    const [activeWordIndex, setActiveWordIndex] = useState<number | null>(null);
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
            opacity: Math.min(0.8, normalizedStrength),
            strokeWidth: Math.max(1, normalizedStrength * 3)
        };
    };

    const handleWordHover = (index: number) => {
        if (index === activeWordIndex) return;

        const newConnections: Connection[] = [];
        const isInputToken = index < analysis.data.input_tokens.length;

        if (isInputToken) {
            // Handle input token connections
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
            // Handle output token connections
            const outputIndex = index - analysis.data.input_tokens.length;
            const row = analysis.data.normalized_association[outputIndex];

            // Get influences from input tokens
            const inputInfluences = row
                .slice(0, analysis.data.input_tokens.length)
                .map((value, idx) => ({value, idx}))
                .filter(({value}) => value > 0)
                .sort((a, b) => b.value - a.value)
                .slice(0, maxConnections);

            // Add connections from input tokens
            for (const {value, idx} of inputInfluences) {
                const connection = createConnection(idx, index, value, inputInfluences);
                if (connection) newConnections.push(connection);
            }

            // Get influences from previous output tokens
            const previousOutputInfluences: Array<{value: number; idx: number}> = [];
            for (let i = analysis.data.input_tokens.length; i < index; i++) {
                const outputIdx = i - analysis.data.input_tokens.length;
                const value = row[analysis.data.input_tokens.length + outputIdx];
                if (value > 0) {
                    previousOutputInfluences.push({value, idx: i});
                }
            }

            // Add connections from previous output tokens
            const topPreviousInfluences = previousOutputInfluences
                .sort((a, b) => b.value - a.value)
                .slice(0, maxConnections);

            for (const {value, idx} of topPreviousInfluences) {
                const connection = createConnection(idx, index, value, topPreviousInfluences);
                if (connection) newConnections.push(connection);
            }
        }

        setActiveWordIndex(index);
        setConnections(newConnections);
    };

    const getTokenStyle = (index: number): React.CSSProperties => {
        if (activeWordIndex === null) return {};

        const isInputToken = index < analysis.data.input_tokens.length;
        let strength = 0;

        // If we're hovering over an input token
        if (activeWordIndex < analysis.data.input_tokens.length) {
            // Only output tokens should light up
            if (!isInputToken) {
                const outputIndex = index - analysis.data.input_tokens.length;
                strength = analysis.data.normalized_association[outputIndex][activeWordIndex];
            }
        }
        // If we're hovering over an output token
        else {
            const activeOutputIndex = activeWordIndex - analysis.data.input_tokens.length;

            // Input tokens can influence any output token
            if (isInputToken) {
                strength = analysis.data.normalized_association[activeOutputIndex][index];
            }
            // Output tokens can only influence later tokens
            else {
                const currentOutputIndex = index - analysis.data.input_tokens.length;
                // Only show influence if this token came before the active token
                if (currentOutputIndex < activeOutputIndex) {
                    const row = analysis.data.normalized_association[activeOutputIndex];
                    strength = row[analysis.data.input_tokens.length + currentOutputIndex] || 0;
                }
            }
        }

        // Highlight the active token
        if (index === activeWordIndex) {
            return {backgroundColor: 'rgba(234, 88, 12, 0.2)'};
        }

        if (useRelativeStrength && strength > 0) {
            const row = isInputToken
                ? analysis.data.normalized_association.map(r => r[index])
                : analysis.data.normalized_association[index - analysis.data.input_tokens.length];
            const maxValue = Math.max(...row);
            strength = strength / maxValue;
        }

        return {
            backgroundColor: `rgba(234, 88, 12, ${strength * 0.1})`,
            opacity: 0.3 + (strength * 0.7) // now ranges from 0.3 to 1.0
        };
    };

    return (
        <div className="space-y-4">
            <div
                ref={containerRef}
                className="relative min-h-[400px] rounded-lg border border-gray-800/50"
            >
                {/* Input tokens section */}
                <div className="relative p-6 border-b border-gray-800/50">
                    <div className="flex flex-wrap gap-4">
                        {analysis.data.input_tokens.map((token, index) => (
                            <div key={index} className="flex flex-col items-center gap-1">
                                <span
                                    ref={el => wordRefs.current[index] = el}
                                    className="px-3 py-1.5 rounded-lg cursor-pointer transition-all duration-200"
                                    style={getTokenStyle(index)}
                                    onMouseEnter={() => handleWordHover(index)}
                                    onMouseLeave={() => {
                                        setActiveWordIndex(null);
                                        setConnections([]);
                                    }}
                                >
                                    {token.clean_token}
                                </span>
                                <div className="w-full bg-gray-200 rounded-full h-2">
                                    <div
                                        className="bg-orange-500 rounded-full h-2 transition-all duration-300"
                                        style={{
                                            width: `${tokenImportance[index] * 100}%`
                                        }}
                                    />
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Connection lines */}
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

                {/* Output tokens section */}
                <div className="relative p-6">
                    <div className="flex flex-wrap gap-4">
                        {analysis.data.output_tokens.map((token, index) => {
                            const globalIndex = index + analysis.data.input_tokens.length;
                            return (
                                <div key={index} className="flex flex-col items-center gap-1">
                                    <span
                                        ref={el => wordRefs.current[globalIndex] = el}
                                        className="px-3 py-1.5 rounded-lg cursor-pointer transition-all duration-200"
                                        style={getTokenStyle(globalIndex)}
                                        onMouseEnter={() => handleWordHover(globalIndex)}
                                        onMouseLeave={() => {
                                            setActiveWordIndex(null);
                                            setConnections([]);
                                        }}
                                    >
                                        {token.clean_token}
                                    </span>
                                    <div className="w-full bg-gray-200 rounded-full h-2">
                                        <div
                                            className="bg-orange-500 rounded-full h-2 transition-all duration-300"
                                            style={{
                                                width: `${tokenImportance[analysis.data.input_tokens.length + index] * 100}%`
                                            }}
                                        />
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default WordCloud;