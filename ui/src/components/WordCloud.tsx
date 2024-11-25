import React, {useRef, useState} from 'react';
import { Lock } from 'lucide-react';
import {AnalysisResult, TokenData} from "@/utils/data.ts";

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
            opacity: Math.min(0.8, normalizedStrength),
            strokeWidth: Math.max(1, normalizedStrength * 3)
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

            const inputInfluences = row
                .slice(0, analysis.data.input_tokens.length)
                .map((value, idx) => ({value, idx}))
                .filter(({value}) => value > 0)
                .sort((a, b) => b.value - a.value)
                .slice(0, maxConnections);

            for (const {value, idx} of inputInfluences) {
                const connection = createConnection(idx, index, value, inputInfluences);
                if (connection) newConnections.push(connection);
            }

            const previousOutputInfluences = [];
            for (let i = analysis.data.input_tokens.length; i < index; i++) {
                const outputIdx = i - analysis.data.input_tokens.length;
                const value = row[analysis.data.input_tokens.length + outputIdx];
                if (value > 0) {
                    previousOutputInfluences.push({value, idx: i});
                }
            }

            const topPreviousInfluences = previousOutputInfluences
                .sort((a, b) => b.value - a.value)
                .slice(0, maxConnections);

            for (const {value, idx} of topPreviousInfluences) {
                const connection = createConnection(idx, index, value, topPreviousInfluences);
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
        if (activeWordIndex === null) return {}; // No style if no active word

        const isInputToken = index < analysis.data.input_tokens.length;
        let strength = 0;

        if (activeWordIndex < analysis.data.input_tokens.length) { // is active word an input token? -> true if yes
            if (!isInputToken) { // current token
                const outputIndex = index - analysis.data.input_tokens.length;
                strength = analysis.data.normalized_association[outputIndex][activeWordIndex]; // getting value from matrix, between 0 and 1
                // console.log(analysis.data.output_tokens[index - analysis.data.input_tokens.length].clean_token, strength)
            }

        } else { // active word is output token
            const activeOutputIndex = activeWordIndex - analysis.data.input_tokens.length;

            if (isInputToken) {
                strength = analysis.data.normalized_association[activeOutputIndex][index];
            } else {
                const currentOutputIndex = index - analysis.data.input_tokens.length;
                if (currentOutputIndex < activeOutputIndex) {
                    const row = analysis.data.normalized_association[activeOutputIndex];
                    strength = row[analysis.data.input_tokens.length + currentOutputIndex] || 0;
                }
            }
        }

        if (index === activeWordIndex) {
            return {};
        }

        if (useRelativeStrength && strength > 0) {
            // const log_table = {
            //     token: isInputToken
            //         ? analysis.data.input_tokens[index].clean_token
            //         : analysis.data.output_tokens[index - analysis.data.input_tokens.length].clean_token,
            //     strength: strength,
            //     backgroundColor: `rgba(234, 88, 12, ${strength * 0.1})`,
            //     opacity: 0.5 + (strength * 0.5)
            // }
            // console.table(log_table)
        }

        return {
            backgroundColor: `rgba(234, 88, 12, ${strength * 0.1})`,
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
            <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                    className="bg-orange-500 rounded-full h-2 transition-all duration-300"
                    style={{
                        width: `${tokenImportance[globalIndex] * 100}%`
                    }}
                />
            </div>
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