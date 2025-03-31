import React, { useMemo } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { AnalysisResult, TokenData } from "@/utils/types";
import { ensureDecompressedAnalysis } from '@/utils/matrixUtils';

interface AssociationMatrixProps {
    analysis: AnalysisResult;
}

const AssociationMatrix: React.FC<AssociationMatrixProps> = ({ analysis }) => {
    // Ensure we're working with decompressed matrices
    const decompressedAnalysis = useMemo(() => ensureDecompressedAnalysis(analysis), [analysis]);
    const { normalized_association, input_tokens, output_tokens } = decompressedAnalysis.data;

    const getColor = (value: number): string => {
        if (value === 0) return 'rgb(243, 244, 246)';
        // Use CSS variables for theming
        return `rgba(var(--token-highlight-rgb), ${value})`;
    };

    const formatNumber = (num: number): string => {
        return (num * 100).toFixed(1) + '%';
    };

    const formatToken = (token: string): string => {
        return token.length > 15 ? token.slice(0, 15) + '...' : token;
    };

    const getColumnToken = (idx: number): TokenData => {
        if (idx < input_tokens.length) {
            return input_tokens[idx];
        }
        return output_tokens[idx - input_tokens.length];
    };

    // Calculate total context width for each row
    const getContextWidth = (rowIdx: number): number => {
        return input_tokens.length + rowIdx;
    };

    return (
        <Card className="w-full">
            <CardHeader className="pb-2">
                <CardTitle className="text-lg">Token Association Matrix</CardTitle>
            </CardHeader>
            <CardContent>
                <div className="overflow-x-auto">
                    <div className="min-w-max">
                        <div className="flex">
                            {/* First column: Row headers (Generated Tokens) */}
                            <div className="flex flex-col mr-2">
                                <div className="h-12" /> {/* Spacer for column headers */}
                                {output_tokens.map((token, idx) => (
                                    <div key={`row-header-${idx}`}
                                         className="h-4 flex items-center">
                                        <span className="text-[10px] block w-12 dark:text-gray-300">
                                            {formatToken(token.clean_token || token.token)}
                                        </span>
                                    </div>
                                ))}
                            </div>

                            {/* Matrix content with column headers */}
                            <div>
                                {/* Column headers (Input + Previous Output Tokens) */}
                                <div className="flex h-12">
                                    {[...Array(input_tokens.length + output_tokens.length)].map((_, colIdx) => {
                                        const token = getColumnToken(colIdx);
                                        return (
                                            <div
                                                key={`col-header-${colIdx}`}
                                                className="w-4 flex items-end justify-center relative"
                                            >
                                                <span
                                                    className="absolute bottom-0 left-2 text-xs whitespace-nowrap origin-bottom-left -rotate-45 dark:text-gray-300">
                                                    {formatToken(token.clean_token || token.token)}
                                                </span>
                                            </div>
                                        );
                                    })}
                                </div>

                                {/* Matrix cells */}
                                <div className="flex flex-col">
                                    {normalized_association.map((row, rowIdx) => (
                                        <div key={`matrix-row-${rowIdx}`} className="flex">
                                            {row.map((value, colIdx) => {
                                                const contextWidth = getContextWidth(rowIdx);
                                                const isActive = colIdx < contextWidth;

                                                return (
                                                    <TooltipProvider key={`cell-${rowIdx}-${colIdx}`}>
                                                        <Tooltip>
                                                            <TooltipTrigger>
                                                                <div
                                                                    className={`w-4 h-4 border border-gray-100 dark:border-gray-800
                                                                        ${!isActive ? 'bg-gray-50 dark:bg-gray-900' : ''}`}
                                                                    style={{
                                                                        backgroundColor: isActive ? getColor(value) : undefined
                                                                    }}
                                                                />
                                                            </TooltipTrigger>
                                                            {isActive && value > 0 && (
                                                                <TooltipContent>
                                                                    <div className="space-y-1 p-1">
                                                                        <p className="text-xs">
                                                                            <span
                                                                                className="font-medium">Generated:</span> {output_tokens[rowIdx].clean_token || output_tokens[rowIdx].token}
                                                                        </p>
                                                                        <p className="text-xs">
                                                                            <span
                                                                                className="font-medium">Influenced by:</span> {getColumnToken(colIdx).clean_token || getColumnToken(colIdx).token}
                                                                        </p>
                                                                        <p className="text-xs">
                                                                            <span
                                                                                className="font-medium">Strength:</span> {formatNumber(value)}
                                                                        </p>
                                                                    </div>
                                                                </TooltipContent>
                                                            )}
                                                        </Tooltip>
                                                    </TooltipProvider>
                                                );
                                            })}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>

                        {/* Legend */}
                        <div className="mt-4 flex items-center gap-2 text-xs">
                            <span className="text-gray-600 dark:text-gray-400">Influence:</span>
                            <div className="flex items-center gap-1">
                                <div className="w-4 h-4" style={{backgroundColor: getColor(0)}}/>
                                <span className="dark:text-gray-300">0%</span>
                                <div className="w-4 h-4" style={{backgroundColor: getColor(0.5)}}/>
                                <span className="dark:text-gray-300">50%</span>
                                <div className="w-4 h-4" style={{backgroundColor: getColor(1)}}/>
                                <span className="dark:text-gray-300">100%</span>
                            </div>
                        </div>
                    </div>
                </div>
            </CardContent>
        </Card>
    );
};

export default AssociationMatrix;