import React, { useMemo } from 'react';
import {
    CircleAlert,
    ArrowUpRight,
    ArrowDownRight,
    Link2,
    Lightbulb,
    Percent
} from 'lucide-react';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import {
    Card,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle,
} from '@/components/ui/card';
import {
    TokenData,
    AnalysisResult,
    TokenConnections,
    TokenConnection
} from '@/utils/types';
import { ensureDecompressedAnalysis } from '@/utils/matrixUtils';

interface TokenInfoSidebarProps {
    token: TokenData;
    analysis: AnalysisResult;
    activeIndex: number;
    isInputToken: boolean;
}

const TokenInfoSidebar: React.FC<TokenInfoSidebarProps> = ({
                                                               token,
                                                               analysis,
                                                               activeIndex,
                                                               isInputToken
                                                           }) => {
    // Ensure we're working with decompressed matrices
    const decompressedAnalysis = useMemo(() => ensureDecompressedAnalysis(analysis), [analysis]);

    // Calculate top influencing and influenced tokens
    const topConnections = useMemo<TokenConnections>(() => {
        if (!token) return { influencers: [], influenced: [] };

        const { input_tokens, output_tokens, normalized_association } = decompressedAnalysis.data;

        // For input tokens: find output tokens that are most influenced by this token
        if (isInputToken) {
            const influences = normalized_association.map((row, idx) => ({
                token: output_tokens[idx],
                strength: row[activeIndex]
            }));

            const topInfluenced = influences
                .sort((a, b) => b.strength - a.strength)
                .filter(item => item.strength > 0)
                .slice(0, 5);

            return { influencers: [], influenced: topInfluenced };
        }
        // For output tokens: find tokens that most influenced this token
        else {
            const outputIdx = activeIndex - input_tokens.length;
            if (outputIdx < 0 || outputIdx >= normalized_association.length) {
                return { influencers: [], influenced: [] };
            }

            const row = normalized_association[outputIdx];

            // Input tokens that influenced this output
            const inputInfluences: TokenConnection[] = input_tokens.map((token, idx) => ({
                token,
                strength: idx < row.length ? row[idx] : 0,
                isInput: true
            }));

            // Previous output tokens that influenced this output
            const prevOutputInfluences: TokenConnection[] = [];
            for (let i = 0; i < outputIdx; i++) {
                const idx = input_tokens.length + i;
                if (idx < row.length) {
                    prevOutputInfluences.push({
                        token: output_tokens[i],
                        strength: row[idx],
                        isInput: false
                    });
                }
            }

            // Combine and sort
            const allInfluences = [...inputInfluences, ...prevOutputInfluences];
            const topInfluencers = allInfluences
                .sort((a, b) => b.strength - a.strength)
                .filter(item => item.strength > 0)
                .slice(0, 5);

            return { influencers: topInfluencers, influenced: [] };
        }
    }, [token, decompressedAnalysis, activeIndex, isInputToken]);

    if (!token) return null;

    return (
        <Card className="absolute right-4 top-20 w-64 shadow-lg token-info-sidebar">
            <CardHeader className="pb-2">
                <div className="flex justify-between items-start">
                    <div>
                        <CardTitle className="text-lg font-medium">Token Info</CardTitle>
                        <CardDescription>Detailed token analysis</CardDescription>
                    </div>
                    {token.isSystem && (
                        <Badge variant="outline" className="bg-yellow-500/10 text-yellow-600 dark:text-yellow-400 dark:border-yellow-800/50 border-yellow-200">
                            <CircleAlert className="w-3 h-3 mr-1" /> System Token
                        </Badge>
                    )}
                </div>
            </CardHeader>
            <CardContent className="space-y-4">
                {/* Token Preview */}
                <div className="space-y-1.5">
                    <p className="text-xs text-gray-500 dark:text-gray-400">Token</p>
                    <div className="flex space-x-2 items-center">
                        <div className={`px-2 py-1 rounded ${
                            isInputToken
                                ? "bg-orange-100 dark:bg-orange-900/20 border-orange-200 dark:border-orange-800/30"
                                : "bg-blue-100 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800/30"
                        } border`}>
                            <code className={`text-sm ${
                                isInputToken
                                    ? "text-orange-600 dark:text-orange-400"
                                    : "text-blue-600 dark:text-blue-400"
                            }`}>
                                {token.clean_token || token.token}
                            </code>
                        </div>
                    </div>
                </div>

                {/* Token Metadata */}
                <div className="space-y-1.5">
                    <p className="text-xs text-gray-500 dark:text-gray-400">Token Metadata</p>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                        <div>
                            <p className="text-xs font-medium text-gray-500 dark:text-gray-400">ID</p>
                            <p className="dark:text-gray-300">{token.token_id}</p>
                        </div>
                        <div>
                            <p className="text-xs font-medium text-gray-500 dark:text-gray-400">Type</p>
                            <p className="dark:text-gray-300">{isInputToken ? 'Input' : 'Output'}</p>
                        </div>
                    </div>
                </div>

                {/* Token Importance */}
                <div className="space-y-1.5">
                    <div className="flex items-center gap-1 text-xs text-gray-500 dark:text-gray-400">
                        <Lightbulb className="w-3.5 h-3.5" />
                        <p>Global Importance</p>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-full importance-bar-bg">
                            <div
                                className={isInputToken ? "importance-bar-fill-input" : "importance-bar-fill-output"}
                                style={{ width: `${(token.calculatedImportance || 0) * 100}%` }}
                            />
                        </div>
                        <span className="font-medium text-sm flex items-center dark:text-gray-300">
              {((token.calculatedImportance || 0) * 100).toFixed(1)}
                            <Percent className="w-3 h-3 ml-0.5" />
            </span>
                    </div>
                </div>

                <Separator />

                {/* Relationship Information */}
                {isInputToken ? (
                    <div className="space-y-2">
                        <div className="flex items-center gap-1.5 text-xs font-medium">
                            <ArrowDownRight className="w-3.5 h-3.5 text-blue-500 dark:text-blue-400" />
                            <span className="dark:text-gray-200">Influences These Tokens</span>
                        </div>

                        {topConnections.influenced.length > 0 ? (
                            <div className="space-y-2">
                                {topConnections.influenced.map((connection, idx) => (
                                    <div key={idx} className="flex justify-between items-center">
                                        <div className="px-1.5 py-0.5 rounded bg-blue-100/50 dark:bg-blue-900/20 text-xs dark:text-blue-200">
                                            {connection.token.clean_token}
                                        </div>
                                        <div className="flex items-center">
                                            <div className="w-16 h-1.5 bg-gray-200 dark:bg-gray-800 rounded-full mr-1.5">
                                                <div
                                                    className="h-full bg-blue-500 dark:bg-blue-400 rounded-full"
                                                    style={{
                                                        width: `${connection.strength * 100}%`
                                                    }}
                                                />
                                            </div>
                                            <span className="text-xs font-medium dark:text-gray-300">{(connection.strength * 100).toFixed(0)}%</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <p className="text-xs text-gray-500 dark:text-gray-400 italic">No significant influences found</p>
                        )}
                    </div>
                ) : (
                    <div className="space-y-2">
                        <div className="flex items-center gap-1.5 text-xs font-medium">
                            <ArrowUpRight className="w-3.5 h-3.5 text-orange-500 dark:text-orange-400" />
                            <span className="dark:text-gray-200">Influenced By These Tokens</span>
                        </div>

                        {topConnections.influencers.length > 0 ? (
                            <div className="space-y-2">
                                {topConnections.influencers.map((connection, idx) => (
                                    <div key={idx} className="flex justify-between items-center">
                                        <div className={`px-1.5 py-0.5 rounded ${
                                            connection.isInput
                                                ? 'bg-orange-100/50 dark:bg-orange-900/20 dark:text-orange-200'
                                                : 'bg-blue-100/50 dark:bg-blue-900/20 dark:text-blue-200'
                                        } text-xs`}>
                                            {connection.token.clean_token}
                                        </div>
                                        <div className="flex items-center">
                                            <div className="w-16 h-1.5 bg-gray-200 dark:bg-gray-800 rounded-full mr-1.5">
                                                <div
                                                    className={`h-full ${
                                                        connection.isInput
                                                            ? 'bg-orange-500 dark:bg-orange-400'
                                                            : 'bg-blue-500 dark:bg-blue-400'
                                                    } rounded-full`}
                                                    style={{
                                                        width: `${connection.strength * 100}%`
                                                    }}
                                                />
                                            </div>
                                            <span className="text-xs font-medium dark:text-gray-300">{(connection.strength * 100).toFixed(0)}%</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <p className="text-xs text-gray-500 dark:text-gray-400 italic">No significant influences found</p>
                        )}
                    </div>
                )}

                {/* Position in Sequence */}
                <div className="space-y-1.5">
                    <div className="flex items-center gap-1.5 text-xs text-gray-500 dark:text-gray-400">
                        <Link2 className="w-3.5 h-3.5" />
                        <p>Position in Sequence</p>
                    </div>
                    <div className="relative h-6 bg-gray-100 dark:bg-gray-800 rounded overflow-hidden">
                        {isInputToken ? (
                            <div
                                className="absolute h-full bg-orange-200 dark:bg-orange-900/30"
                                style={{
                                    left: '0%',
                                    width: '50%'
                                }}
                            >
                                <div
                                    className="absolute top-0 bottom-0 w-1.5 bg-orange-500 dark:bg-orange-400"
                                    style={{
                                        left: `${(activeIndex / decompressedAnalysis.data.input_tokens.length) * 100}%`
                                    }}
                                />
                            </div>
                        ) : (
                            <div className="flex h-full">
                                <div className="w-1/2 bg-orange-200 dark:bg-orange-900/30" />
                                <div className="w-1/2 bg-blue-200 dark:bg-blue-900/30">
                                    <div
                                        className="absolute top-0 bottom-0 w-1.5 bg-blue-500 dark:bg-blue-400"
                                        style={{
                                            left: `${50 + ((activeIndex - decompressedAnalysis.data.input_tokens.length) / decompressedAnalysis.data.output_tokens.length) * 50}%`
                                        }}
                                    />
                                </div>
                            </div>
                        )}

                        <div className="absolute inset-0 flex justify-between items-center px-1.5 text-[9px] font-medium pointer-events-none dark:text-gray-400">
                            <span>Input Start</span>
                            <span className="mx-auto">|</span>
                            <span>Output End</span>
                        </div>
                    </div>
                </div>
            </CardContent>
        </Card>
    );
};

export default TokenInfoSidebar;