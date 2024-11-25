import {FC, useMemo, useState} from 'react';
import { Slider } from '@/components/ui/slider';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { motion } from 'motion/react';
import { WordCloud } from './WordCloud';
import type { AnalysisResult } from '@/utils/data';
import { createWordConnections } from '@/utils/dataProcessing';

interface TokenExplorerProps {
    analysis: AnalysisResult;
    isDark?: boolean;
}

export const TokenExplorer: FC<TokenExplorerProps> = ({
                                                                analysis
                                                            }) => {
    const [maxConnections, setMaxConnections] = useState(5);

    // Fixed minWeight to a reasonable default instead of making it adjustable
    const minWeight = 0.1;

    const connections = useMemo(() =>
            createWordConnections(analysis, minWeight),
        [analysis]
    );

    return (
        <Card className="w-full bg-white dark:bg-gray-900">
            <CardHeader className="border-b border-gray-100 dark:border-gray-800">
                <div className="flex items-center justify-between">
                    <CardTitle className="text-xl font-semibold">
                        Token Relationships
                    </CardTitle>
                    <motion.div
                        className="text-sm font-medium text-gray-500 dark:text-gray-400"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.2 }}
                    >
                        {connections.length} tokens
                    </motion.div>
                </div>
            </CardHeader>
            <CardContent className="p-0"> {/* Remove padding for better space usage */}
                <div>
                    {/* Simplified controls */}
                    <div className="p-4 border-b border-gray-100 dark:border-gray-800">
                        <div className="flex items-center gap-4">
                            <label
                                className="text-sm font-medium text-gray-700 dark:text-gray-300 min-w-[180px]"
                            >
                                Max Connections per Token
                            </label>
                            <Slider
                                value={[maxConnections]}
                                onValueChange={([value]) => setMaxConnections(value)}
                                min={1}
                                max={10}
                                step={1}
                                className="w-48"
                            />
                            <span className="text-sm text-gray-500 dark:text-gray-400 min-w-[30px]">
                                {maxConnections}
                            </span>
                        </div>
                    </div>

                    {/* Word Cloud with maximized height */}
                    <div className="h-[600px]">
                        <WordCloud
                        />
                    </div>
                </div>
            </CardContent>
        </Card>
    );
};

export default TokenExplorer;