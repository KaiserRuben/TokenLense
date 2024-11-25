import React, {useState} from 'react';
import {Slider} from '@/components/ui/slider';
import {Card, CardContent} from '@/components/ui/card';
import {AnimatePresence, motion} from 'motion/react';
import type {AnalysisResult} from '@/utils/data';
import WordCloud from "@/components/WordCloud.tsx";
import {Switch} from "@/components/ui/switch.tsx";

interface TokenExplorerProps {
    analysis: AnalysisResult;
}

export const TokenExplorer: React.FC<TokenExplorerProps> = ({analysis}) => {
    const [maxConnections, setMaxConnections] = useState(3);
    const [useRelativeStrength, setUseRelativeStrength] = useState(false);

    return (
        <Card>
            <CardContent className="p-0"> {/* Remove padding for better space usage */}
                <AnimatePresence key={1}>
                    <motion.div>
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
                                    max={analysis.data.normalized_association.length}
                                    step={1}
                                    className="w-48"
                                />
                                <span className="text-sm text-gray-500 dark:text-gray-400 min-w-[30px]">
                                {maxConnections}
                            </span>
                                <span className="text-sm text-gray-400">Normalize per token</span>
                                <Switch
                                    checked={useRelativeStrength}
                                    onCheckedChange={setUseRelativeStrength}
                                />
                            </div>
                        </div>

                        {/* Word Cloud with maximized height */}
                        <div className="">
                            <WordCloud
                                analysis={analysis}
                                maxConnections={maxConnections}
                                useRelativeStrength={useRelativeStrength}
                            />
                        </div>
                    </motion.div>
                </AnimatePresence>
            </CardContent>
        </Card>
    );
};

export default TokenExplorer;