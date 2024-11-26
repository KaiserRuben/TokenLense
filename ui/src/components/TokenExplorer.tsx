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
    const [maxConnections, setMaxConnections] = useState(3)  //Math.round(analysis.data.output_tokens.length/4));
    const [useRelativeStrength, setUseRelativeStrength] = useState(true);
    const [showBackground, setShowBackground] = useState(true);
    const [showConnections, setShowConnections] = useState(true);
    const [showImportanceBars, setShowImportanceBars] = useState(false);

    // Just to trick TypeScript into not complaining about unused variables
    void(setUseRelativeStrength)

    return (
        <Card>
            <CardContent className="p-0">
                <AnimatePresence key={1}>
                    <motion.div>
                        <div className="p-4 space-y-4 border-b border-gray-100 dark:border-gray-800">
                            {/* Connection controls */}
                            <div className="flex items-center gap-4">
                                <label className="text-sm font-medium text-gray-700 dark:text-gray-300 min-w-[180px]">
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
                            </div>

                            {/* Visualization toggles */}
                            <div className="flex items-center gap-8">
                                {/*<div className="flex items-center gap-2">*/}
                                {/*    <span className="text-sm text-gray-400">Normalize per token</span>*/}
                                {/*    <Switch*/}
                                {/*        checked={useRelativeStrength}*/}
                                {/*        onCheckedChange={setUseRelativeStrength}*/}
                                {/*    />*/}
                                {/*</div>*/}
                                <div className="flex items-center gap-2">
                                    <span className="text-sm text-gray-400">Show background</span>
                                    <Switch
                                        checked={showBackground}
                                        onCheckedChange={setShowBackground}
                                    />
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className="text-sm text-gray-400">Show connections</span>
                                    <Switch
                                        checked={showConnections}
                                        onCheckedChange={setShowConnections}
                                    />
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className="text-sm text-gray-400">Show importance bars</span>
                                    <Switch
                                        checked={showImportanceBars}
                                        onCheckedChange={setShowImportanceBars}
                                    />
                                </div>
                            </div>
                        </div>

                        <div className="">
                            <WordCloud
                                analysis={analysis}
                                maxConnections={maxConnections}
                                useRelativeStrength={useRelativeStrength}
                                showBackground={showBackground}
                                showConnections={showConnections}
                                showImportanceBars={showImportanceBars}
                            />
                        </div>
                    </motion.div>
                </AnimatePresence>
            </CardContent>
        </Card>
    );
};

export default TokenExplorer;