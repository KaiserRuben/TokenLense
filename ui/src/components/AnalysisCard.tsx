import {AnalysisResult} from "@/utils/data.ts";
import React from "react";
import {Card, CardContent} from "@/components/ui/card.tsx";
import {Brain, CalendarDays, Cpu} from "lucide-react";

interface AnalysisCardProps {
    analysis: AnalysisResult;
    onClick: () => void;
}

export const AnalysisCard: React.FC<AnalysisCardProps> = ({analysis, onClick}) => {

    const date = new Date(analysis.metadata.timestamp);
    const formattedDate = new Intl.DateTimeFormat('en-US', {
        day: 'numeric',
        month: 'short',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
    }).format(date);

    const inputTokenCount = analysis.data.input_tokens.length;
    const outputTokenCount = analysis.data.output_tokens.length;

    return (
        <Card
            onClick={onClick}
            className="group relative overflow-hidden backdrop-blur-md cursor-pointer
                transition-all duration-300 hover:scale-[1.02]
                dark:bg-[rgba(20,20,25,0.9)] dark:border-white/8 dark:hover:border-white/15
                bg-white/70 border-black/5 hover:border-black/10"
        >
            <div className="absolute inset-0 bg-gradient-to-br from-transparent via-white/5 to-transparent
                    opacity-0 group-hover:opacity-80 transition-opacity duration-500"/>

            <CardContent className="relative space-y-6 p-6">
                {/* Query Preview */}
                <div className="space-y-4">
                    <p className="text-xl font-light tracking-tight
                      dark:text-white text-gray-900">
                        {analysis.data.input_preview}
                    </p>

                    {/* Metrics */}
                    <div className="flex items-center gap-8">
                        {/* Input Tokens */}
                        <div className="space-y-1.5">
                            <div className="flex items-center gap-2">
                                <Brain className="w-4 h-4 dark:text-gray-400 text-gray-600"/>
                                <span className="text-sm font-medium dark:text-gray-400 text-gray-600">
                  Input Tokens
                </span>
                            </div>
                            <p className="text-2xl font-light pl-6 dark:text-white text-gray-900">
                                {inputTokenCount.toLocaleString()}
                            </p>
                        </div>

                        {/* Output Tokens */}
                        <div className="space-y-1.5">
                            <div className="flex items-center gap-2">
                                <Cpu className="w-4 h-4 dark:text-gray-400 text-gray-600"/>
                                <span className="text-sm font-medium dark:text-gray-400 text-gray-600">
                  Output Tokens
                </span>
                            </div>
                            <p className="text-2xl font-light pl-6 dark:text-white text-gray-900">
                                {outputTokenCount.toLocaleString()}
                            </p>
                        </div>
                    </div>
                </div>

                {/* Metadata */}
                <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-4">
            <span className="font-medium dark:text-gray-300 text-gray-700">
              {analysis.metadata.llm_id}
            </span>
                        <div className="flex items-center gap-1.5 dark:text-gray-400 text-gray-600">
                            <CalendarDays className="w-4 h-4"/>
                            <span>{formattedDate}</span>
                        </div>
                    </div>
                    <div className="px-2 py-0.5 rounded-full text-xs font-medium
                        dark:bg-blue-500/20 dark:text-blue-400
                        bg-blue-100 text-blue-700">
                        v{analysis.metadata.version}
                    </div>
                </div>
            </CardContent>
        </Card>
    );
};
export default AnalysisCard;