import React, { useState, useMemo } from 'react';
import {ArrowLeft, Brain, Cpu, Settings2, Terminal, Timer} from 'lucide-react';
import { useNavigate, useParams } from 'react-router';
import { motion, AnimatePresence } from 'motion/react';
import { useAnalysis } from '@/contexts/AnalysisContext';
import { TokenExplorer } from '@/components/TokenExplorer';
import LoadingState from '@/components/LoadingState';
import ErrorState from '@/components/ErrorState';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from '@/components/ui/tooltip';
import {cleanSystemTokens} from "@/utils/data.ts";
import AssociationMatrix from "@/components/AssociationMatrix.tsx";

interface MetadataCardProps {
    title: string;
    value: string | number;
    icon?: React.ReactNode;
}

const MetadataCard: React.FC<MetadataCardProps> = ({ title, value, icon }) => (
    <Card className="relative overflow-hidden group hover:shadow-lg transition-all duration-300">
        <CardContent className="pt-6">
            <div className="flex items-start justify-between">
                <div>
                    <p className={`text-sm font-medium text-gray-600 dark:text-gray-400`}>
                        {title}
                    </p>
                    <h3 className={`mt-2 text-2xl font-light text-gray-900 dark:text-white`}>
                        {value}
                    </h3>
                </div>
                {icon && (
                    <span className={`p-2 rounded-lg bg-gray-100 dark:bg-gray-800`}>
            {icon}
          </span>
                )}
            </div>
            <div
                className="absolute inset-x-0 bottom-0 h-0.5 bg-gradient-to-r from-blue-500/30 to-purple-500/30
                   transform origin-left scale-x-0 transition-transform group-hover:scale-x-100"
            />
        </CardContent>
    </Card>
);

const Dashboard: React.FC = () => {
    const { id } = useParams<{ id: string }>();
    const navigate = useNavigate();
    const { analyses, selectedAnalysis, setSelectedAnalysis, loading, error } = useAnalysis();
    const [activeTab, setActiveTab] = useState('relationships');

    // Memoized metadata values
    const metadata = useMemo(() => {
        if (!selectedAnalysis) return null;
        return {
            tokenCountOutput: selectedAnalysis.data.output_tokens.length,
            tokenCountInput: selectedAnalysis.data.input_tokens.length,
            modelVersion: selectedAnalysis.metadata.llm_version,
            timestamp: new Date(selectedAnalysis.metadata.timestamp).toLocaleString(),
        };
    }, [selectedAnalysis]);

    // Find and set the selected analysis if not already set
    React.useEffect(() => {
        if (!selectedAnalysis && id && analyses.length > 0) {
            const found = analyses.find(a => {
                return encodeURIComponent(a.metadata.timestamp) === encodeURIComponent(id)
            });
            if (found) {
                setSelectedAnalysis(found);
            }
        }
    }, [id, analyses, selectedAnalysis, setSelectedAnalysis]);

    const handleBack = () => {
        setSelectedAnalysis(null);
        navigate('/');
    };

    if (loading) {
        return <LoadingState />;
    }

    if (error || !selectedAnalysis || !metadata) {
        return (
            <ErrorState
                error={error || "Analysis not found"}
                onBack={handleBack}
                title="Analysis Failed"
                subtitle="Unable to load analysis data. Please try again."
            />
        );
    }

    return (
        <motion.div
            className="min-h-screen"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            key={0}
        >
            {/* Navigation Header */}
            <motion.div
                className="h-16 border-b dark:border-gray-800 border-gray-200"
                initial={{ y: -20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.2 }}
                key={1}
            >
                <div className="max-w-[1400px] mx-auto px-6 h-full flex items-center justify-between">
                    <button
                        onClick={handleBack}
                        className="group flex items-center gap-2 text-sm font-medium
                     dark:text-gray-300 text-gray-700
                     hover:text-gray-900 dark:hover:text-white
                     transition-colors"
                    >
                        <ArrowLeft className="w-4 h-4 transition-transform group-hover:-translate-x-1" />
                        Back to Overview
                    </button>

                    <div className="flex items-center gap-3">
                        <Badge
                            variant="secondary"
                            className="bg-blue-500/10 text-blue-500 dark:bg-blue-500/20 dark:text-blue-400"
                        >
                            v{selectedAnalysis.metadata.version}
                        </Badge>
                        <TooltipProvider>
                            <Tooltip>
                                <TooltipTrigger asChild>
                                    <Button variant="ghost" size="icon">
                                        <Settings2 className="w-4 h-4" />
                                    </Button>
                                </TooltipTrigger>
                                <TooltipContent>
                                    <p>Analysis Settings</p>
                                </TooltipContent>
                            </Tooltip>
                        </TooltipProvider>
                    </div>
                </div>
            </motion.div>

            {/* Dashboard Content */}
            <div className="max-w-[1400px] mx-auto px-6 py-8">
                <motion.div
                    className="space-y-8"
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.3 }}
                    key={2}
                >
                    {/* Analysis Metadata Header */}
                    <div className="space-y-6">
                        <div>
                            <p className="text-s dark:text-gray-400 text-gray-600  whitespace-pre-wrap">
                                {cleanSystemTokens(selectedAnalysis.data.input_tokens)
                                    .replace(/\u010a/g, '\n')
                                    .trim()}
                            </p>
                            <p className="text-lg dark:text-gray-200 text-gray-800 whitespace-pre-wrap">
                                {selectedAnalysis.data.output_tokens
                                    .map(t => t.clean_token)
                                    .join(' ')
                                    .replace("<|eot_id|>", "")
                                    .replace(/\u010a/g, '\n').trim()}
                            </p>
                        </div>
                    </div>

                    {/* Main Content Tabs */}
                    <Tabs value={activeTab} onValueChange={setActiveTab}>
                        <TabsList>
                        <TabsTrigger value="relationships">Token Relationships</TabsTrigger>
                            <TabsTrigger value="matrix">Association Matrix</TabsTrigger>
                            <TabsTrigger value="stats">Statistics</TabsTrigger>
                        </TabsList>

                        <AnimatePresence key={0}>
                            <TabsContent value="relationships" className="mt-6">
                                <motion.div
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: -20 }}
                                    key={3}
                                >
                                    <TokenExplorer analysis={selectedAnalysis} />
                                </motion.div>
                            </TabsContent>

                            <TabsContent value="matrix" className="mt-6">
                                <motion.div
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: -20 }}
                                    key={4}
                                >
                                    {/* Matrix visualization will go here */}
                                    <AssociationMatrix analysis={selectedAnalysis} />
                                </motion.div>
                            </TabsContent>

                            <TabsContent value="stats" className="mt-6">
                                <motion.div
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: -20 }}
                                    key={5}
                                >
                                    {/* Statistics will go here */}
                                    <Card>
                                        <CardHeader>
                                            <CardTitle>Analysis Statistics</CardTitle>
                                        </CardHeader>
                                        <CardContent>
                                            <div className="h-[600px] flex items-center justify-center text-gray-500">
                                                Detailed statistics coming soon...
                                            </div>
                                        </CardContent>
                                    </Card>
                                </motion.div>
                            </TabsContent>
                        </AnimatePresence>
                    </Tabs>
                </motion.div>
            </div>
            <div className="px-6">
                {/* Metadata Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    <MetadataCard
                        title="Total Input Tokens"
                        value={metadata.tokenCountInput}
                        icon={<Brain className="w-4 h-4" />}
                    />
                    <MetadataCard
                        title="Total Output Tokens"
                        value={metadata.tokenCountOutput}
                        icon={<Cpu className="w-4 h-4" />}
                    />
                    <MetadataCard
                        title="Model Version"
                        value={metadata.modelVersion}
                        icon={<Terminal className="w-4 h-4" />}
                    />
                    <MetadataCard
                        title="Timestamp"
                        value={metadata.timestamp}
                        icon={<Timer className="w-4 h-4" />}
                    />
                </div>
            </div>
        </motion.div>
    );
};

export default Dashboard;