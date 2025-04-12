"use client"

import {useState, useEffect, Suspense} from 'react';
import {getPerformanceData} from '@/lib/api';
import {PerformancePageData, MethodTimingResult} from '@/lib/types';
import {Card, CardHeader, CardTitle, CardDescription, CardContent} from '@/components/ui/card';
import {Tabs, TabsContent, TabsList, TabsTrigger} from '@/components/ui/tabs';
import {Badge} from '@/components/ui/badge';
import {AlertCircle, Clock, Zap, Info} from 'lucide-react';
import MethodPerformanceChart from '@/components/charts/MethodPerformanceChart';
import TokensPerSecondHeatmap from '@/components/charts/TokensPerSecondHeatmap';
import HardwareComparisonChart from '@/components/charts/HardwareComparisonChart';
import PerformanceInsightsCard from '@/components/performance/PerformanceInsightsCard';
import Loading from '@/components/ui/skeleton';

const ChartLoadingPlaceholder = () => {
    return (
        <div className="flex items-center justify-center min-h-[400px]">
            <Loading className="w-full h-[400px]"/>
        </div>
    );
};

export default function PerformancePage() {
    const [performanceData, setPerformanceData] = useState<PerformancePageData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // Filters
    const [selectedModels, setSelectedModels] = useState<string[]>([]);
    const [selectedMethods, setSelectedMethods] = useState<string[]>([]);
    const [activeTab, setActiveTab] = useState('overview');

    useEffect(() => {
        async function fetchData() {
            try {
                setLoading(true);
                const data = await getPerformanceData();
                setPerformanceData(data);

                // Initialize filters with all available options
                if (data) {
                    const models = Array.from(new Set(data.methodPerformance.map(item => item.model)));
                    const methods = Array.from(new Set(data.methodPerformance.map(item => item.method)));
                    setSelectedModels(models);
                    setSelectedMethods(methods);
                }
            } catch (err) {
                console.error('Error fetching performance data:', err);
                setError('Failed to load performance data. Please try again later.');
            } finally {
                setLoading(false);
            }
        }

        fetchData();
    }, []);

    // Define proper types for summary statistics
    interface SummaryStats {
        totalModels: number;
        totalMethods: number;
        fastestMethod: MethodTimingResult | undefined;
        slowestMethod: MethodTimingResult | undefined;
        fastestTokens: MethodTimingResult | undefined;
        successRate: number;
    }

    // Calculate summary statistics
    const getSummaryStats = (): SummaryStats | null => {
        if (!performanceData?.rawData.methodTiming) return null;

        const successfulMethods = performanceData.rawData.methodTiming.filter(
            m => m.success_rate && m.success_rate > 0
        );

        const fastestMethod = [...successfulMethods].sort((a, b) => {
            const aTime = a.average_time !== undefined ? a.average_time : (a.average_prompt_time || 0);
            const bTime = b.average_time !== undefined ? b.average_time : (b.average_prompt_time || 0);
            return aTime - bTime;
        })[0];

        const slowestMethod = [...successfulMethods].sort((a, b) => {
            const aTime = a.average_time !== undefined ? a.average_time : (a.average_prompt_time || 0);
            const bTime = b.average_time !== undefined ? b.average_time : (b.average_prompt_time || 0);
            return bTime - aTime;
        })[0];

        const fastestTokens = [...successfulMethods]
            .filter(m => m.tokens_per_second)
            .sort((a, b) => (b.tokens_per_second || 0) - (a.tokens_per_second || 0))[0];

        return {
            totalModels: new Set(successfulMethods.map(m => m.model)).size,
            totalMethods: new Set(successfulMethods.map(m => m.method || m.attribution_method || '')).size,
            fastestMethod,
            slowestMethod,
            fastestTokens,
            successRate: (successfulMethods.length / performanceData.rawData.methodTiming.length) * 100
        };
    };

    // Filter data based on selected filters
    const getFilteredData = () => {
        if (!performanceData) return null;

        return {
            methodPerformance: performanceData.methodPerformance.filter(
                item => selectedModels.includes(item.model) && selectedMethods.includes(item.method)
            ),
            tokensPerSecond: performanceData.tokensPerSecond.filter(
                item => selectedModels.includes(item.model) && selectedMethods.includes(item.method)
            ),
            hardwareComparison: performanceData.hardwareComparison.filter(
                item => selectedMethods.includes(item.method)
            ),
            rawData: performanceData.rawData
        };
    };

    const filteredData = getFilteredData();
    const summaryStats = getSummaryStats();

    if (loading) {
        return <ChartLoadingPlaceholder/>;
    }

    if (error || !performanceData) {
        return (
            <div className="container py-8">
                <div className="bg-destructive/10 p-4 rounded-md text-destructive flex items-center gap-2">
                    <AlertCircle size={16}/>
                    <span>{error || 'Failed to load performance data'}</span>
                </div>
            </div>
        );
    }

    // Get unique models and methods for filters
    const allModels = Array.from(new Set(performanceData.methodPerformance.map(item => item.model)));
    const allMethods = Array.from(new Set(performanceData.methodPerformance.map(item => item.method)));

    const formatMethodName = (method: string) => {
        if (!method) return 'Unknown';

        // Remove 'method_' prefix if it exists
        let name = method.replace('method_', '');

        // Replace underscores with spaces
        name = name.replace(/_/g, ' ');

        // Capitalize first letter of each word
        return name
            .split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    };

    return (
        <div className="space-y-8">
            <div className="max-w-4xl">
                <h1 className="text-4xl font-bold tracking-tight">Performance Analysis</h1>
                <p className="mt-4 text-lg text-muted-foreground">
                    Explore and compare performance metrics across different attribution methods, models, and hardware
                    configurations.
                </p>
            </div>

            {/* Filters */}
            <div className="flex flex-col md:flex-row gap-4">
                <Card className="w-full md:w-1/2">
                    <CardHeader className="pb-2">
                        <CardTitle className="text-lg">Model Filter</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="flex flex-wrap gap-2">
                            {allModels.map(model => (
                                <Badge
                                    key={model}
                                    variant={selectedModels.includes(model) ? "default" : "outline"}
                                    className="cursor-pointer"
                                    onClick={() => {
                                        if (selectedModels.includes(model)) {
                                            // Don't allow deselecting all models
                                            if (selectedModels.length > 1) {
                                                setSelectedModels(selectedModels.filter(m => m !== model));
                                            }
                                        } else {
                                            setSelectedModels([...selectedModels, model]);
                                        }
                                    }}
                                >
                                    {model}
                                </Badge>
                            ))}
                        </div>
                    </CardContent>
                </Card>

                <Card className="w-full md:w-1/2">
                    <CardHeader className="pb-2">
                        <CardTitle className="text-lg">Method Filter</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="flex flex-wrap gap-2">
                            {allMethods.map(method => (
                                <Badge
                                    key={method}
                                    variant={selectedMethods.includes(method) ? "default" : "outline"}
                                    className="cursor-pointer"
                                    onClick={() => {
                                        if (selectedMethods.includes(method)) {
                                            // Don't allow deselecting all methods
                                            if (selectedMethods.length > 1) {
                                                setSelectedMethods(selectedMethods.filter(m => m !== method));
                                            }
                                        } else {
                                            setSelectedMethods([...selectedMethods, method]);
                                        }
                                    }}
                                >
                                    {formatMethodName(method)}
                                </Badge>
                            ))}
                        </div>
                    </CardContent>
                </Card>
            </div>

            {/* Dashboard Tabs */}
            <Tabs defaultValue="overview" value={activeTab} onValueChange={setActiveTab} className="w-full">
                <TabsContent value="overview" className="m-0">
                    {/* Summary Stats */}
                    {summaryStats && (
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
                            <Card>
                                <CardHeader className="pb-2">
                                    <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className="flex items-baseline justify-between">
                                        <div className="text-2xl font-bold">{summaryStats.successRate.toFixed(1)}%</div>
                                        <div className="text-xs text-muted-foreground">
                                            {summaryStats.totalModels} models × {summaryStats.totalMethods} methods
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>

                            <Card>
                                <CardHeader className="pb-2">
                                    <CardTitle className="text-sm font-medium">Fastest Method</CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className="flex flex-col gap-1">
                                        <div className="flex items-center gap-1">
                                            <Clock size={14} className="text-green-500"/>
                                            <span className="text-lg font-semibold">
                        {formatMethodName(summaryStats.fastestMethod?.method || summaryStats.fastestMethod?.attribution_method || '')}
                      </span>
                                        </div>
                                        <div className="text-xs text-muted-foreground">
                                            {(summaryStats.fastestMethod?.average_time || summaryStats.fastestMethod?.average_prompt_time || 0).toFixed(2)}s
                                            on {summaryStats.fastestMethod?.model}
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>

                            <Card>
                                <CardHeader className="pb-2">
                                    <CardTitle className="text-sm font-medium">Slowest Method</CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className="flex flex-col gap-1">
                                        <div className="flex items-center gap-1">
                                            <Clock size={14} className="text-red-500"/>
                                            <span className="text-lg font-semibold">
                        {formatMethodName(summaryStats.slowestMethod?.method || summaryStats.slowestMethod?.attribution_method || '')}
                      </span>
                                        </div>
                                        <div className="text-xs text-muted-foreground">
                                            {(summaryStats.slowestMethod?.average_time || summaryStats.slowestMethod?.average_prompt_time || 0).toFixed(2)}s
                                            on {summaryStats.slowestMethod?.model}
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>

                            <Card>
                                <CardHeader className="pb-2">
                                    <CardTitle className="text-sm font-medium">Best Throughput</CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className="flex flex-col gap-1">
                                        <div className="flex items-center gap-1">
                                            <Zap size={14} className="text-yellow-500"/>
                                            <span className="text-lg font-semibold">
                        {formatMethodName(summaryStats.fastestTokens?.method || summaryStats.fastestTokens?.attribution_method || '')}
                      </span>
                                        </div>
                                        <div className="text-xs text-muted-foreground">
                                            {(summaryStats.fastestTokens?.tokens_per_second || 0).toFixed(1)} tokens/s
                                            on {summaryStats.fastestTokens?.model}
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>
                        </div>
                    )}

                    {/* Main Charts */}
                    <div className="grid grid-cols-1 gap-8">
                        <Card>
                            <CardHeader>
                                <div className="flex flex-col md:flex-row md:items-center md:justify-between">
                                    <div>
                                        <CardTitle>Method Performance Comparison</CardTitle>
                                        <CardDescription>
                                            Comparison of average processing time across different attribution methods
                                            and models
                                        </CardDescription>
                                    </div>
                                    <div className="flex items-center text-xs text-muted-foreground mt-2 md:mt-0">
                                        <Info size={14} className="mr-1"/>
                                        <span>Lower bars indicate better performance</span>
                                    </div>
                                </div>
                            </CardHeader>
                            <CardContent>
                                <Suspense fallback={<ChartLoadingPlaceholder/>}>
                                    <MethodPerformanceChart data={filteredData?.methodPerformance || []}/>
                                </Suspense>
                            </CardContent>
                        </Card>

                        <Card>
                            <CardHeader>
                                <div className="flex flex-col md:flex-row md:items-center md:justify-between">
                                    <div>
                                        <CardTitle>Tokens Processed Per Second</CardTitle>
                                        <CardDescription>
                                            Heatmap visualization of tokens processed per second for each model and
                                            attribution method
                                        </CardDescription>
                                    </div>
                                </div>
                            </CardHeader>
                            <CardContent>
                                <Suspense fallback={<ChartLoadingPlaceholder/>}>
                                    <TokensPerSecondHeatmap data={filteredData?.tokensPerSecond || []}/>
                                </Suspense>
                            </CardContent>
                        </Card>

                        <Card>
                            <CardHeader>
                                <div className="flex flex-col md:flex-row md:items-center md:justify-between">
                                    <div>
                                        <CardTitle>Hardware Comparison</CardTitle>
                                        <CardDescription>
                                            Performance comparison across different hardware configurations
                                        </CardDescription>
                                    </div>
                                    <div className="flex items-center text-xs text-muted-foreground mt-2 md:mt-0">
                                        <Info size={14} className="mr-1"/>
                                        <span>Compare how different hardware affects each method</span>
                                    </div>
                                </div>
                            </CardHeader>
                            <CardContent>
                                <Suspense fallback={<ChartLoadingPlaceholder/>}>
                                    <HardwareComparisonChart data={filteredData?.hardwareComparison || []}/>
                                </Suspense>
                            </CardContent>
                        </Card>
                    </div>
                </TabsContent>

                <TabsContent value="insights" className="m-0">
                    <PerformanceInsightsCard data={performanceData}/>
                </TabsContent>

                <TabsContent value="models" className="m-0">
                    {/* Model-specific analysis would go here */}
                    <div className="grid grid-cols-1 gap-8">
                        <Card>
                            <CardHeader>
                                <CardTitle>Model-Specific Performance</CardTitle>
                                <CardDescription>
                                    Detailed analysis of each model&#39;s performance across methods
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <p className="text-muted-foreground">This feature is under development.</p>
                            </CardContent>
                        </Card>
                    </div>
                </TabsContent>

                <TabsContent value="methods" className="m-0">
                    {/* Method-specific analysis would go here */}
                    <div className="grid grid-cols-1 gap-8">
                        <Card>
                            <CardHeader>
                                <CardTitle>Method-Specific Analysis</CardTitle>
                                <CardDescription>
                                    Detailed analysis of each attribution method across models
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <p className="text-muted-foreground">This feature is under development.</p>
                            </CardContent>
                        </Card>
                    </div>
                </TabsContent>

                <TabsList className="mt-8">
                    <TabsTrigger value="overview">Overview</TabsTrigger>
                    <TabsTrigger value="insights">Performance Insights</TabsTrigger>
                    <TabsTrigger value="models">Models</TabsTrigger>
                    <TabsTrigger value="methods">Methods</TabsTrigger>
                </TabsList>
            </Tabs>
        </div>
    );
}