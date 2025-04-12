"use client"

import { useMemo } from 'react';
import {MethodTimingResult, PerformancePageData} from '@/lib/types';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, TrendingDown, AlertCircle, Clock, Lightbulb, Zap } from 'lucide-react';

type Props = {
    data: PerformancePageData;
};

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

export default function PerformanceInsightsCard({ data }: Props) {
    // Define proper types for insight data
    interface Insight {
        title: string;
        description: string;
        type: 'success' | 'warning' | 'info';
        icon: React.FC<{ className?: string; size?: number }>;
    }

    // Generate insights from the performance data
    const insights = useMemo((): Insight[] => {
        if (!data) return [];

        const insights: Insight[] = [];
        const { methodTiming } = data.rawData;

        // Filter to only successful runs
        const successfulData = methodTiming.filter(item =>
            item.success_rate && item.success_rate > 0 &&
            (item.average_time !== undefined || item.average_prompt_time !== undefined)
        );

        if (successfulData.length === 0) {
            return [
                {
                    title: "No successful method runs found",
                    description: "All methods appear to have 0% success rate in the current dataset.",
                    type: "warning",
                    icon: AlertCircle
                }
            ];
        }

        // Find the fastest and slowest methods overall
        const sortedBySpeed = [...successfulData].sort((a, b) => {
            const aTime = a.average_time !== undefined ? a.average_time : (a.average_prompt_time || 0);
            const bTime = b.average_time !== undefined ? b.average_time : (b.average_prompt_time || 0);
            return aTime - bTime;
        });

        const fastestOverall = sortedBySpeed[0];
        const slowestOverall = sortedBySpeed[sortedBySpeed.length - 1];

        // Calculate speed difference
        const fastestTime = fastestOverall.average_time !== undefined ?
            fastestOverall.average_time : (fastestOverall.average_prompt_time || 0);
        const slowestTime = slowestOverall.average_time !== undefined ?
            slowestOverall.average_time : (slowestOverall.average_prompt_time || 0);

        const speedupFactor = slowestTime / fastestTime;

        insights.push({
            title: "Performance Range",
            description: `The fastest method (${formatMethodName(fastestOverall.method || fastestOverall.attribution_method || '')}) is ${speedupFactor.toFixed(1)}x faster than the slowest method (${formatMethodName(slowestOverall.method || slowestOverall.attribution_method || '')}).`,
            type: "info",
            icon: TrendingUp
        });

        interface MethodConsistency {
            method: string;
            consistency: number;
            count: number;
            avg?: number;
        }

        // Group by method and find the most consistent method
        const methodGroups = new Map<string, MethodTimingResult[]>();
        successfulData.forEach(item => {
            const method = item.method || item.attribution_method || 'unknown';
            if (!methodGroups.has(method)) {
                methodGroups.set(method, []);
            }
            methodGroups.get(method)?.push(item);
        });

        // Calculate performance consistency for each method
        const methodConsistency: MethodConsistency[] = Array.from(methodGroups.entries()).map(([method, items]) => {
            // Need at least 2 data points to calculate consistency
            if (items.length < 2) return { method, consistency: 1, count: items.length };

            const times = items.map(item =>
                item.average_time !== undefined ? item.average_time : (item.average_prompt_time || 0)
            );

            const avg = times.reduce((sum, time) => sum + time, 0) / times.length;
            const variance = times.reduce((sum, time) => sum + Math.pow(time - avg, 2), 0) / times.length;
            const stdDev = Math.sqrt(variance);
            const coefficient = stdDev / avg; // Lower is more consistent

            return { method, consistency: coefficient, count: items.length, avg };
        }).filter(item => item.count >= 2); // Only methods with enough data points

        // Sort by consistency (lower coefficient = more consistent)
        methodConsistency.sort((a, b) => a.consistency - b.consistency);

        if (methodConsistency.length > 0) {
            const mostConsistent = methodConsistency[0];
            insights.push({
                title: "Most Consistent Method",
                description: `${formatMethodName(mostConsistent.method)} shows the most consistent performance across different models with a variance of only ${(mostConsistent.consistency * 100).toFixed(1)}%.`,
                type: "success",
                icon: Clock
            });
        }

        // Find methods with failed runs
        const failedMethods = methodTiming.filter(item =>
            !item.success_rate || item.success_rate === 0
        );

        if (failedMethods.length > 0) {
            // Group by method
            const failuresByMethod = new Map<string, string[]>();
            failedMethods.forEach(item => {
                const method = item.method || item.attribution_method || 'unknown';
                const model = item.model || 'unknown';

                if (!failuresByMethod.has(method)) {
                    failuresByMethod.set(method, []);
                }
                failuresByMethod.get(method)?.push(model);
            });

            // Find method with most failures
            let methodWithMostFailures = '';
            let maxFailures = 0;
            let failedModels: string[] = [];

            failuresByMethod.forEach((models, method) => {
                if (models.length > maxFailures) {
                    maxFailures = models.length;
                    methodWithMostFailures = method;
                    failedModels = models;
                }
            });

            if (methodWithMostFailures) {
                insights.push({
                    title: "Compatibility Issues",
                    description: `${formatMethodName(methodWithMostFailures)} failed to run on ${maxFailures} models, including ${failedModels.slice(0, 3).join(', ')}${failedModels.length > 3 ? '...' : ''}.`,
                    type: "warning",
                    icon: AlertCircle
                });
            }
        }

        // Define proper types for best method tracking
        interface BestMethodInfo {
            method: string;
            time: number;
        }

        // Find best method for each model
        const modelBestMethods = new Map<string, BestMethodInfo>();
        successfulData.forEach(item => {
            const model = item.model;
            const method = item.method || item.attribution_method || 'unknown';
            const time = item.average_time !== undefined ? item.average_time : (item.average_prompt_time || 0);

            if (!modelBestMethods.has(model) || (modelBestMethods.get(model)?.time || Infinity) > time) {
                modelBestMethods.set(model, { method, time });
            }
        });

        // Count which method is best most often
        const methodBestCounts = new Map<string, number>();
        modelBestMethods.forEach(({ method }) => {
            methodBestCounts.set(method, (methodBestCounts.get(method) || 0) + 1);
        });

        let bestOverallMethod = '';
        let bestOverallCount = 0;

        methodBestCounts.forEach((count, method) => {
            if (count > bestOverallCount) {
                bestOverallCount = count;
                bestOverallMethod = method;
            }
        });

        if (bestOverallMethod) {
            const percentage = (bestOverallCount / modelBestMethods.size) * 100;
            insights.push({
                title: "Best Overall Method",
                description: `${formatMethodName(bestOverallMethod)} is the fastest method for ${bestOverallCount} out of ${modelBestMethods.size} models (${percentage.toFixed(0)}%).`,
                type: "success",
                icon: Zap
            });
        }

        // Check if attention method is consistently faster
        const attentionResults = successfulData.filter(item =>
            (item.method === 'attention' || item.attribution_method === 'attention')
        );

        if (attentionResults.length > 0) {
            const attentionModels = new Set(attentionResults.map(item => item.model));

            let attentionWins = 0;

            attentionModels.forEach(model => {
                const modelMethods = successfulData.filter(item => item.model === model);
                const attentionMethod = modelMethods.find(item =>
                    item.method === 'attention' || item.attribution_method === 'attention'
                );

                if (attentionMethod) {
                    const attentionTime = attentionMethod.average_time !== undefined ?
                        attentionMethod.average_time : (attentionMethod.average_prompt_time || 0);

                    const otherMethodsTime = modelMethods
                        .filter(item => item.method !== 'attention' && item.attribution_method !== 'attention')
                        .map(item => item.average_time !== undefined ? item.average_time : (item.average_prompt_time || 0));

                    const minOtherTime = Math.min(...otherMethodsTime);

                    if (attentionTime < minOtherTime) {
                        attentionWins++;
                    }
                }
            });

            if (attentionWins > 0) {
                const winPercentage = (attentionWins / attentionModels.size) * 100;

                insights.push({
                    title: "Attention Method Performance",
                    description: `The Attention method is the fastest option for ${attentionWins} out of ${attentionModels.size} models (${winPercentage.toFixed(0)}%).`,
                    type: "info",
                    icon: Lightbulb
                });
            }
        }

        // Define types for hardware impact analysis
        interface HardwarePerformanceData {
            times: number[];
            avg: number;
        }

        interface HardwareImpactInfo {
            method: string;
            minHardware: string;
            maxHardware: string;
            speedupFactor: number;
        }

        // Look for hardware performance differences
        const hardwareImpact: HardwareImpactInfo[] = [];

        // Get methods that have been run on multiple hardware configs
        const methodsWithMultipleHardware = successfulData.reduce((acc, item) => {
            const method = item.method || item.attribution_method || 'unknown';
            const hardware = item.gpu_info || 'unknown';

            if (!acc[method]) acc[method] = new Set<string>();
            acc[method].add(hardware);

            return acc;
        }, {} as Record<string, Set<string>>);

        // Find methods with significant hardware differences
        Object.entries(methodsWithMultipleHardware)
            // eslint-disable-next-line @typescript-eslint/no-unused-vars
            .filter(([_, hardwareSet]) => hardwareSet.size > 1)
            // eslint-disable-next-line @typescript-eslint/no-unused-vars
            .forEach(([method, _]) => {
                const methodResults = successfulData.filter(item =>
                    (item.method === method || item.attribution_method === method)
                );

                // Group by hardware
                const hardwarePerformance = methodResults.reduce((acc, item) => {
                    const hardware = item.gpu_info || 'unknown';
                    const time = item.average_time !== undefined ? item.average_time : (item.average_prompt_time || 0);

                    if (!acc[hardware]) {
                        acc[hardware] = { times: [], avg: 0 };
                    }

                    acc[hardware].times.push(time);

                    return acc;
                }, {} as Record<string, HardwarePerformanceData>);

                // Calculate average time for each hardware
                Object.entries(hardwarePerformance).forEach(([, data]) => {
                    data.avg = data.times.reduce((sum, time) => sum + time, 0) / data.times.length;
                });

                // Find min and max
                let minHardware = '';
                let maxHardware = '';
                let minTime = Infinity;
                let maxTime = -Infinity;

                Object.entries(hardwarePerformance).forEach(([hardware, data]) => {
                    if (data.avg < minTime) {
                        minTime = data.avg;
                        minHardware = hardware;
                    }

                    if (data.avg > maxTime) {
                        maxTime = data.avg;
                        maxHardware = hardware;
                    }
                });

                const speedupFactor = maxTime / minTime;

                if (speedupFactor >= 1.5) { // 50% improvement threshold
                    hardwareImpact.push({
                        method,
                        minHardware,
                        maxHardware,
                        speedupFactor
                    });
                }
            });

        if (hardwareImpact.length > 0) {
            // Sort by speedup factor
            hardwareImpact.sort((a, b) => b.speedupFactor - a.speedupFactor);

            const topImpact = hardwareImpact[0];

            insights.push({
                title: "Hardware Impact",
                description: `${formatMethodName(topImpact.method)} shows a ${topImpact.speedupFactor.toFixed(1)}x speedup on ${topImpact.minHardware} compared to ${topImpact.maxHardware}.`,
                type: "info",
                icon: TrendingDown
            });
        }

        return insights;
    }, [data]);

    if (!data || insights.length === 0) {
        return (
            <Card>
                <CardHeader>
                    <CardTitle>Performance Insights</CardTitle>
                    <CardDescription>
                        Not enough data available to generate insights.
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="flex items-center justify-center h-32 bg-muted rounded-md">
                        <p className="text-muted-foreground">Try adjusting your filters to see insights</p>
                    </div>
                </CardContent>
            </Card>
        );
    }

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {insights.map((insight, index) => (
                <Card key={index} className={`border-l-4 ${
                    insight.type === 'success' ? 'border-l-green-500' :
                        insight.type === 'warning' ? 'border-l-yellow-500' :
                            'border-l-blue-500'
                }`}>
                    <CardHeader className="pb-2">
                        <div className="flex items-center justify-between">
                            <CardTitle className="text-lg flex items-center gap-2">
                                <insight.icon className={`
                  ${insight.type === 'success' ? 'text-green-500' :
                                    insight.type === 'warning' ? 'text-yellow-500' :
                                        'text-blue-500'}
                `} size={18} />
                                {insight.title}
                            </CardTitle>
                            <Badge variant={
                                insight.type === 'success' ? 'default' :
                                    insight.type === 'warning' ? 'destructive' :
                                        'secondary'
                            }>
                                {insight.type === 'success' ? 'Insight' :
                                    insight.type === 'warning' ? 'Warning' :
                                        'Information'}
                            </Badge>
                        </div>
                    </CardHeader>
                    <CardContent>
                        <p className="text-sm text-muted-foreground">
                            {insight.description}
                        </p>
                    </CardContent>
                </Card>
            ))}
        </div>
    );
}