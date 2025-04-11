import { NextResponse } from 'next/server';
import { getTimingResults } from '@/lib/api';
import { MethodTimingResult, PromptTimingResult, MethodPerformanceData, TokensPerSecondData, HardwareComparisonData } from '@/lib/types';

export async function GET() {
  try {
    const { method_timing, prompt_timing } = await getTimingResults();

    // Process data for different visualizations
    const methodPerformance = processMethodPerformance(method_timing);
    const tokensPerSecond = processTokensPerSecond(prompt_timing);
    const hardwareComparison = processHardwareComparison(method_timing);

    return NextResponse.json({
      methodPerformance,
      tokensPerSecond,
      hardwareComparison,
      // Add original data for other potential visualizations
      rawData: {
        methodTiming: method_timing,
        promptTiming: prompt_timing
      }
    });
  } catch (error) {
    console.error('Error in performance API route:', error);
    return NextResponse.json(
      { error: 'Failed to process performance data' },
      { status: 500 }
    );
  }
}

/**
 * Process method timing results for performance visualization
 */
function processMethodPerformance(data: MethodTimingResult[]): MethodPerformanceData[] {
  return data.map(item => {
    // Determine hardware type (CPU vs GPU)
    const hardware = item.cuda_available && item.gpu_info ? 'GPU' : 'CPU';

    return {
      method: item.attribution_method,
      model: item.model,
      avg_time: item.average_prompt_time,
      hardware
    };
  });
}

/**
 * Process prompt timing results for tokens per second visualization
 */
function processTokensPerSecond(data: PromptTimingResult[]): TokensPerSecondData[] {
  return data.map(item => ({
    method: item.attribution_method,
    model: item.model,
    tokens_per_second: item.tokens_per_second
  }));
}

/**
 * Process hardware comparison data
 */
function processHardwareComparison(data: MethodTimingResult[]): HardwareComparisonData[] {
  return data.map(item => {
    const hardware = item.gpu_info || item.cpu_model;

    return {
      method: item.attribution_method,
      hardware,
      avg_time: item.average_prompt_time,
      model: item.model
    };
  });
}