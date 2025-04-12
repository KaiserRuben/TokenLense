import { NextResponse } from 'next/server';
import { getTimingResults } from '@/lib/api';

export async function GET() {
  try {
    const { method_timing, prompt_timing } = await getTimingResults();

    // Process data for different visualizations
    const methodPerformance = method_timing.map(item => {
      // Determine hardware type (CPU vs GPU)
      const hardware = item.cuda_available && item.gpu_info ? 'GPU' : 'CPU';
      
      // Use either average_time or average_prompt_time, whichever is available
      const avgTime = item.average_time !== undefined ? item.average_time : (item.average_prompt_time || 0);
      
      return {
        method: item.attribution_method,
        model: item.model,
        avg_time: avgTime,
        hardware
      };
    });
    
    const tokensPerSecond = prompt_timing.map(item => ({
      method: item.attribution_method,
      model: item.model,
      tokens_per_second: item.tokens_per_second
    }));
    
    const hardwareComparison = method_timing.map(item => {
      const hardware = item.gpu_info || item.cpu_model || 'Unknown';
      
      // Use either average_time or average_prompt_time, whichever is available
      const avgTime = item.average_time !== undefined ? item.average_time : (item.average_prompt_time || 0);
      
      return {
        method: item.attribution_method,
        hardware,
        avg_time: avgTime,
        model: item.model
      };
    });

    return NextResponse.json({
      methodPerformance,
      tokensPerSecond,
      hardwareComparison,
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