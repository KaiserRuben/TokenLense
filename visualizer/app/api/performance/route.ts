import { NextResponse } from 'next/server';
import { MethodTimingResult, PromptTimingResult } from '@/lib/types';

// Base API URL from server environment variable
const API_BASE_URL = process.env.API_URL || 'http://localhost:8000';

export async function GET() {
  try {
    const response = await fetch(`${API_BASE_URL}/performance/timing`);
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }
    
    const data = await response.json();
    const { method_timing, prompt_timing } = data;

    // Process data for different visualizations
    const methodPerformance = method_timing.map((item: MethodTimingResult) => {
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
    
    const tokensPerSecond = prompt_timing.map((item: PromptTimingResult) => ({
      method: item.attribution_method,
      model: item.model,
      tokens_per_second: item.tokens_per_second
    }));
    
    const hardwareComparison = method_timing.map((item: MethodTimingResult) => {
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
    console.error('Performance data API error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch performance data' }, 
      { status: 500 }
    );
  }
}