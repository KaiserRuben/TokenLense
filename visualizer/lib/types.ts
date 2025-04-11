// API Types
export interface SystemInfo {
  hostname: string;
  platform: string;
  platform_release: string;
  processor: string;
  cpu_count: number;
  memory_gb: number;
  python_version: string;
  torch_version: string;
  cuda_available: boolean;
  mps_available: boolean;
  gpu_info: string | null;
}

export interface MethodTimingResult {
  model: string;
  attribution_method: string;
  successful_prompts: number;
  total_prompts: number;
  success_rate: number;
  model_loading_time: number;
  attribution_time: number;
  average_prompt_time: number;
  total_time: number;
  platform: string;
  cpu_model: string;
  cpu_cores: number;
  memory_gb: number;
  gpu_info: string;
  cuda_available: boolean;
  mps_available: boolean;
  torch_version: string;
}

export interface PromptTimingResult {
  model: string;
  attribution_method: string;
  prompt: string;
  prompt_tokens: number;
  generation_tokens: number;
  attribution_time: number;
  tokens_per_second: number;
}

export interface TimingResults {
  method_timing: MethodTimingResult[];
  prompt_timing: PromptTimingResult[];
}

// Processed data types for visualizations
export interface MethodPerformanceData {
  method: string;
  model: string;
  avg_time: number;
  hardware: string;
}

export interface TokensPerSecondData {
  method: string;
  model: string;
  tokens_per_second: number;
}

export interface HardwareComparisonData {
  method: string;
  hardware: string;
  avg_time: number;
  model: string;
}

// Performance API response type
export interface PerformancePageData {
  methodPerformance: MethodPerformanceData[];
  tokensPerSecond: TokensPerSecondData[];
  hardwareComparison: HardwareComparisonData[];
  rawData: {
    methodTiming: MethodTimingResult[];
    promptTiming: PromptTimingResult[];
  };
}

// Attribution types
export interface AttributionData {
  model: string;
  method: string;
  file_id: number;
  prompt: string;
  generation: string;
  source_tokens: string[];
  target_tokens: string[];
  attribution_matrix: number[][];
  aggregation: string;
}