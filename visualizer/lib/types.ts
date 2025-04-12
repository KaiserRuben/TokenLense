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
  attribution_method: string; // Method name from API
  method?: string; // For compatibility
  average_prompt_time: number; // Average time in seconds
  average_time?: number; // For compatibility
  min_time: number;
  max_time: number;
  success_rate: number;
  tokens_per_second: number;
  hardware_id?: string; // Added by processTimingResults
  gpu_info?: string | null; // GPU information if available
  cpu_model?: string; // CPU model information
  cuda_available?: boolean; // Whether CUDA is available
}

export interface PromptTimingResult {
  prompt: string;
  model: string;
  attribution_method: string; // Method name from API
  method?: string; // For compatibility
  time: number;
  success: boolean;
  tokens: number;
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

// New types for TokenLense API

// Basic token representation
export interface TokenWithId {
  id: number;
  token: string;
}

// Aggregation methods from API
export type AggregationMethod = 'sum' | 'mean' | 'l2_norm' | 'abs_sum' | 'max';

// Attribution matrix information
export interface AttributionMatrixInfo {
  shape: number[];
  dtype: string;
  is_attention: boolean;
  tensor_type: string;
}

// Model information
export interface ModelInfo {
  models: string[];
}

// Methods for a specific model
export interface ModelMethods {
  model: string;
  methods: string[];
}

// Files for a specific model and method
export interface ModelMethodFile {
  model: string;
  method: string;
  files: string[];
  file_details?: Record<string, any>[] | null;
}

// Available aggregation methods
export interface AggregationOptions {
  methods: AggregationMethod[];
  default: AggregationMethod;
}

// Basic attribution response
export interface AttributionResponse {
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

// Detailed attribution response with token IDs
export interface AttributionDetailedResponse {
  model: string;
  method: string;
  file_id: number;
  prompt: string;
  generation: string;
  source_tokens: TokenWithId[];
  target_tokens: TokenWithId[];
  attribution_matrix: number[][];
  matrix_info: AttributionMatrixInfo;
  aggregation: string;
  exec_time: number;
  original_attribution_shape: number[];
}

// Transformed data for visualization
export interface TokenData {
  token: string;
  token_id: number;
  clean_token: string;
}

export interface AnalysisData {
  input_tokens: TokenData[];
  output_tokens: TokenData[];
  association_matrix: number[][];
  normalized_association: number[][];
  input_preview?: string;
}

export interface AnalysisMetadata {
  timestamp: string;
  llm_id: string;
  llm_version: string;
  prompt: string;
  generation_params: {
    max_new_tokens: number;
    [key: string]: unknown;
  };
  version: string;
}

export interface AnalysisResult {
  metadata: AnalysisMetadata;
  data: AnalysisData;
}

// Interface for visualization settings
export interface VisualizationSettings {
  maxConnections: number;
  useRelativeStrength: boolean;
  showBackground: boolean;
  showConnections: boolean;
  showImportanceBars: boolean;
  contextSize: number;
}

// Interface for method comparison
export interface MethodComparisonData {
  file: string;
  methods: string[];
  attributions: Record<string, AttributionResponse>;
}

// Interface for token importance
export interface TokenImportanceData {
  token: string;
  token_index: number;
  files: string[];
  importances: Record<string, number>;
  importance?: number; // Normalized importance value between 0 and 1
}