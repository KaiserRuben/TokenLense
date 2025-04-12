import {
  SystemInfo,
  TimingResults,
  MethodTimingResult,
  ModelInfo,
  ModelMethods,
  ModelMethodFile,
  AggregationOptions,
  AttributionResponse,
  AttributionDetailedResponse,
  AggregationMethod,
  PerformancePageData,
  TokenImportanceData
} from './types';

// Base API URL - Updated to use relative paths for server-side API routes
const API_BASE_URL = '';

/**
 * Function to fetch from the API with error handling
 */
async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, options);
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }
    
    return await response.json() as T;
  } catch (error) {
    console.error('API fetch error:', error);
    throw error;
  }
}

/**
 * Get system information
 */
export async function getSystemInfo(): Promise<SystemInfo> {
  return fetchAPI<SystemInfo>('/api/performance/system');
}

/**
 * Get timing results
 */
export async function getTimingResults(): Promise<TimingResults> {
  return fetchAPI<TimingResults>('/api/performance/timing');
}

/**
 * Get processed performance data (for performance page)
 */
export async function getPerformanceData(): Promise<PerformancePageData> {
  return fetchAPI<PerformancePageData>('/api/performance');
}

/**
 * Get list of available models
 */
export async function getModels(): Promise<string[]> {
  const response = await fetchAPI<ModelInfo>('/api/models');
  return response.models;
}

/**
 * Get attribution methods for a specific model
 */
export async function getModelMethods(model: string): Promise<string[]> {
  const response = await fetchAPI<ModelMethods>(`/api/models/${model}/methods`);
  return response.methods;
}

/**
 * Get files for a specific model and method
 */
export async function getModelMethodFiles(
  model: string, 
  method: string, 
  includeDetails: boolean = false
): Promise<ModelMethodFile> {
  return fetchAPI<ModelMethodFile>(
    `/api/models/${model}/methods/${method}/files?include_details=${includeDetails}`
  );
}

/**
 * Get available aggregation methods
 */
export async function getAggregationMethods(): Promise<AggregationOptions> {
  return fetchAPI<AggregationOptions>('/api/attribution/aggregation_methods');
}

/**
 * Get attribution data for a model, method, and file
 */
export async function getAttribution(
  model: string, 
  method: string, 
  fileId: number, 
  aggregation: AggregationMethod = 'sum'
): Promise<AttributionResponse> {
  return fetchAPI<AttributionResponse>(
    `/api/attribution/${model}/${method}/${fileId}?aggregation=${aggregation}`
  );
}

/**
 * Get detailed attribution data including token IDs and tensor information
 */
export async function getDetailedAttribution(
  model: string, 
  method: string, 
  fileId: number, 
  aggregation: AggregationMethod = 'sum'
): Promise<AttributionDetailedResponse> {
  return fetchAPI<AttributionDetailedResponse>(
    `/api/attribution/${model}/${method}/${fileId}/detailed?aggregation=${aggregation}`
  );
}

/**
 * Get token importance across all files for a model and method
 */
export async function getTokenImportanceAcrossFiles(
  model: string,
  method: string,
  aggregation: AggregationMethod = 'sum'
): Promise<TokenImportanceData[]> {
  try {
    return fetchAPI<TokenImportanceData[]>(
      `/api/token-importance/${model}/${method}?aggregation=${aggregation}`
    );
  } catch (error) {
    console.error('Error fetching token importance across files:', error);
    return [];
  }
}

/**
 * Process timing results to add hardware identifiers
 */
export function processTimingResults(results: MethodTimingResult[]): MethodTimingResult[] {
  return results.map(result => {
    // Create a hardware identifier based on CPU model and GPU info
    const hasGpu = result.cuda_available && result.gpu_info;
    const hardwareId = hasGpu ? `GPU: ${result.gpu_info}` : `CPU: ${result.cpu_model}`;
    
    return {
      ...result,
      hardware_id: hardwareId
    } as MethodTimingResult;
  });
}