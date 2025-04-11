import { SystemInfo, TimingResults, MethodTimingResult } from './types';

// Base API URL - in a real environment, this would come from environment variables
const API_BASE_URL = 'http://localhost:8000';

/**
 * Function to fetch from the API with error handling
 */
async function fetchAPI<T>(endpoint: string): Promise<T> {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`);
    
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
  return fetchAPI<SystemInfo>('/system_info');
}

/**
 * Get timing results
 */
export async function getTimingResults(): Promise<TimingResults> {
  return fetchAPI<TimingResults>('/timing_results');
}

/**
 * Get list of available models
 */
export async function getModels(): Promise<string[]> {
  const response = await fetchAPI<{ models: string[] }>('/models');
  return response.models;
}

/**
 * Get attribution methods for a specific model
 */
export async function getModelMethods(model: string): Promise<string[]> {
  const response = await fetchAPI<{ model: string; methods: string[] }>(`/models/${model}/methods`);
  return response.methods;
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