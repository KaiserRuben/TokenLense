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
  MethodPerformanceData,
  PromptTimingResult,
  TokensPerSecondData,
  HardwareComparisonData,
  TokenImportanceData
} from './types';

// Base API URL - in a real environment, this would come from environment variables
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

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
  return fetchAPI<SystemInfo>('/performance/system');
}

/**
 * Get timing results
 */
export async function getTimingResults(): Promise<TimingResults> {
  return fetchAPI<TimingResults>('/performance/timing');
}

/**
 * Get processed performance data (for performance page)
 */
export async function getPerformanceData(): Promise<PerformancePageData> {
  const data = await getTimingResults();
  
  // Process data for different visualizations
  const methodPerformance = processMethodPerformance(data.method_timing);
  const tokensPerSecond = processTokensPerSecond(data.prompt_timing);
  const hardwareComparison = processHardwareComparison(data.method_timing);

  return {
    methodPerformance,
    tokensPerSecond,
    hardwareComparison,
    rawData: {
      methodTiming: data.method_timing,
      promptTiming: data.prompt_timing
    }
  };
}

/**
 * Process method timing results for performance visualization
 */
function processMethodPerformance(data: MethodTimingResult[]): MethodPerformanceData[] {
  return data.map(item => {
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
}

/**
 * Get list of available models
 */
export async function getModels(): Promise<string[]> {
  const response = await fetchAPI<ModelInfo>('/models/');
  return response.models;
}

/**
 * Get attribution methods for a specific model
 */
export async function getModelMethods(model: string): Promise<string[]> {
  const response = await fetchAPI<ModelMethods>(`/models/${model}/methods`);
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
    `/models/${model}/methods/${method}/files?include_details=${includeDetails}`
  );
}

/**
 * Get available aggregation methods
 */
export async function getAggregationMethods(): Promise<AggregationOptions> {
  return fetchAPI<AggregationOptions>('/attribution/aggregation_methods');
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
    `/attribution/${model}/${method}/${fileId}?aggregation=${aggregation}`
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
    `/attribution/${model}/${method}/${fileId}/detailed?aggregation=${aggregation}`
  );
}

/**
 * Get token importance across all files for a model and method
 * This function works on the client side by aggregating data from existing endpoints
 */
export async function getTokenImportanceAcrossFiles(
  model: string,
  method: string,
  aggregation: AggregationMethod = 'sum'
): Promise<TokenImportanceData[]> {
  try {
    // First get all files for this model/method
    const filesData = await getModelMethodFiles(model, method);
    
    if (!filesData.files || filesData.files.length === 0) {
      return [];
    }
    
    // Use file indices instead of actual file IDs - this matches the way files are accessed
    // in the model pages (0, 1, 2...) rather than using the actual file IDs
    const fileIndices = Array.from({ length: filesData.files.length }, (_, index) => index);
    
    // Get attributions for each file by index, not by ID
    const attributionPromises = fileIndices.map(index => 
      getAttribution(model, method, index, aggregation)
        .catch(error => {
          console.error(`Error fetching attribution for file index ${index}:`, error);
          return null;
        })
    );
    
    const attributions = (await Promise.all(attributionPromises)).filter(Boolean) as AttributionResponse[];
    
    if (attributions.length === 0) {
      return [];
    }
    
    // Process all tokens and their importance across files
    const tokenMap = new Map<string, { 
      token: string, 
      importances: Record<string, number>,
      totalImportance: number 
    }>();
    
    attributions.forEach((attribution, index) => {
      if (!attribution) return;
      
      const fileId = attribution.file_id.toString();
      const sourceTokens = attribution.source_tokens;
      const matrix = attribution.attribution_matrix;
      
      // Calculate importance for input tokens using the same logic as in calculateTokenImportance
      const inputImportance = Array(sourceTokens.length).fill(0);
      
      matrix.forEach(row => {
        row.forEach((value, colIdx) => {
          if (colIdx < sourceTokens.length) {
            inputImportance[colIdx] += value;
          }
        });
      });
      
      // Add to token map
      sourceTokens.forEach((token, idx) => {
        if (!token.trim()) return; // Skip empty tokens
        
        const importance = inputImportance[idx];
        if (!tokenMap.has(token)) {
          tokenMap.set(token, { 
            token, 
            importances: {}, 
            totalImportance: 0 
          });
        }
        
        const tokenData = tokenMap.get(token)!;
        tokenData.importances[fileId] = importance;
        tokenData.totalImportance += importance;
      });
    });
    
    // Convert map to array and sort by importance
    const result = Array.from(tokenMap.values())
      .map(({ token, importances, totalImportance }) => ({
        token,
        token_index: -1, // Not applicable for aggregated view
        files: Object.keys(importances),
        importances,
        importance: totalImportance
      }))
      .sort((a, b) => b.importance - a.importance);
    
    // Normalize importance values to [0,1]
    const maxImportance = Math.max(...result.map(item => item.importance), 0.0001);
    return result.map(item => ({
      ...item,
      importance: item.importance / maxImportance
    }));
  } catch (error) {
    console.error('Error fetching token importance across files:', error);
    return [];
  }
}

// /**
//  * Get raw attribution data (for debugging)
//  */
// export async function getRawAttribution(
//   model: string,
//   method: string,
//   fileId: number
// ): Promise<Record<string, any>> {
//   return fetchAPI<Record<string, any>>(
//     `/attribution/${model}/${method}/${fileId}/raw`
//   );
// }

// /**
//  * Compare attribution data across multiple files
//  */
// export async function compareAttributions(
//   files: string[],
//   aggregation: AggregationMethod = 'sum'
// ): Promise<Record<string, any>> {
//   const params = new URLSearchParams();
//   files.forEach(file => params.append('files', file));
//   params.append('aggregation', aggregation);
//
//   return fetchAPI<Record<string, any>>(
//     `/attribution/compare?${params.toString()}`
//   );
// }
//
// /**
//  * Get token importance across multiple attribution files
//  */
// export async function getTokenImportance(
//   files: string[],
//   tokenIndex: number,
//   isTarget: boolean = true,
//   aggregation: AggregationMethod = 'sum'
// ): Promise<Record<string, any>> {
//   const params = new URLSearchParams();
//   files.forEach(file => params.append('files', file));
//   params.append('token_index', tokenIndex.toString());
//   params.append('is_target', isTarget.toString());
//   params.append('aggregation', aggregation);
//
//   return fetchAPI<Record<string, any>>(
//     `/attribution/token_importance?${params.toString()}`
//   );
// }

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

// /**
//  * Helper function to transform API attribution data to the format expected by visualization components
//  */
// export function transformAttributionData(data: AttributionResponse): any {
//   // Implementation will depend on the exact requirements of the visualization components
//   // This is a placeholder for the transformation logic
//   return {
//     data: {
//       input_tokens: data.source_tokens.map((token, id) => ({
//         token,
//         token_id: id,
//         clean_token: token
//       })),
//       output_tokens: data.target_tokens.map((token, id) => ({
//         token,
//         token_id: id,
//         clean_token: token
//       })),
//       association_matrix: data.attribution_matrix,
//       normalized_association: normalizeMatrix(data.attribution_matrix)
//     },
//     metadata: {
//       llm_version: data.model,
//       timestamp: new Date().toISOString()
//     }
//   };
// }

/**
 * Helper function to normalize an attribution matrix
 */
function normalizeMatrix(matrix: number[][]): number[][] {
  if (!matrix.length) return [];
  
  const normalized = matrix.map(row => {
    const max = Math.max(...row.filter(v => v > 0), 0.0001);
    return row.map(val => val / max);
  });
  
  return normalized;
}