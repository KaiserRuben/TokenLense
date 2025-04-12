import { NextResponse } from 'next/server';
import { AttributionResponse, TokenImportanceData } from '@/lib/types';

// Base API URL from server environment variable
const API_BASE_URL = process.env.API_URL || 'http://localhost:8000';

/**
 * API route handler for /api/token-importance/[model]/[method]
 * Gets token importance across all files for a model and method
 */
export async function GET(
  request: Request,
  { params }: { params: Promise<{ model: string; method: string }> }
) {
  const { model, method } = await params;
  const url = new URL(request.url);
  const aggregation = url.searchParams.get('aggregation') || 'sum';
  
  try {
    // First get all files for this model/method
    const filesResponse = await fetch(
      `${API_BASE_URL}/models/${model}/methods/${method}/files`
    );
    
    if (!filesResponse.ok) {
      throw new Error(`API error: ${filesResponse.status} ${filesResponse.statusText}`);
    }
    
    const filesData = await filesResponse.json();
    
    if (!filesData.files || filesData.files.length === 0) {
      return NextResponse.json([]);
    }
    
    // Use file indices instead of actual file IDs
    const fileIndices = Array.from({ length: filesData.files.length }, (_, index) => index);
    
    // Get attributions for each file
    const attributionPromises = fileIndices.map(index => {
      return fetch(
        `${API_BASE_URL}/attribution/${model}/${method}/${index}?aggregation=${aggregation}`
      )
      .then(res => {
        if (!res.ok) {
          throw new Error(`API error: ${res.status} ${res.statusText}`);
        }
        return res.json();
      })
      .catch(error => {
        console.error(`Error fetching attribution for file index ${index}:`, error);
        return null;
      });
    });
    
    const attributions = (await Promise.all(attributionPromises)).filter(Boolean) as AttributionResponse[];
    
    if (attributions.length === 0) {
      return NextResponse.json([]);
    }
    
    // Process tokens and their importance across files
    const tokenMap = new Map<string, { 
      token: string, 
      importances: Record<string, number>,
      totalImportance: number 
    }>();
    
    attributions.forEach((attribution) => {
      if (!attribution) return;
      
      const fileId = attribution.file_id.toString();
      const sourceTokens = attribution.source_tokens;
      const matrix = attribution.attribution_matrix;
      
      // Calculate importance for input tokens
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
    const normalizedResult = result.map(item => ({
      ...item,
      importance: item.importance / maxImportance
    }));
    
    return NextResponse.json(normalizedResult);
  } catch (error) {
    console.error(`Token importance API error for ${model}/${method}:`, error);
    return NextResponse.json(
      { error: `Failed to fetch token importance data for model ${model} and method ${method}` }, 
      { status: 500 }
    );
  }
}
