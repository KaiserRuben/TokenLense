import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"
import { TokenData, AttributionResponse, AnalysisResult } from './types';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Clean special tokens and formats from token text
 */
export function cleanTokenText(tokenText: string): string {
  // Remove special token markers
  let cleaned = tokenText
    .replace(/^<.*>$/, '') // Remove tokens like <s>, </s>
    .replace(/\u0120/g, ' ') // Replace U+0120 (common in tokenizers) with space
    .replace('<|eot_id|>', '') // Remove EOT marker
    .replace(/\u010a/g, '\n'); // Replace U+010A with newline
  
  // Remove empty tokens
  if (cleaned.trim() === '') {
    return cleaned;
  }
  
  return cleaned;
}

/**
 * Clean system tokens that shouldn't be displayed
 */
// export function cleanSystemTokens(tokens: TokenData[], startIndex = 0, endIndex?: number): string {
//   const SYSTEM_TOKENS = new Set([
//     "<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>",
//     "<|eot_id|>", "system"
//   ]);
//
//   return tokens
//     .slice(startIndex, endIndex)
//     .filter(t => !SYSTEM_TOKENS.has(t.clean_token))
//     .map(t => {
//       if (t.token === 'assistant' && t.clean_token === 'assistant') {
//         return "\nAssistant: ";
//       }
//       if (t.token === 'user' && t.clean_token === 'user') {
//         return "\nUser: ";
//       }
//       return t.clean_token;
//     })
//     .filter(t => t.trim() !== '') // Remove empty tokens
//     .join(' ')
//     .trim();
// }

/**
 * Calculate token importance based on attribution matrix
 */
export function calculateTokenImportance(
  sourceTokens: TokenData[] | string[],
  attributionMatrix: number[][]
): number[] {
  if (!attributionMatrix.length || !sourceTokens.length) {
    return [];
  }
  
  // Calculate importance for input tokens (columns in the matrix)
  const inputImportance = Array(sourceTokens.length).fill(0);
  
  attributionMatrix.forEach(row => {
    row.forEach((value, colIdx) => {
      if (colIdx < sourceTokens.length) {
        inputImportance[colIdx] += value;
      }
    });
  });
  
  // Calculate importance for output tokens (rows in the matrix)
  const outputImportance = attributionMatrix.map(row => 
    row.reduce((sum, val) => sum + val, 0)
  );
  
  // Combine both importance arrays
  const combinedImportance = [...inputImportance, ...outputImportance];
  
  // Normalize to [0,1] range
  const maxImportance = Math.max(...combinedImportance, 0.0001);
  return combinedImportance.map(val => val / maxImportance);
}

/**
 * Convert API attribution data to the format expected by visualization components
 */
export function convertAttributionToAnalysisResult(data: AttributionResponse): AnalysisResult {
  // Convert source tokens to TokenData
  const inputTokens = data.source_tokens.map((token, index) => ({
    token,
    token_id: index,
    clean_token: cleanTokenText(token)
  }));
  
  // Convert target tokens to TokenData
  const outputTokens = data.target_tokens.map((token, index) => ({
    token,
    token_id: index,
    clean_token: cleanTokenText(token)
  }));
  
  // Create normalized association matrix
  const normalizedMatrix = data.attribution_matrix.map(row => {
    const maxVal = Math.max(...row.filter(v => v > 0), 0.0001);
    return row.map(val => val / maxVal);
  });
  
  // Create the input preview
  const inputPreview = inputTokens
    .map(t => t.clean_token)
    .filter(t => t.trim() !== '')
    .join(' ')
    .trim();
  
  return {
    metadata: {
      timestamp: new Date().toISOString(),
      llm_id: data.model,
      llm_version: data.model,
      prompt: data.prompt,
      generation_params: {
        max_new_tokens: outputTokens.length
      },
      version: '1.0'
    },
    data: {
      input_tokens: inputTokens,
      output_tokens: outputTokens,
      association_matrix: data.attribution_matrix,
      normalized_association: normalizedMatrix,
      input_preview: inputPreview
    }
  };
}

/**
 * Find top influences for a token based on attribution matrix
 */
export function findTopInfluences(
  tokenIndex: number,
  inputTokensLength: number,
  attributionMatrix: number[][],
  maxConnections: number = 3
): {index: number, value: number}[] {
  if (!attributionMatrix.length) {
    return [];
  }
  
  const isInputToken = tokenIndex < inputTokensLength;
  let influences: {index: number, value: number}[] = [];
  
  if (isInputToken) {
    // Find output tokens influenced by this input token
    influences = attributionMatrix.map((row, outputIdx) => ({
      index: inputTokensLength + outputIdx,
      value: row[tokenIndex]
    }));
  } else {
    // Find tokens that influenced this output token
    const outputIndex = tokenIndex - inputTokensLength;
    if (outputIndex >= attributionMatrix.length) {
      return [];
    }
    
    const row = attributionMatrix[outputIndex];
    
    // Influence from input tokens
    const inputInfluences = row
      .slice(0, inputTokensLength)
      .map((value, idx) => ({
        index: idx,
        value
      }));
    
    // Influence from previous output tokens
    const prevOutputInfluences = row
      .slice(inputTokensLength, inputTokensLength + outputIndex)
      .map((value, idx) => ({
        index: inputTokensLength + idx,
        value
      }));
    
    influences = [...inputInfluences, ...prevOutputInfluences];
  }
  
  // Filter out zero values and sort by influence
  return influences
    .filter(inf => inf.value > 0)
    .sort((a, b) => b.value - a.value)
    .slice(0, maxConnections);
}

export function formatTimestamp(timestamp: string) {
  // Extract year, month, and day
  const year = timestamp.substring(0, 4);
  const month = timestamp.substring(4, 6);
  const day = timestamp.substring(6, 8);

  // Create a valid date string in format YYYY-MM-DD
  const dateString = `${year}-${month}-${day}`;

  // Create a new Date object
  const date = new Date(dateString);

  // Format the date as a localized string
  return date.toDateString();
}
