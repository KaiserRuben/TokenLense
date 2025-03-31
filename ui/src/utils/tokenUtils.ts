// src/utils/tokenUtils.ts
import {
    AnalysisResult,
    ProcessedTokens,
    TokenImportanceSettings,
    SYSTEM_TOKENS,
    TokenData
} from './types';
import { ensureDecompressedAnalysis } from './matrixUtils';

/**
 * Cleans system tokens from input text for display
 */
export function cleanSystemTokens(tokens: TokenData[], startIndex: number = 0, endIndex: number = tokens.length): string {
    const systemSet = new Set(SYSTEM_TOKENS);

    return tokens
        .slice(startIndex, endIndex)
        .filter(t => !systemSet.has(t.clean_token))
        .map(t => {
            if (t.token === 'assistant' && t.clean_token === 'assistant') {
                return "\nAssistant: ";
            }
            if (t.token === 'user' && t.clean_token === 'user') {
                return "\nUser: ";
            }
            return t.clean_token;
        })
        .filter(t => t.trim() !== '') // Remove empty tokens
        .join(' ')
        .trim();
}

/**
 * Process tokens with importance calculations and system token marking
 */
export function processTokens(analysis: AnalysisResult, settings: TokenImportanceSettings): ProcessedTokens {
    // Ensure we're working with decompressed matrices
    const decompressedAnalysis = ensureDecompressedAnalysis(analysis);

    const { input_tokens, output_tokens, normalized_association } = decompressedAnalysis.data;

    // Calculate importance based on selected method
    let importance: number[] = [];

    switch(settings.method) {
        case 'association':
            importance = calculateAssociationImportance(input_tokens, output_tokens, normalized_association);
            break;
        case 'weighted':
            importance = calculateWeightedImportance(input_tokens, output_tokens, normalized_association);
            break;
        case 'centrality':
            importance = calculateNetworkCentrality(input_tokens, output_tokens, normalized_association);
            break;
        case 'positional':
            importance = calculatePositionalImportance(input_tokens, output_tokens, normalized_association);
            break;
        case 'comprehensive':
        default:
            importance = calculateComprehensiveImportance(
                input_tokens,
                output_tokens,
                normalized_association,
                settings.weights
            );
            break;
    }

    // Mark system tokens and add calculated importance
    const systemSet = new Set(SYSTEM_TOKENS);

    const processedInput = input_tokens.map((token, idx) => ({
        ...token,
        isSystem: systemSet.has(token.clean_token),
        calculatedImportance: importance[idx]
    }));

    const processedOutput = output_tokens.map((token, idx) => ({
        ...token,
        isSystem: systemSet.has(token.clean_token),
        calculatedImportance: importance[idx + input_tokens.length]
    }));

    return {
        input_tokens: processedInput,
        output_tokens: processedOutput
    };
}

/**
 * Filter tokens based on settings
 */
export function filterTokens(processedTokens: ProcessedTokens, showSystemTokens: boolean): ProcessedTokens {
    if (showSystemTokens) {
        return processedTokens;
    }

    return {
        input_tokens: processedTokens.input_tokens.filter(token => !token.isSystem),
        output_tokens: processedTokens.output_tokens.filter(token => !token.isSystem)
    };
}

// Importance calculation methods

/**
 * Association-based importance calculation
 */
function calculateAssociationImportance(
    input_tokens: TokenData[],
    output_tokens: TokenData[],
    normalized_association: number[][]
): number[] {
    // For input tokens: sum each column of the association matrix
    const inputImportance = input_tokens.map((_, idx) => {
        return normalized_association
            .map(row => row[idx])
            .reduce((sum, val) => sum + val, 0);
    });

    // For output tokens: sum each row of the association matrix
    const outputImportance = normalized_association.map(row => {
        return row.reduce((sum, val) => sum + val, 0);
    });

    // Normalize to 0-1 scale
    const importancePerToken = [...inputImportance, ...outputImportance];
    const maxElement = Math.max(...importancePerToken) || 1; // Avoid division by zero
    return importancePerToken.map(e => e / maxElement);
}

/**
 * Weighted association calculation (emphasizes stronger connections)
 */
function calculateWeightedImportance(
    input_tokens: TokenData[],
    output_tokens: TokenData[],
    normalized_association: number[][]
): number[] {
    // Apply quadratic weighting to emphasize stronger connections
    const inputImportance = input_tokens.map((_, idx) => {
        return normalized_association
            .map(row => Math.pow(row[idx], 2)) // Square values to emphasize stronger connections
            .reduce((sum, val) => sum + val, 0);
    });

    // For output tokens
    const outputImportance = normalized_association.map(row => {
        return row.map(val => Math.pow(val, 2)).reduce((sum, val) => sum + val, 0);
    });

    const importancePerToken = [...inputImportance, ...outputImportance];
    const maxElement = Math.max(...importancePerToken) || 1;
    return importancePerToken.map(e => e / maxElement);
}

/**
 * Network centrality calculation
 */
function calculateNetworkCentrality(
    input_tokens: TokenData[],
    output_tokens: TokenData[],
    normalized_association: number[][]
): number[] {
    const totalTokens = input_tokens.length + output_tokens.length;

    // Initialize with equal importance
    let importance = new Array(totalTokens).fill(1);

    // Perform power iteration (simplified eigenvector centrality)
    for (let iteration = 0; iteration < 10; iteration++) {
        const newImportance = new Array(totalTokens).fill(0);

        // Update importance based on connections
        for (let i = 0; i < output_tokens.length; i++) {
            for (let j = 0; j < input_tokens.length + i; j++) {
                const value = normalized_association[i][j];
                // Bidirectional influence
                newImportance[j] += value * importance[input_tokens.length + i];
                newImportance[input_tokens.length + i] += value * importance[j];
            }
        }

        // Normalize
        const sum = newImportance.reduce((a, b) => a + b, 0);
        importance = newImportance.map(v => v / (sum || 1));
    }

    return importance;
}

/**
 * Position-enhanced importance calculation
 */
function calculatePositionalImportance(
    input_tokens: TokenData[],
    output_tokens: TokenData[],
    normalized_association: number[][]
): number[] {
    const baseImportance = calculateAssociationImportance(input_tokens, output_tokens, normalized_association);

    // Position factors: beginning and end tokens get boosted importance
    const positionFactors = [];

    // Input tokens positions
    for (let i = 0; i < input_tokens.length; i++) {
        // Beginning of input has higher importance
        if (i < 3) {
            positionFactors.push(1.2 - (i * 0.05));
        }
        // End of input has higher importance
        else if (i >= input_tokens.length - 3) {
            positionFactors.push(1.1 + ((i - (input_tokens.length - 3)) * 0.05));
        }
        else {
            positionFactors.push(1.0);
        }
    }

    // Output tokens positions
    for (let i = 0; i < output_tokens.length; i++) {
        const token = output_tokens[i].clean_token;

        // Boost tokens after punctuation (likely beginning of new sentences)
        const previousToken = i > 0 ? output_tokens[i-1].clean_token : "";
        const isPunctuation = previousToken && /[.!?]/.test(previousToken);

        if (i === 0 || i === output_tokens.length - 1) {
            // First and last output tokens
            positionFactors.push(1.2);
        } else if (isPunctuation) {
            positionFactors.push(1.15);
        } else {
            positionFactors.push(1.0);
        }
    }

    // Apply position factors to base importance
    return baseImportance.map((imp, idx) => Math.min(imp * positionFactors[idx], 1.0));
}

/**
 * Comprehensive importance calculation (combines multiple methods)
 */
function calculateComprehensiveImportance(
    input_tokens: TokenData[],
    output_tokens: TokenData[],
    normalized_association: number[][],
    weights: { association: number; centrality: number; positional: number; }
): number[] {
    const associationImp = calculateAssociationImportance(input_tokens, output_tokens, normalized_association);
    const centralityImp = calculateNetworkCentrality(input_tokens, output_tokens, normalized_association);
    const positionImp = calculatePositionalImportance(input_tokens, output_tokens, normalized_association);

    // Combined weighted approach
    const combinedImportance = associationImp.map((_, idx) => {
        return (weights.association * associationImp[idx] +
            weights.centrality * centralityImp[idx] +
            weights.positional * positionImp[idx]);
    });

    // Normalize
    const maxImportance = Math.max(...combinedImportance) || 1;
    return combinedImportance.map(imp => Math.min(imp / maxImportance, 1.0));
}