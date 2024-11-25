import {AnalysisResult} from "@/utils/data.ts";
import {WordConnection} from "@/components/WordCloud.tsx";

/**
 * Creates word connections from analysis data by processing the normalized association matrix
 * @param analysis - The analysis result containing tokens and their associations
 * @param minWeight - Minimum weight threshold for connections (0-1)
 * @returns Array of word connections with their weights
 */
export function createWordConnections(
    analysis: AnalysisResult,
    minWeight: number = 0.0
): WordConnection[]{
    // Filter out system tokens and get clean tokens
    const tokens = analysis.data.input_tokens.filter(token => {
        const isSystemToken = token.clean_token.startsWith('<|') ||
            ['system', 'user', 'assistant'].includes(token.clean_token);
        return !isSystemToken;
    });

    // Process normalized association matrix
    return tokens.map((token, idx) => {
        // Get weights for current token
        const weights = analysis.data.normalized_association[idx];

        // Create connections object
        const connections = weights
            .map((weight, targetIdx) => {
                // Skip self-connections and weak connections
                if (targetIdx === idx || weight < minWeight) {
                    return null;
                }

                const targetToken = tokens[targetIdx];
                // Skip system tokens in targets
                if (!targetToken || targetToken.clean_token.startsWith('<|')) {
                    return null;
                }

                return {
                    targetWord: targetToken.clean_token,
                    targetIndex: targetIdx,
                    weight
                };
            })
            .filter((conn): conn is NonNullable<typeof conn> => conn !== null);

        return {
            word: token.clean_token,
            index: idx,
            connections
        };
    });
}

/**
 * Optional: Helper function to normalize weights to a specific range
 * Useful if you want to adjust the visual strength of connections
 */
export function normalizeWeight(
    weight: number,
    minDisplay: number = 0.2,
    maxDisplay: number = 1
): number {
    return minDisplay + (weight * (maxDisplay - minDisplay));
}

// Optional: Types for the functions
export interface ProcessingOptions {
    minWeight?: number;
    excludeTokens?: string[];
    maxConnections?: number;
}

/**
 * Advanced version with more options
 */
export function createWordConnectionsAdvanced(
    analysis: AnalysisResult,
    options: ProcessingOptions = {}
):WordConnection[] {
    const {
        minWeight = 0.1,
        excludeTokens = ['<|begin_of_text|>', '<|end_of_text|>', 'system', 'user', 'assistant'],
        maxConnections = 5
    } = options;

    // Filter out system tokens and get clean tokens
    const tokens = analysis.data.input_tokens.filter(token => {
        const isExcluded = excludeTokens.includes(token.clean_token) ||
            token.clean_token.startsWith('<|');
        return !isExcluded;
    });

    return tokens.map((token, idx) => {
        const weights = analysis.data.normalized_association[idx];

        // Get top connections
        const topConnections = weights
            .map((weight, targetIdx) => ({
                targetIdx,
                weight,
            }))
            .filter(conn =>
                conn.targetIdx !== idx &&
                conn.weight >= minWeight &&
                tokens[conn.targetIdx]
            )
            .sort((a, b) => b.weight - a.weight)
            .slice(0, maxConnections)
            .map(conn => ({
                targetWord: tokens[conn.targetIdx].clean_token,
                targetIndex: conn.targetIdx,
                weight: normalizeWeight(conn.weight)
            }));

        return {
            word: token.clean_token,
            index: idx,
            connections: topConnections
        };
    });
}