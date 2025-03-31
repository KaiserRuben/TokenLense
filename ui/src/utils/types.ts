// src/utils/types.ts
import { CompressedMatrix } from './matrixUtils';

/**
 * Represents a single token and its properties
 */
export interface TokenData {
    /** Raw token as it appears in the model's vocabulary */
    token: string;
    /** Unique identifier for this token in the model's vocabulary */
    token_id: number;
    /** Cleaned version of the token (whitespace and special chars removed) */
    clean_token: string;
    /** Computed importance of this token (added during processing) */
    calculatedImportance?: number;
    /** Flag indicating if this is a system token */
    isSystem?: boolean;
}

/**
 * Represents metadata about the analysis run
 */
export interface AnalysisMetadata {
    /** Timestamp of the analysis */
    timestamp: string;
    /** Full model identifier (e.g. "meta-llama/Meta-Llama-3.1-8B-Instruct") */
    llm_id: string;
    /** Short model version (e.g. "Meta-Llama-3.1-8B-Instruct") */
    llm_version: string;
    /** The complete prompt that was used, including system and user messages */
    prompt: string;
    /** Parameters used for generation */
    generation_params: {
        max_new_tokens: number;
        [key: string]: unknown;
    };
    version: string;
}

/**
 * Contains the actual token association analysis
 */
export interface AnalysisData {
    input_preview: string;
    /** All tokens from the input sequence (including system & formatting tokens) */
    input_tokens: TokenData[];
    /** All tokens that were generated as output */
    output_tokens: TokenData[];
    /**
     * The association matrix showing influence of each token on generation
     */
    association_matrix: number[][] | CompressedMatrix;
    /**
     * Normalized version of the association matrix
     * Same structure as association_matrix but with values scaled to [0,1]
     */
    normalized_association: number[][] | CompressedMatrix;
}

export interface AnalysisResult {
    metadata: AnalysisMetadata;
    data: AnalysisData;
}

export interface ProcessedTokens {
    input_tokens: TokenData[];
    output_tokens: TokenData[];
}

export type ImportanceMethod = 'association' | 'weighted' | 'centrality' | 'positional' | 'comprehensive';

export interface TokenImportanceWeights {
    association: number;
    centrality: number;
    positional: number;
}

export interface TokenImportanceSettings {
    method: ImportanceMethod;
    weights: TokenImportanceWeights;
    showSystemTokens: boolean;
    maxConnections: number;
    highlightThreshold: number;
}

export interface Connection {
    path: string;
    opacity: number;
    strokeWidth: number;
}

export interface TokenConnection {
    token: TokenData;
    strength: number;
    isInput?: boolean;
}

export interface TokenConnections {
    influencers: TokenConnection[];
    influenced: TokenConnection[];
}

// Default settings
export const DEFAULT_IMPORTANCE_SETTINGS: TokenImportanceSettings = {
    method: 'comprehensive',
    weights: {
        association: 0.5,
        centrality: 0.3,
        positional: 0.2,
    },
    showSystemTokens: false,
    maxConnections: 5,
    highlightThreshold: 0.15
};

// System tokens to filter
export const SYSTEM_TOKENS = [
    "<|begin_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "system",
    "user",
    "assistant",
    "Ċ",
    "ĊĊ"
];