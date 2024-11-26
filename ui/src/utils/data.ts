import {CompressedMatrix, decompressAnalysis} from "@/utils/matrixUtils.ts";

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
}

/**
 * Represents metadata about the analysis run
 */
interface AnalysisMetadata {
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
interface AssociationData {
    input_preview: string;
    /** All tokens from the input sequence (including system & formatting tokens) */
    input_tokens: TokenData[];

    /** All tokens that were generated as output */
    output_tokens: TokenData[];

    /**
     * The association matrix showing influence of each token on generation
     *
     * Matrix dimensions are [output_tokens.length][max_width]
     * where max_width = input_tokens.length + output_tokens.length
     *
     * For each row i (generating output_token[i]):
     * - First input_tokens.length columns: influences of input tokens
     * - Next i columns: influences of previously generated tokens
     * - Remaining columns: padded with zeros
     *
     * Example for input "A B C" generating "X Y":
     * max_width = 3 + 2 - 1 = 4 columns
     * [
     *   // Generating "X": looks at ["A", "B", "C"]
     *   [0.1, 0.2, 0.3, 0.0, 0.0],  // last value is padding, plus one value 0 cause x has 0 self-attention
     *
     *   // Generating "Y": looks at ["A", "B", "C", "X"]
     *   [0.2, 0.1, 0.4, 0.5, 0.0]
     * ]
     *
     * Values:
     * - Higher values indicate stronger influence
     * - Values are gradient-based importance scores
     * - Padded values are always 0
     * - Each row i contains i meaningful values after input_tokens.length
     */
    association_matrix: number[][];

    /**
     * Optional normalized version of the association matrix
     * Same structure as association_matrix but with values scaled to [0,1]
     * Padded values remain 0
     */
    normalized_association: number[][];
}

export interface AnalysisResult {
    metadata: AnalysisMetadata;
    data: AssociationData;
}

const SYSTEM_TOKENS = new Set([
    "<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>",
    "<|eot_id|>", "system"
]);

export function cleanSystemTokens(tokens: TokenData[], startIndex: number = 0, endIndex: number = tokens.length) {
    return tokens
        .slice(startIndex, endIndex)
        .filter(t => !SYSTEM_TOKENS.has(t.clean_token))
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

export const getInputPreview = (tokens: TokenData[], maxLength = 150) => {
    // Find the user section
    let startIndex = -1;
    for (let i = 0; i < tokens.length; i++) {
        if (tokens[i].clean_token === "user") {
            // Find the following end_header_id
            const endHeaderIndex = tokens.slice(i).findIndex(t => t.clean_token === "<|end_header_id|>");
            if (endHeaderIndex !== -1) {
                // Find the following newline
                const newlineIndex = tokens.slice(i + endHeaderIndex).findIndex(t => t.token === "\u010a");
                if (newlineIndex !== -1) {
                    startIndex = i + endHeaderIndex + newlineIndex + 1;
                    break;
                }
            }
        }
    }

    // If we couldn't find the right starting point, start from beginning
    startIndex = startIndex === -1 ? 0 : startIndex;

    // Find the end index (before the next header or EOT)
    let endIndex = tokens.slice(startIndex).findIndex(t =>
        t.clean_token === "<|start_header_id|>" ||
        t.clean_token === "<|eot_id|>"
    );
    endIndex = endIndex === -1 ? tokens.length : startIndex + endIndex;
    const filteredTokens = cleanSystemTokens(tokens, startIndex, endIndex);

    return filteredTokens.length > maxLength
        ? `${filteredTokens.slice(0, maxLength)}...`
        : filteredTokens;
};
// Validates a single AnalysisResult
// Add this helper to detect compressed format
const isCompressedMatrix = (data: CompressedMatrix): boolean => {
    return data &&
        Array.isArray(data.values) &&
        Array.isArray(data.indices) &&
        Array.isArray(data.shape) &&
        data.shape.length === 2;
};

// Modify your existing validation to handle compressed format
export const validateAnalysisResult = (data: unknown): data is AnalysisResult => {
    try {
        if (!data || typeof data !== 'object') return false;

        // Check if it's compressed and needs decompression
        const anyData = data as unknown as { data: { association_matrix: CompressedMatrix } };
        if (isCompressedMatrix(anyData.data?.association_matrix)) {
            data = decompressAnalysis(data as AnalysisResult);
        }

        // Rest of your existing validation...
        const metadata = (data as AnalysisResult).metadata;
        if (
            typeof metadata.timestamp !== 'string' ||
            typeof metadata.llm_id !== 'string' ||
            typeof metadata.llm_version !== 'string' ||
            typeof metadata.prompt !== 'string' ||
            typeof metadata.version !== 'string' ||
            typeof metadata.generation_params !== 'object' ||
            typeof metadata.generation_params.max_new_tokens !== 'number'
        ) return false;

        // Validate data structure
        const associationData = (data as AnalysisResult).data;
        if (
            !Array.isArray(associationData.input_tokens) ||
            !Array.isArray(associationData.output_tokens) ||
            !Array.isArray(associationData.association_matrix)
        ) return false;

        // Validate token structure
        const validateToken = (token: unknown): boolean => {
            if (!token || typeof token !== 'object') return false;
            const t = token as TokenData;
            return (
                typeof t.token === 'string' &&
                typeof t.token_id === 'number' &&
                typeof t.clean_token === 'string'
            );
        };

        if (!associationData.input_tokens.every(validateToken) ||
            !associationData.output_tokens.every(validateToken)) {
            return false;
        }

        // Validate matrix structure
        const maxWidth = associationData.input_tokens.length + associationData.output_tokens.length;

        if (!associationData.association_matrix.every((row: unknown) =>
            Array.isArray(row) && row.length === maxWidth &&
            row.every((val: unknown) => typeof val === 'number')
        )) {
            return false;
        }

        return true;
    } catch {
        return false;
    }
};

interface GetAnalysisParams {
    offset: number;
    limit: number;
}

/**
 * Loads and validates all analysis result JSONs from the data directory
 * @returns Promise containing array of valid analysis results
 * @throws Error if no valid results are found or if loading fails
 */
export const getAnalysisResults = async ({offset, limit}: GetAnalysisParams): Promise<AnalysisResult[]> => {
    const results: AnalysisResult[] = [];
    const errors: string[] = [];

    try {
// For compressed files
        const compressedModules = import.meta.glob('/src/data/*.compressed.json', {
            eager: true
        });

// For regular JSON files
        const regularModules = import.meta.glob('/src/data/!(*.compressed).json', {
            eager: true
        });
// In production, only use compressed files
        const modules = import.meta.env.PROD ? compressedModules : regularModules;

        const moduleEntries = Object.entries(modules)
            .slice(offset, offset + limit);

        for (const [path, module] of moduleEntries) {
            try {
                // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                // @ts-expect-error
                const data = module.default || module;
                if (validateAnalysisResult(data)) {
                    const enrichedData = {
                        ...data,
                        data: {
                            ...data.data,
                            input_preview: getInputPreview(data.data.input_tokens)
                        }
                    };
                    results.push(enrichedData);
                } else {
                    errors.push(`Invalid data structure in ${path}`);
                }
            } catch (e) {
                errors.push(`Error loading ${path}: ${e instanceof Error ? e.message : 'Unknown error'}`);
            }
        }

        if (results.length === 0 && offset === 0) {
            throw new Error(`No valid analysis results found. ${errors.length > 0 ? `Errors: ${errors.join(', ')}` : ''}`);
        }

        return results;
    } catch (e) {
        throw new Error(`Failed to load analysis results: ${e instanceof Error ? e.message : 'Unknown error'}`);
    }
};
