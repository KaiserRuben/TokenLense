interface TokenData {
    token: string;
    token_id: number;
    clean_token: string;
}

interface AnalysisMetadata {
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

interface AssociationData {
    input_tokens: TokenData[];
    output_tokens: TokenData[];
    association_matrix: number[][];
    normalized_association?: number[][];
    input_preview: string; // Added this field
}

export interface AnalysisResult {
    metadata: AnalysisMetadata;
    data: AssociationData;
}

const SYSTEM_TOKENS = new Set([
    "<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>",
    "<|eot_id|>", "system", "user", "assistant"
]);

const getInputPreview = (tokens: Array<{ clean_token: string; token: string }>, maxLength = 150) => {
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

    const filteredTokens = tokens
        .slice(startIndex, endIndex)
        .filter(t => !SYSTEM_TOKENS.has(t.clean_token))
        .map(t => t.clean_token)
        .filter(t => t.trim() !== '') // Remove empty tokens
        .join(' ')
        .trim();

    return filteredTokens.length > maxLength
        ? `${filteredTokens.slice(0, maxLength)}...`
        : filteredTokens;
};
// Validates a single AnalysisResult
const validateAnalysisResult = (data: unknown): data is AnalysisResult => {
    try {
        // Check basic structure
        if (!data || typeof data !== 'object' || !('metadata' in data) || !('data' in data)) return false;

        // Validate metadata
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

        // Validate data
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
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
    } catch (e) {
        return false;
    }
};

/**
 * Loads and validates all analysis result JSONs from the data directory
 * @returns Promise containing array of valid analysis results
 * @throws Error if no valid results are found or if loading fails
 */
export const getAnalysisResults = async (): Promise<AnalysisResult[]> => {
    const results: AnalysisResult[] = [];
    const errors: string[] = [];

    try {
        // Load all JSON files from the data directory
        const modules = import.meta.glob('/src/data/*.json', { eager: true });

        Object.entries(modules).forEach(([path, module]) => {
            try {
                // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                // @ts-expect-error
                const data = module.default || module; // Handle both ESM and CJS modules
                if (validateAnalysisResult(data)) {
                    // Add input preview to the data
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
        });

        if (results.length === 0) {
            throw new Error(`No valid analysis results found. ${errors.length > 0 ? `Errors: ${errors.join(', ')}` : ''}`);
        }

        return results;
    } catch (e) {
        throw new Error(`Failed to load analysis results: ${e instanceof Error ? e.message : 'Unknown error'}`);
    }
};