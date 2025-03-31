// src/utils/matrixUtils.ts
import { AnalysisResult } from './types';

export interface CompressedMatrix {
    values: number[];
    indices: number[];
    shape: [number, number];
    isCompressed: true; // Type discriminator
}

export type Matrix = number[][] | CompressedMatrix;

/**
 * Determines if a matrix should be compressed based on sparsity
 */
export function shouldCompress(matrix: number[][]): boolean {
    if (!matrix.length || !matrix[0].length) return false;

    let nonZeroCount = 0;
    const totalElements = matrix.length * matrix[0].length;

    for (const row of matrix) {
        for (const value of row) {
            if (Math.abs(value) > 1e-6) nonZeroCount++;
        }
    }

    const sparsity = nonZeroCount / totalElements;
    return sparsity < 0.5 && totalElements > 100;
}

/**
 * Converts a regular matrix to compressed format
 */
function createCompressedMatrix(matrix: number[][]): CompressedMatrix {
    const rows = matrix.length;
    const cols = matrix[0]?.length ?? 0;
    const values: number[] = [];
    const indices: number[] = [];

    matrix.forEach((row, i) => {
        row.forEach((value, j) => {
            if (Math.abs(value) > 1e-4) {
                values.push(Number(value.toFixed(4)));
                indices.push((i << 16) | j);
            }
        });
    });

    return {
        values,
        indices,
        shape: [rows, cols],
        isCompressed: true as const
    };
}

/**
 * Compresses a matrix if it meets the criteria for compression
 */
export function compressMatrix(matrix: number[][]): Matrix {
    if (!shouldCompress(matrix)) {
        return matrix;
    }

    const compressed = createCompressedMatrix(matrix);

    // Verify compression is actually beneficial
    const originalSize = JSON.stringify(matrix).length;
    const compressedSize = JSON.stringify(compressed).length;

    return compressedSize < originalSize ? compressed : matrix;
}

/**
 * Decompresses a matrix from compressed format to regular 2D array
 */
export function decompressMatrix(matrix: Matrix): number[][] {
    // If it's already a regular matrix, return as is
    if (Array.isArray(matrix)) {
        return matrix;
    }

    const [rows, cols] = matrix.shape;
    const result = Array(rows).fill(0).map(() => Array(cols).fill(0));

    for (let i = 0; i < matrix.values.length; i++) {
        const packed = matrix.indices[i];
        const row = packed >> 16;
        const col = packed & 0xFFFF;
        result[row][col] = matrix.values[i];
    }

    return result;
}

/**
 * Type guard to check if a value is a compressed matrix
 */
export function isCompressedMatrix(value: unknown): value is CompressedMatrix {
    return Boolean(
        value &&
        typeof value === 'object' &&
        'isCompressed' in value &&
        (value as CompressedMatrix).isCompressed === true
    );
}

/**
 * Compresses the matrices in an analysis result
 */
export function compressAnalysis(result: AnalysisResult): AnalysisResult {
    if (!result.data.association_matrix || !Array.isArray(result.data.association_matrix) ||
        !result.data.association_matrix.length) return result;

    const compressedMain = compressMatrix(result.data.association_matrix);

    const compressedNorm = result.data.normalized_association && Array.isArray(result.data.normalized_association)
        ? compressMatrix(result.data.normalized_association)
        : result.data.normalized_association;

    return {
        ...result,
        data: {
            ...result.data,
            association_matrix: compressedMain as number[][],
            normalized_association: compressedNorm as number[][]
        }
    };
}

/**
 * Decompresses the matrices in an analysis result
 */
export function decompressAnalysis(result: AnalysisResult): AnalysisResult {
    // Handle both association_matrix and normalized_association
    const association_matrix = result.data.association_matrix
        ? decompressMatrix(result.data.association_matrix as Matrix)
        : [];

    const normalized_association = result.data.normalized_association
        ? decompressMatrix(result.data.normalized_association as Matrix)
        : association_matrix;

    return {
        ...result,
        data: {
            ...result.data,
            association_matrix,
            normalized_association
        }
    };
}

/**
 * Ensures that an analysis result has decompressed matrices
 */
export function ensureDecompressedAnalysis(analysis: AnalysisResult): AnalysisResult {
    if (isCompressedMatrix(analysis.data.association_matrix) ||
        isCompressedMatrix(analysis.data.normalized_association)) {
        return decompressAnalysis(analysis);
    }
    return analysis;
}