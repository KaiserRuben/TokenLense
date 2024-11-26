import {compressAnalysis} from "./utils/matrixUtils";
import {AnalysisResult} from "./utils/data";

async function compressDataFiles() {
    const glob = new Bun.Glob("*.json");
    const files = await Array.fromAsync(glob.scan("./src/data"));

    let totalSaved = 0;
    let totalSizeOld = 0;
    let totalSizeNew = 0;
    const results: string[] = [];

    for (const filepath of files) {
        if (filepath.includes('.compressed.')) continue;

        try {
            // Read and parse the file
            const data = await Bun.file(`./src/data/${filepath}`).json() as AnalysisResult;

            // Calculate matrix stats before compression
            const mainMatrixSize = data.data.association_matrix.length *
                (data.data.association_matrix[0]?.length || 0);
            const normMatrixSize = data.data.normalized_association?.length ?
                (data.data.normalized_association.length *
                    data.data.normalized_association[0].length) : 0;

            // Compress
            const compressed = compressAnalysis(data);

            // Create new filename
            const newPath = filepath.replace('.json', '.compressed.json');

            // Write compressed file
            await Bun.write(`./src/data/${newPath}`, JSON.stringify(compressed));

            // Calculate stats
            const originalSize = new TextEncoder().encode(JSON.stringify(data)).length;
            const compressedSize = new TextEncoder().encode(JSON.stringify(compressed)).length;
            const savedBytes = originalSize - compressedSize;
            const ratio = ((1 - compressedSize / originalSize) * 100).toFixed(1);

            // Count non-zero elements in original matrix
            let nonZeroCount = 0;
            data.data.association_matrix.forEach(row =>
                row.forEach(val => {
                    if (Math.abs(val) > 1e-6) nonZeroCount++;
                })
            );
            const sparsity = (nonZeroCount / mainMatrixSize * 100).toFixed(1);

            totalSaved += savedBytes;
            totalSizeOld += originalSize;
            totalSizeNew += compressedSize;
            results.push(
                `${filepath}:\n` +
                `  Matrix size: ${mainMatrixSize} elements (${sparsity}% non-zero)\n` +
                `  Norm matrix: ${normMatrixSize} elements\n` +
                `  File size: ${(originalSize/1024).toFixed(1)}KB → ` +
                `${(compressedSize/1024).toFixed(1)}KB (${ratio}% reduction)\n`
            );
        } catch (e) {
            results.push(`❌ Error compressing ${filepath}: ${e}`);
        }
    }

    // Print summary
    console.log("\nCompression Results:");
    console.log("=".repeat(50));
    results.forEach(result => console.log(result));
    console.log("=".repeat(50));
    console.log("Total space old: " + (totalSizeOld/1024/1024).toFixed(2) + "MB");
    console.log("Total space new: " + (totalSizeNew/1024/1024).toFixed(2) + "MB");
    console.log(`Total space saved: ${(totalSaved/1024/1024).toFixed(2)}MB (${((1 - totalSizeNew/totalSizeOld) * 100).toFixed(1)}% reduction) `);
}

console.log("🚀 Starting compression...");
compressDataFiles()
    .then(() => console.log("✨ Compression complete"))
    .catch(console.error);