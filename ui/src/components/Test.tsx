import { useState, useEffect } from 'react';
import {AnalysisResult, getAnalysisResults} from "../utils/data.ts";

const YourComponent = () => {
    const [results, setResults] = useState<AnalysisResult[]>([]);
    const [error, setError] = useState<string>();

    useEffect(() => {
        getAnalysisResults()
            .then(setResults)
            .catch(error => setError(error.message));
    }, []);

    if (error) {
        return <div>Error: {error}</div>;
    }

    return (
        <div>
            {results.map((result, index) => (
                <div key={index}>
                    {/* Use your result data */}
                    <h2>{result.metadata.llm_version}</h2>
                    {/* ... */}
                </div>
            ))}
        </div>
    );
};

export default YourComponent;