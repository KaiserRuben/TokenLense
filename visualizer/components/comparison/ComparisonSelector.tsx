"use client"

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { getModels, getModelMethods, getModelMethodFiles, getAggregationMethods } from '@/lib/api';

interface ComparisonSelectorProps {
    initialFileId?: number;
}

const ComparisonSelector: React.FC<ComparisonSelectorProps> = ({ initialFileId = 0 }) => {
    const router = useRouter();

    // Comparison type
    const [comparisonType, setComparisonType] = useState<'models' | 'methods' | 'both'>('models');

    // Data states
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [models, setModels] = useState<string[]>([]);
    const [methods, setMethods] = useState<{[modelName: string]: string[]}>({});
    const [files, setFiles] = useState<{[modelName: string]: {[methodName: string]: any[]}}>({});

    // Selection states
    const [selectedModel, setSelectedModel] = useState<string>('');
    const [selectedLeftModel, setSelectedLeftModel] = useState<string>('');
    const [selectedRightModel, setSelectedRightModel] = useState<string>('');
    const [selectedMethod, setSelectedMethod] = useState<string>('');
    const [selectedLeftMethod, setSelectedLeftMethod] = useState<string>('');
    const [selectedRightMethod, setSelectedRightMethod] = useState<string>('');
    const [selectedFileId, setSelectedFileId] = useState<number>(initialFileId);
    const [leftAggregation, setLeftAggregation] = useState<string>('sum');
    const [rightAggregation, setRightAggregation] = useState<string>('sum');
    const [availableAggregations, setAvailableAggregations] = useState<string[]>([]);

    // Load models on component mount and get available aggregation methods
    useEffect(() => {
        async function loadInitialData() {
            try {
                setLoading(true);

                // Fetch both models and aggregation methods in parallel
                const [modelsList, aggregationOptions] = await Promise.all([
                    getModels(),
                    getAggregationMethods()
                ]);

                // Set models
                setModels(modelsList);
                if (modelsList.length > 0) {
                    setSelectedModel(modelsList[0]);
                    setSelectedLeftModel(modelsList[0]);
                    if (modelsList.length > 1) {
                        setSelectedRightModel(modelsList[1]);
                    } else {
                        setSelectedRightModel(modelsList[0]);
                    }
                }

                // Set aggregation methods
                const methods = aggregationOptions.methods || ['sum', 'mean', 'l2_norm', 'abs_sum', 'max'];
                setAvailableAggregations(methods);

                // Use default if provided, otherwise use the first available method
                const defaultMethod = aggregationOptions.default || methods[0] || 'sum';
                setLeftAggregation(defaultMethod);
                setRightAggregation(defaultMethod);

            } catch (err) {
                setError('Failed to load initial data. Please try again later.');
                console.error('Error loading initial data:', err);
            } finally {
                setLoading(false);
            }
        }

        loadInitialData();
    }, []);

    // Load methods when models are selected
    useEffect(() => {
        async function loadMethodsForModel(model: string) {
            if (!model) return;

            // If we already have methods for this model, skip fetching
            if (methods[model] && methods[model].length > 0) {
                return;
            }

            try {
                setLoading(true);
                console.log(`Loading methods for ${model}...`);
                const methodsList = await getModelMethods(model);

                // Update methods state
                setMethods(prev => ({
                    ...prev,
                    [model]: methodsList
                }));

                return methodsList;
            } catch (err) {
                setError(`Failed to load methods for ${model}. Please try again later.`);
                console.error('Error loading methods:', err);
                return [];
            } finally {
                setLoading(false);
            }
        }

        // Load methods for all relevant models
        async function loadAllMethods() {
            const modelsToLoad = [];

            if (comparisonType === 'models' || comparisonType === 'both') {
                if (selectedLeftModel) modelsToLoad.push(selectedLeftModel);
                if (selectedRightModel && selectedRightModel !== selectedLeftModel) {
                    modelsToLoad.push(selectedRightModel);
                }
            }

            if (comparisonType === 'methods') {
                if (selectedModel) modelsToLoad.push(selectedModel);
            }

            // Load methods for all models in parallel
            await Promise.all(modelsToLoad.map(model => loadMethodsForModel(model)));

            // Set initial method selections if needed
            if (comparisonType === 'models') {
                if (selectedLeftModel && methods[selectedLeftModel]?.length > 0 && !selectedMethod) {
                    setSelectedMethod(methods[selectedLeftModel][0]);
                }
            } else if (comparisonType === 'methods') {
                if (selectedModel && methods[selectedModel]?.length > 0) {
                    if (!selectedLeftMethod) {
                        setSelectedLeftMethod(methods[selectedModel][0]);
                    }
                    if (!selectedRightMethod && methods[selectedModel].length > 1) {
                        setSelectedRightMethod(methods[selectedModel][1]);
                    } else if (!selectedRightMethod) {
                        setSelectedRightMethod(methods[selectedModel][0]);
                    }
                }
            } else if (comparisonType === 'both') {
                if (selectedLeftModel && methods[selectedLeftModel]?.length > 0 && !selectedLeftMethod) {
                    setSelectedLeftMethod(methods[selectedLeftModel][0]);
                }
                if (selectedRightModel && methods[selectedRightModel]?.length > 0 && !selectedRightMethod) {
                    setSelectedRightMethod(methods[selectedRightModel][0]);
                }
            }
        }

        loadAllMethods();
    }, [comparisonType, selectedModel, selectedLeftModel, selectedRightModel, methods]);

    // Load files when methods are selected
    useEffect(() => {
        async function loadFiles(model: string, method: string) {
            if (!model || !method) return;

            try {
                setLoading(true);
                const fileData = await getModelMethodFiles(model, method, true);

                // ModelMethodFile already has the files property
                const filesList = fileData.files || [];

                // Update files state
                setFiles(prev => ({
                    ...prev,
                    [model]: {
                        ...(prev[model] || {}),
                        [method]: filesList
                    }
                }));
            } catch (err) {
                setError(`Failed to load files for ${model}/${method}. Please try again later.`);
                console.error('Error loading files:', err);
            } finally {
                setLoading(false);
            }
        }

        if (comparisonType === 'models') {
            // For model comparison, load files for both models with the shared method
            if (selectedMethod) {
                if (selectedLeftModel) {
                    loadFiles(selectedLeftModel, selectedMethod);
                }
                if (selectedRightModel && selectedRightModel !== selectedLeftModel) {
                    loadFiles(selectedRightModel, selectedMethod);
                }
            }
        } else {
            // For method comparison, load files for the shared model with both methods
            if (selectedModel) {
                if (selectedLeftMethod) {
                    loadFiles(selectedModel, selectedLeftMethod);
                }
                if (selectedRightMethod && selectedRightMethod !== selectedLeftMethod) {
                    loadFiles(selectedModel, selectedRightMethod);
                }
            }
        }
    }, [selectedMethod, selectedLeftMethod, selectedRightMethod, comparisonType, selectedModel, selectedLeftModel, selectedRightModel]);

    // Handle model selection changes
    const handleModelChange = (side: 'left' | 'right' | 'shared', modelName: string) => {
        if (side === 'left') {
            setSelectedLeftModel(modelName);
            // Reset left method when model changes
            setSelectedLeftMethod('');

            // Load methods for this model if needed
            if (modelName && (!methods[modelName] || methods[modelName].length === 0)) {
                getModelMethods(modelName).then(methodsList => {
                    setMethods(prev => ({
                        ...prev,
                        [modelName]: methodsList
                    }));

                    // Auto-select first method if applicable
                    if (methodsList.length > 0 && comparisonType === 'both') {
                        setSelectedLeftMethod(methodsList[0]);
                    }
                }).catch(err => {
                    console.error(`Failed to load methods for ${modelName}:`, err);
                });
            }
        } else if (side === 'right') {
            setSelectedRightModel(modelName);
            // Reset right method when model changes
            setSelectedRightMethod('');

            // Load methods for this model if needed
            if (modelName && (!methods[modelName] || methods[modelName].length === 0)) {
                getModelMethods(modelName).then(methodsList => {
                    setMethods(prev => ({
                        ...prev,
                        [modelName]: methodsList
                    }));

                    // Auto-select first method if applicable
                    if (methodsList.length > 0 && comparisonType === 'both') {
                        setSelectedRightMethod(methodsList[0]);
                    }
                }).catch(err => {
                    console.error(`Failed to load methods for ${modelName}:`, err);
                });
            }
        } else {
            setSelectedModel(modelName);
            // Reset methods when shared model changes
            setSelectedLeftMethod('');
            setSelectedRightMethod('');

            // Load methods for this model if needed
            if (modelName && (!methods[modelName] || methods[modelName].length === 0)) {
                getModelMethods(modelName).then(methodsList => {
                    setMethods(prev => ({
                        ...prev,
                        [modelName]: methodsList
                    }));

                    // Auto-select methods if applicable
                    if (methodsList.length > 0 && comparisonType === 'methods') {
                        setSelectedLeftMethod(methodsList[0]);
                        if (methodsList.length > 1) {
                            setSelectedRightMethod(methodsList[1]);
                        } else {
                            setSelectedRightMethod(methodsList[0]);
                        }
                    }
                }).catch(err => {
                    console.error(`Failed to load methods for ${modelName}:`, err);
                });
            }
        }
    };
    // Handle form submission
    const handleCompare = () => {
        let url = '/compare?';

        if (comparisonType === 'models') {
            url += `type=models&leftModel=${encodeURIComponent(selectedLeftModel)}&rightModel=${encodeURIComponent(selectedRightModel)}&method=${encodeURIComponent(selectedMethod)}&fileId=${selectedFileId}`;
        } else if (comparisonType === 'methods') {
            url += `type=methods&model=${encodeURIComponent(selectedModel)}&leftMethod=${encodeURIComponent(selectedLeftMethod)}&rightMethod=${encodeURIComponent(selectedRightMethod)}&fileId=${selectedFileId}`;
        } else if (comparisonType === 'both') {
            url += `type=both&leftModel=${encodeURIComponent(selectedLeftModel)}&rightModel=${encodeURIComponent(selectedRightModel)}&leftMethod=${encodeURIComponent(selectedLeftMethod)}&rightMethod=${encodeURIComponent(selectedRightMethod)}&fileId=${selectedFileId}`;
        }

        // Add aggregation methods to URL
        url += `&leftAggregation=${encodeURIComponent(leftAggregation)}&rightAggregation=${encodeURIComponent(rightAggregation)}`;

        router.push(url);
    };

    // Handle type change
    const handleTypeChange = (type: 'models' | 'methods' | 'both') => {
        setComparisonType(type);

        if (type === 'models') {
            // For model comparison, set a shared method
            setSelectedMethod('');
            if (methods[selectedModel]?.length > 0) {
                setSelectedMethod(methods[selectedModel][0]);
            }
        } else if (type === 'methods') {
            // For method comparison, set a shared model
            setSelectedLeftMethod('');
            setSelectedRightMethod('');
            if (methods[selectedModel]?.length > 0) {
                setSelectedLeftMethod(methods[selectedModel][0]);
                if (methods[selectedModel].length > 1) {
                    setSelectedRightMethod(methods[selectedModel][1]);
                } else {
                    setSelectedRightMethod(methods[selectedModel][0]);
                }
            }
        } else if (type === 'both') {
            // For full comparison, set different models and methods
            if (methods[selectedLeftModel]?.length > 0) {
                setSelectedLeftMethod(methods[selectedLeftModel][0]);
            }
            if (methods[selectedRightModel]?.length > 0) {
                setSelectedRightMethod(methods[selectedRightModel][0]);
            }
        }
    };

    return (
        <div className="space-y-6">
            <div className="p-6 border rounded-lg">
                <h3 className="text-lg font-medium mb-4">Configure Comparison</h3>

                {/* Comparison Type Selection */}
                <div className="space-y-4 mb-6">
                    <label className="text-sm font-medium">What would you like to compare?</label>
                    <div className="flex flex-wrap gap-4">
                        <div className="flex items-center gap-2">
                            <input
                                type="radio"
                                id="compareModels"
                                checked={comparisonType === 'models'}
                                onChange={() => handleTypeChange('models')}
                                className="w-4 h-4"
                            />
                            <label htmlFor="compareModels">Two Models (same method)</label>
                        </div>

                        <div className="flex items-center gap-2">
                            <input
                                type="radio"
                                id="compareMethods"
                                checked={comparisonType === 'methods'}
                                onChange={() => handleTypeChange('methods')}
                                className="w-4 h-4"
                            />
                            <label htmlFor="compareMethods">Two Methods (same model)</label>
                        </div>

                        <div className="flex items-center gap-2">
                            <input
                                type="radio"
                                id="compareBoth"
                                checked={comparisonType === 'both'}
                                onChange={() => handleTypeChange('both')}
                                className="w-4 h-4"
                            />
                            <label htmlFor="compareBoth">Two Models with Different Methods</label>
                        </div>
                    </div>
                </div>

                {/* Model/Method Selection Forms */}
                {comparisonType === 'models' ? (
                    <div className="space-y-6">
                        {/* Model Comparison Form */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                                <label className="text-sm font-medium block mb-2">Left Model</label>
                                <select
                                    className="w-full p-2 border rounded"
                                    value={selectedLeftModel}
                                    onChange={(e) => setSelectedLeftModel(e.target.value)}
                                >
                                    <option value="">Select a model</option>
                                    {models.map((model) => (
                                        <option key={`left-${model}`} value={model}>{model}</option>
                                    ))}
                                </select>
                            </div>

                            <div>
                                <label className="text-sm font-medium block mb-2">Right Model</label>
                                <select
                                    className="w-full p-2 border rounded"
                                    value={selectedRightModel}
                                    onChange={(e) => setSelectedRightModel(e.target.value)}
                                >
                                    <option value="">Select a model</option>
                                    {models.map((model) => (
                                        <option key={`right-${model}`} value={model}>{model}</option>
                                    ))}
                                </select>
                            </div>
                        </div>

                        <div>
                            <label className="text-sm font-medium block mb-2">Attribution Method</label>
                            <select
                                className="w-full p-2 border rounded"
                                value={selectedMethod}
                                onChange={(e) => setSelectedMethod(e.target.value)}
                                disabled={!selectedLeftModel || !selectedRightModel}
                            >
                                <option value="">Select a method</option>
                                {selectedLeftModel && methods[selectedLeftModel]?.map((method) => (
                                    <option key={method} value={method}>{method.replace(/_/g, ' ')}</option>
                                ))}
                            </select>
                        </div>
                    </div>
                ) : comparisonType === 'methods' ? (
                    <div className="space-y-6">
                        {/* Method Comparison Form */}
                        <div>
                            <label className="text-sm font-medium block mb-2">Model</label>
                            <select
                                className="w-full p-2 border rounded"
                                value={selectedModel}
                                onChange={(e) => handleModelChange('shared', e.target.value)}
                            >
                                <option value="">Select a model</option>
                                {models.map((model) => (
                                    <option key={model} value={model}>{model}</option>
                                ))}
                            </select>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                                <label className="text-sm font-medium block mb-2">Left Method</label>
                                <select
                                    className="w-full p-2 border rounded"
                                    value={selectedLeftMethod}
                                    onChange={(e) => setSelectedLeftMethod(e.target.value)}
                                    disabled={!selectedModel}
                                >
                                    <option value="">Select a method</option>
                                    {selectedModel && methods[selectedModel]?.map((method) => (
                                        <option key={`left-${method}`} value={method}>{method.replace(/_/g, ' ')}</option>
                                    ))}
                                </select>
                            </div>

                            <div>
                                <label className="text-sm font-medium block mb-2">Right Method</label>
                                <select
                                    className="w-full p-2 border rounded"
                                    value={selectedRightMethod}
                                    onChange={(e) => setSelectedRightMethod(e.target.value)}
                                    disabled={!selectedModel}
                                >
                                    <option value="">Select a method</option>
                                    {selectedModel && methods[selectedModel]?.map((method) => (
                                        <option key={`right-${method}`} value={method}>{method.replace(/_/g, ' ')}</option>
                                    ))}
                                </select>
                            </div>
                        </div>
                    </div>
                ) : (
                    <div className="space-y-6">
                        {/* Both Models and Methods Comparison Form */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            {/* Left side */}
                            <div className="space-y-4 border-r pr-4">
                                <h4 className="font-medium">Left Side</h4>
                                <div>
                                    <label className="text-sm font-medium block mb-2">Model</label>
                                    <select
                                        className="w-full p-2 border rounded"
                                        value={selectedLeftModel}
                                        onChange={(e) => {
                                            setSelectedLeftModel(e.target.value);
                                            handleModelChange('left', e.target.value);
                                        }}
                                    >
                                        <option value="">Select a model</option>
                                        {models.map((model) => (
                                            <option key={`left-model-${model}`} value={model}>{model}</option>
                                        ))}
                                    </select>
                                </div>

                                <div>
                                    <label className="text-sm font-medium block mb-2">Method</label>
                                    <select
                                        className="w-full p-2 border rounded"
                                        value={selectedLeftMethod}
                                        onChange={(e) => setSelectedLeftMethod(e.target.value)}
                                        disabled={!selectedLeftModel}
                                    >
                                        <option value="">Select a method</option>
                                        {selectedLeftModel && methods[selectedLeftModel]?.map((method) => (
                                            <option key={`left-method-${method}`} value={method}>{method.replace(/_/g, ' ')}</option>
                                        ))}
                                    </select>
                                </div>
                            </div>

                            {/* Right side */}
                            <div className="space-y-4">
                                <h4 className="font-medium">Right Side</h4>
                                <div>
                                    <label className="text-sm font-medium block mb-2">Model</label>
                                    <select
                                        className="w-full p-2 border rounded"
                                        value={selectedRightModel}
                                        onChange={(e) => {
                                            setSelectedRightModel(e.target.value);
                                            handleModelChange('right', e.target.value);
                                        }}
                                    >
                                        <option value="">Select a model</option>
                                        {models.map((model) => (
                                            <option key={`right-model-${model}`} value={model}>{model}</option>
                                        ))}
                                    </select>
                                </div>

                                <div>
                                    <label className="text-sm font-medium block mb-2">Method</label>
                                    <select
                                        className="w-full p-2 border rounded"
                                        value={selectedRightMethod}
                                        onChange={(e) => setSelectedRightMethod(e.target.value)}
                                        disabled={!selectedRightModel}
                                    >
                                        <option value="">Select a method</option>
                                        {selectedRightModel && methods[selectedRightModel]?.map((method) => (
                                            <option key={`right-method-${method}`} value={method}>{method.replace(/_/g, ' ')}</option>
                                        ))}
                                    </select>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* File Selection and Aggregation Methods */}
                <div className="mt-6 space-y-6">
                    {/* File Selection */}
                    <div>
                        <label className="text-sm font-medium block mb-2">Attribution File</label>
                        <select
                            className="w-full p-2 border rounded"
                            value={selectedFileId}
                            onChange={(e) => setSelectedFileId(parseInt(e.target.value))}
                        >
                            {[...Array(10)].map((_, index) => (
                                <option key={index} value={index}>File {index + 1}</option>
                            ))}
                        </select>
                    </div>

                    {/* Aggregation Methods */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                            <label className="text-sm font-medium block mb-2">Left Aggregation Method</label>
                            <select
                                className="w-full p-2 border rounded"
                                value={leftAggregation}
                                onChange={(e) => setLeftAggregation(e.target.value)}
                            >
                                {availableAggregations.map((method) => (
                                    <option key={`left-agg-${method}`} value={method}>{method.replace(/_/g, ' ')}</option>
                                ))}
                            </select>
                            <p className="text-xs text-muted-foreground mt-1">
                                How attribution weights are combined for the left visualization
                            </p>
                        </div>

                        <div>
                            <label className="text-sm font-medium block mb-2">Right Aggregation Method</label>
                            <select
                                className="w-full p-2 border rounded"
                                value={rightAggregation}
                                onChange={(e) => setRightAggregation(e.target.value)}
                            >
                                {availableAggregations.map((method) => (
                                    <option key={`right-agg-${method}`} value={method}>{method.replace(/_/g, ' ')}</option>
                                ))}
                            </select>
                            <p className="text-xs text-muted-foreground mt-1">
                                How attribution weights are combined for the right visualization
                            </p>
                        </div>
                    </div>
                </div>

                {/* Error display */}
                {error && (
                    <div className="mt-4 p-3 bg-red-50 text-red-600 rounded">
                        {error}
                    </div>
                )}

                {/* Compare button */}
                <div className="mt-6">
                    <button
                        onClick={handleCompare}
                        disabled={loading || (
                            comparisonType === 'models' ?
                                !selectedLeftModel || !selectedRightModel || !selectedMethod :
                                !selectedModel || !selectedLeftMethod || !selectedRightMethod
                        )}
                        className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed"
                    >
                        {loading ? 'Loading...' : 'Compare'}
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ComparisonSelector;