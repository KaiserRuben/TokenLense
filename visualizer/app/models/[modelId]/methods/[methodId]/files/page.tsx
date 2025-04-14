"use client"

import {useEffect, useState} from 'react';
import {getModelMethodFiles} from '@/lib/api';
import Link from 'next/link';
import {ArrowLeft, Search} from 'lucide-react';
import {useParams} from 'next/navigation';
import {ModelMethodFile} from '@/lib/types';
import {formatTimestamp} from "@/lib/utils";
import { Input } from "@/components/ui/input";

export default function FilesPage() {
    const params = useParams();
    const modelId = decodeURIComponent(params.modelId as string);
    const methodId = decodeURIComponent(params.methodId as string);

    const [fileData, setFileData] = useState<ModelMethodFile | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [searchTerm, setSearchTerm] = useState<string>('');

    useEffect(() => {
        async function loadFiles() {
            try {
                setLoading(true);
                const data = await getModelMethodFiles(modelId, methodId, true);
                setFileData(data);
            } catch (err) {
                setError(`Failed to load attribution files for ${modelId}/${methodId}. Please try again later.`);
                console.error('Error loading files:', err);
            } finally {
                setLoading(false);
            }
        }

        if (modelId && methodId) {
            loadFiles();
        }
    }, [modelId, methodId]);

    // Function to truncate text
    const truncateText = (text: string, maxLength = 100) => {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    };
    
    // Function to filter and sort files based on search term
    const filteredFiles = () => {
        if (!fileData || !searchTerm.trim()) {
            return fileData?.files.map((_, i) => i);
        }
        
        const searchTermLower = searchTerm.toLowerCase();
        
        // Create a structure to track match information for sorting
        interface MatchInfo {
            index: number;
            matchesPrompt: boolean;
            matchesGeneration: boolean;
            matchFilePath: boolean;
        }
        
        // Map files to match info for filtering and sorting
        const matches: MatchInfo[] = fileData.files.map((file, index) => {
            const details = fileData.file_details?.[index];
            const prompt = details?.prompt || '';
            const generation = details?.generation || '';
            
            // Default match info (no matches)
            const matchInfo: MatchInfo = {
                index,
                matchesPrompt: false,
                matchesGeneration: false,
                matchFilePath: false
            };
            
            // Check for matches in different fields
            if (prompt.toLowerCase().includes(searchTermLower)) {
                matchInfo.matchesPrompt = true;
            }
            
            if (generation.toLowerCase().includes(searchTermLower)) {
                matchInfo.matchesGeneration = true;
            }
            
            if (file.toLowerCase().includes(searchTermLower)) {
                matchInfo.matchFilePath = true;
            }
            
            return matchInfo;
        });
        
        // Filter out non-matches and sort
        const filtered = matches
            .filter(m => m.matchesPrompt || m.matchesGeneration || m.matchFilePath)
            .sort((a, b) => {
                // Prioritize prompt matches first
                if (a.matchesPrompt && !b.matchesPrompt) return -1;
                if (!a.matchesPrompt && b.matchesPrompt) return 1;
                
                // Then prioritize generation matches
                if (a.matchesGeneration && !b.matchesGeneration) return -1;
                if (!a.matchesGeneration && b.matchesGeneration) return 1;
                
                // Keep original order if same match type
                return a.index - b.index;
            });
        
        // Return just the indexes in sorted order
        return filtered.map(m => m.index);
    };

    return (
        <div className="space-y-8">
            <div className="flex items-center gap-2">
                <Link
                    href={`/models/${encodeURIComponent(modelId)}`}
                    className="flex items-center text-sm text-muted-foreground hover:text-foreground transition-colors"
                >
                    <ArrowLeft className="mr-1 h-4 w-4"/>
                    Back to {modelId} Methods
                </Link>
            </div>

            <div>
                <h1 className="text-3xl font-bold">{modelId}</h1>
                <h2 className="text-xl font-medium mt-2">{methodId.replace(/_/g, ' ')}</h2>
                <p className="mt-2 text-muted-foreground">
                    Select an attribution file to visualize token relationships
                </p>
            </div>
            
            {/* Search bar with shadcn input */}
            <div className="relative w-full">
                <Input
                    type="text"
                    className="w-full pl-10"
                    placeholder="Search in prompts and responses..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                />
            </div>

            {loading && (
                <div className="flex items-center justify-center h-64">
                    <div className="animate-pulse text-muted-foreground">Loading attribution files...</div>
                </div>
            )}

            {error && (
                <div className="p-4 border border-red-200 bg-red-50 text-red-600 rounded-md">
                    {error}
                </div>
            )}

            {!loading && !error && fileData && (
                <div className="space-y-4">
                    {filteredFiles()?.map((fileIndex) => (
                        <Link
                            key={fileIndex}
                            href={`/attribution/${encodeURIComponent(modelId)}/${encodeURIComponent(methodId)}/${fileIndex}`}
                            className="block p-6 rounded-lg border shadow-sm hover:shadow-md transition-all"
                        >
                            <div className="flex justify-between items-start">
                                <div className="flex-1">{fileData.file_details && fileData.file_details[fileIndex] && (
                                    <>
                                        <h3 className="font-medium">
                                            <span className="font-medium">Prompt:</span>{' '}
                                            {truncateText(fileData.file_details[fileIndex].prompt || 'No prompt available')}
                                        </h3>
                                        
                                        {/* Show generation if available */}
                                        {fileData.file_details[fileIndex].generation && (
                                            <p className="mt-1 text-sm text-gray-700 dark:text-gray-300">
                                                <span className="font-medium">Response:</span>{' '}
                                                {truncateText(fileData.file_details[fileIndex].generation, 150)}
                                            </p>
                                        )}
                                        
                                        {/* Show match information if searching */}
                                        {searchTerm && (
                                            <div className="mt-1 text-xs text-blue-600 flex gap-4">
                                                {fileData.file_details[fileIndex].prompt?.toLowerCase().includes(searchTerm.toLowerCase()) && (
                                                    <span className="mr-2 p-2 px-1.5 py-0.5 bg-blue-100 dark:bg-blue-900 rounded">
                                                        Prompt match
                                                    </span>
                                                )}
                                                {fileData.file_details[fileIndex].generation?.toLowerCase().includes(searchTerm.toLowerCase()) && (
                                                    <span className="mr-2 px-1.5 py-0.5 bg-green-100 dark:bg-green-900 rounded">
                                                        Response match
                                                    </span>
                                                )}
                                            </div>
                                        )}
                                        
                                        <p className="mt-1 text-sm">
                                            File {fileIndex + 1}
                                            <span className="pl-2 pr-2">-</span>
                                            {fileData.file_details[fileIndex].timestamp && (
                                                <span className="mt-1 text-xs text-muted-foreground">
                                                {formatTimestamp(fileData.file_details[fileIndex].timestamp)}
                                            </span>
                                            )}
                                        </p>
                                    </>
                                )}
                                    {!fileData.file_details && (
                                        <p className="mt-1 text-sm text-muted-foreground">
                                            Filepath: {fileData.files[fileIndex]}
                                        </p>
                                    )}
                                </div>
                                <span className="text-blue-600 text-sm hover:underline shrink-0 ml-4">View Attribution</span>
                            </div>
                        </Link>
                    ))}

                    {filteredFiles()?.length === 0 && (
                        <div className="p-6 rounded-lg border text-center text-muted-foreground">
                            {searchTerm ? 
                                `No files matching "${searchTerm}" found.` : 
                                'No attribution files available for this model and method.'
                            }
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}