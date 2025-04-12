"use client"

import { useEffect, useState } from 'react';
import { getModelMethodFiles } from '@/lib/api';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { useParams } from 'next/navigation';
import { ModelMethodFile } from '@/lib/types';
import {formatTimestamp} from "@/lib/utils";

export default function FilesPage() {
  const params = useParams();
  const modelId = decodeURIComponent(params.modelId as string);
  const methodId = decodeURIComponent(params.methodId as string);
  
  const [fileData, setFileData] = useState<ModelMethodFile | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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

  return (
    <div className="space-y-8">
      <div className="flex items-center gap-2">
        <Link 
          href={`/models/${encodeURIComponent(modelId)}`}
          className="flex items-center text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          <ArrowLeft className="mr-1 h-4 w-4" />
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
          {fileData.files.map((file, index) => (
            <Link
              key={index}
              href={`/attribution/${encodeURIComponent(modelId)}/${encodeURIComponent(methodId)}/${index}`}
              className="block p-6 rounded-lg border shadow-sm hover:shadow-md transition-all"
            >
              <div className="flex justify-between items-start">
                <div>
                  <h3 className="font-medium">File {index + 1}</h3>
                  {fileData.file_details && fileData.file_details[index] && (
                    <>
                      <p className="mt-1 text-sm">
                        <span className="font-medium">Prompt:</span>{' '}
                        {truncateText(fileData.file_details[index].prompt || 'No prompt available')}
                      </p>
                      {fileData.file_details[index].timestamp && (
                        <p className="mt-1 text-xs text-muted-foreground">
                          {formatTimestamp(fileData.file_details[index].timestamp)}
                        </p>
                      )}
                    </>
                  )}
                  {!fileData.file_details && (
                    <p className="mt-1 text-sm text-muted-foreground">
                      Filepath: {file}
                    </p>
                  )}
                </div>
                <span className="text-blue-600 text-sm hover:underline">View Attribution</span>
              </div>
            </Link>
          ))}
          
          {fileData.files.length === 0 && (
            <div className="p-6 rounded-lg border text-center text-muted-foreground">
              No attribution files available for this model and method.
            </div>
          )}
        </div>
      )}
    </div>
  );
}