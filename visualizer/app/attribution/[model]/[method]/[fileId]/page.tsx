"use client"

import { useEffect, useState } from 'react';
import { getAttribution } from '@/lib/api';
import { convertAttributionToAnalysisResult } from '@/lib/utils';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { useParams, useSearchParams } from 'next/navigation';
import { AttributionResponse, AnalysisResult, AggregationMethod } from '@/lib/types';
import TokenExplorer from '@/components/attribution/TokenExplorer';

export default function AttributionPage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const model = decodeURIComponent(params.model as string);
  const method = decodeURIComponent(params.method as string);
  const fileId = parseInt(params.fileId as string, 10);
  const aggregation = (searchParams.get('aggregation') as AggregationMethod) || 'sum';
  
  const [attribution, setAttribution] = useState<AttributionResponse | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadAttribution() {
      try {
        setLoading(true);
        const data = await getAttribution(model, method, fileId, aggregation);
        setAttribution(data);
        
        // Convert API response to the format expected by visualization components
        const result = convertAttributionToAnalysisResult(data);
        setAnalysisResult(result);
      } catch (err) {
        setError(`Failed to load attribution data. Please try again later.`);
        console.error('Error loading attribution data:', err);
      } finally {
        setLoading(false);
      }
    }
    
    if (model && method && !isNaN(fileId)) {
      loadAttribution();
    }
  }, [model, method, fileId, aggregation]);

  return (
    <div className="space-y-8">
      <div className="flex items-center gap-2">
        <Link 
          href={`/models/${encodeURIComponent(model)}/methods/${encodeURIComponent(method)}/files`}
          className="flex items-center text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          <ArrowLeft className="mr-1 h-4 w-4" />
          Back to Files
        </Link>
      </div>
      
      <div>
        <h1 className="text-3xl font-bold">{model}</h1>
        <h2 className="text-xl font-medium mt-2">{method.replace(/_/g, ' ')}</h2>
        <p className="mt-2 text-muted-foreground">
          Token attribution visualization using {aggregation} aggregation
        </p>
      </div>
      
      {loading && (
        <div className="flex items-center justify-center h-64">
          <div className="animate-pulse text-muted-foreground">Loading attribution data...</div>
        </div>
      )}
      
      {error && (
        <div className="p-4 border border-red-200 bg-red-50 text-red-600 rounded-md">
          {error}
        </div>
      )}
      
      {!loading && !error && attribution && analysisResult && (
        <div className="space-y-8">
          {/* Prompt and generation display */}
          <div className="space-y-4">
            <div className="p-4 border rounded-lg">
              <h3 className="text-sm font-medium text-muted-foreground mb-2">Prompt</h3>
              <p className="whitespace-pre-wrap">{attribution.prompt}</p>
            </div>
            
            <div className="p-4 border rounded-lg">
              <h3 className="text-sm font-medium text-muted-foreground mb-2">Generation</h3>
              <p className="whitespace-pre-wrap">{attribution.generation}</p>
            </div>
          </div>
          
          {/* Aggregation method selector */}
          <div className="flex space-x-2">
            <span className="text-sm font-medium">Aggregation Method:</span>
            <div className="flex space-x-2">
              {['sum', 'mean', 'l2_norm', 'abs_sum', 'max'].map((method) => (
                <Link
                  key={method}
                  href={`/attribution/${encodeURIComponent(model)}/${encodeURIComponent(params.method as string)}/${fileId}?aggregation=${method}`}
                  className={`px-2 py-1 text-xs rounded ${
                    aggregation === method
                      ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300'
                      : 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
                  }`}
                >
                  {method.replace('_', ' ')}
                </Link>
              ))}
            </div>
          </div>
          
          {/* Token visualization */}
          <div className="border rounded-lg">
            <TokenExplorer analysis={analysisResult} />
          </div>
          
          {/* Attribution matrix info */}
          <div className="p-4 border rounded-lg">
            <h3 className="text-lg font-medium mb-2">Matrix Information</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-muted-foreground">Source Tokens</p>
                <p className="font-medium">{attribution.source_tokens.length}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Target Tokens</p>
                <p className="font-medium">{attribution.target_tokens.length}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Aggregation Method</p>
                <p className="font-medium">{attribution.aggregation}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Matrix Dimensions</p>
                <p className="font-medium">
                  {attribution.attribution_matrix.length} × 
                  {attribution.attribution_matrix[0]?.length || 0}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}