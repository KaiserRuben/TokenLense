"use client"

import { useEffect, useState } from 'react';
import { getModels } from '@/lib/api';
import Link from 'next/link';

export default function Home() {
  const [models, setModels] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadModels() {
      try {
        setLoading(true);
        const modelData = await getModels();
        setModels(modelData);
      } catch (err) {
        setError('Failed to load models. Please try again later.');
        console.error('Error loading models:', err);
      } finally {
        setLoading(false);
      }
    }
    
    loadModels();
  }, []);

  return (
    <div className="space-y-8">
      <div className="max-w-3xl">
        <h1 className="text-4xl font-bold tracking-tight">TokenLense</h1>
        <p className="mt-4 text-lg text-muted-foreground">
          Advanced visualization for language model token attribution analysis.
          Explore how different models and attribution methods reveal patterns in token relationships.
        </p>
      </div>
      
      <div className="mt-12">
        <h2 className="text-2xl font-semibold mb-4">Select a Model</h2>
        
        {loading && (
          <div className="flex items-center justify-center h-64">
            <div className="animate-pulse text-muted-foreground">Loading models...</div>
          </div>
        )}
        
        {error && (
          <div className="p-4 border border-red-200 bg-red-50 text-red-600 rounded-md">
            {error}
          </div>
        )}
        
        {!loading && !error && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {models.map(model => (
              <Link 
                key={model} 
                href={`/models/${encodeURIComponent(model)}`}
                className="group p-6 rounded-lg border shadow-sm hover:shadow-md transition-all"
              >
                <h3 className="text-xl font-medium group-hover:text-blue-600 transition-colors">
                  {model}
                </h3>
                <p className="mt-2 text-sm text-muted-foreground">
                  Explore attribution methods and visualizations for {model}
                </p>
              </Link>
            ))}
            
            {models.length === 0 && (
              <div className="col-span-3 p-6 rounded-lg border text-center text-muted-foreground">
                No models available. Please check your API configuration.
              </div>
            )}
          </div>
        )}
      </div>
      
      <div className="mt-12 border-t pt-8">
        <h2 className="text-2xl font-semibold mb-4">About TokenLense</h2>
        <div className="prose max-w-none">
          <p>
            TokenLense provides advanced visualizations for token attribution data from language models.
            It helps researchers and developers understand how different parts of an input prompt influence
            the model&#39;s output generation, revealing insights into the model&#39;s internal decision-making.
          </p>
          <p>
            Select a model above to explore its attribution methods and visualize token relationships.
          </p>
        </div>
      </div>
    </div>
  );
}