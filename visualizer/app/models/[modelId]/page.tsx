"use client"

import { useEffect, useState } from 'react';
import { getModelMethods } from '@/lib/api';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { useParams } from 'next/navigation';

export default function ModelPage() {
  const params = useParams();
  const modelId = decodeURIComponent(params.modelId as string);
  
  const [methods, setMethods] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadMethods() {
      try {
        setLoading(true);
        const methodData = await getModelMethods(modelId);
        setMethods(methodData);
      } catch (err) {
        setError(`Failed to load attribution methods for ${modelId}. Please try again later.`);
        console.error('Error loading methods:', err);
      } finally {
        setLoading(false);
      }
    }
    
    if (modelId) {
      loadMethods();
    }
  }, [modelId]);

  // Method descriptions (for static display)
  const methodDescriptions: Record<string, string> = {
    "attention": "Visualizes attention weights between tokens, showing how the model attends to input tokens when generating each output token.",
    "input_x_gradient": "Multiplies input embeddings by their gradients to identify which input dimensions most influenced the output.",
    "integrated_gradients": "Calculates path integrals of gradients to provide a more accurate attribution of the model's prediction.",
    "layer_gradient_x_activation": "Multiplies gradients by activations at specific layers to identify important activations.",
    "lime": "Uses local perturbations of the input to approximate the model's behavior and attribute token importance.",
    "saliency": "Uses gradient magnitude to identify which tokens have the greatest impact on the model's prediction."
  };

  return (
    <div className="space-y-8">
      <div className="flex items-center gap-2">
        <Link 
          href="/"
          className="flex items-center text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          <ArrowLeft className="mr-1 h-4 w-4" />
          Back to Models
        </Link>
      </div>
      
      <div>
        <h1 className="text-3xl font-bold">{modelId}</h1>
        <p className="mt-2 text-muted-foreground">
          Select an attribution method to explore token relationships
        </p>
      </div>
      
      {loading && (
        <div className="flex items-center justify-center h-64">
          <div className="animate-pulse text-muted-foreground">Loading attribution methods...</div>
        </div>
      )}
      
      {error && (
        <div className="p-4 border border-red-200 bg-red-50 text-red-600 rounded-md">
          {error}
        </div>
      )}
      
      {!loading && !error && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {methods.map(method => (
            <Link
              key={method}
              href={`/models/${encodeURIComponent(modelId)}/methods/${encodeURIComponent(method)}/files`}
              className="group p-6 rounded-lg border shadow-sm hover:shadow-md transition-all"
            >
              <h3 className="text-xl font-medium group-hover:text-blue-600 transition-colors">
                {method.replace(/_/g, ' ')}
              </h3>
              <p className="mt-2 text-sm text-muted-foreground">
                {methodDescriptions[method] || 
                 `Explore how ${method.replace(/_/g, ' ')} attributes token influences.`}
              </p>
            </Link>
          ))}
          
          {methods.length === 0 && (
            <div className="col-span-2 p-6 rounded-lg border text-center text-muted-foreground">
              No attribution methods available for this model. Please select a different model.
            </div>
          )}
        </div>
      )}
      
      <div className="mt-8 pt-6 border-t">
        <h2 className="text-xl font-semibold mb-4">About Attribution Methods</h2>
        <div className="prose max-w-none">
          <p>
            Attribution methods help us understand how language models make decisions by revealing
            which tokens in the input most strongly influence each token in the output.
            Different methods provide different perspectives on token relationships:
          </p>
          <ul>
            <li><strong>Attention</strong> - Shows direct attention patterns between tokens</li>
            <li><strong>Gradient-based methods</strong> - Reveal which tokens cause the greatest change in output</li>
            <li><strong>Perturbation methods</strong> - Show how changes to input tokens affect the output</li>
          </ul>
        </div>
      </div>
    </div>
  );
}