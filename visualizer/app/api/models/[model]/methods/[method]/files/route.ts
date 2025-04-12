import { NextResponse } from 'next/server';

// Base API URL from server environment variable
const API_BASE_URL = process.env.API_URL || 'http://localhost:8000';

/**
 * API route handler for /api/models/[model]/methods/[method]/files
 * Gets files for a specific model and method
 */
export async function GET(
  request: Request,
  { params }: { params: { model: string; method: string } }
) {
  const { model, method } = params;
  const url = new URL(request.url);
  const includeDetails = url.searchParams.get('include_details') === 'true';
  
  try {
    const response = await fetch(
      `${API_BASE_URL}/models/${model}/methods/${method}/files?include_details=${includeDetails}`
    );
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }
    
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error(`Model method files API error for ${model}/${method}:`, error);
    return NextResponse.json(
      { error: `Failed to fetch files for model ${model} and method ${method}` }, 
      { status: 500 }
    );
  }
}
