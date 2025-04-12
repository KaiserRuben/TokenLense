import { NextResponse } from 'next/server';

// Base API URL from server environment variable
const API_BASE_URL = process.env.API_URL || 'http://localhost:8000';

/**
 * API route handler for /api/attribution/[model]/[method]/[fileId]
 * Gets attribution data for a model, method, and file
 */
export async function GET(
  request: Request,
  { params }: { params: { model: string; method: string; fileId: string } }
) {
  const { model, method, fileId } = params;
  const url = new URL(request.url);
  const aggregation = url.searchParams.get('aggregation') || 'sum';
  
  try {
    const response = await fetch(
      `${API_BASE_URL}/attribution/${model}/${method}/${fileId}?aggregation=${aggregation}`
    );
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }
    
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error(`Attribution API error for ${model}/${method}/${fileId}:`, error);
    return NextResponse.json(
      { error: `Failed to fetch attribution data for model ${model}, method ${method}, file ${fileId}` }, 
      { status: 500 }
    );
  }
}
