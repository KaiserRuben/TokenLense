import { NextResponse } from 'next/server';

// Base API URL from server environment variable
const API_BASE_URL = process.env.API_URL || 'http://localhost:8000';

/**
 * API route handler for /api/models/[model]/methods
 * Gets attribution methods for a specific model
 */
export async function GET(
  request: Request,
  { params }: { params: Promise<{ model: string }> }
) {
  const { model } = await params;
  
  try {
    const response = await fetch(`${API_BASE_URL}/models/${model}/methods`);
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }
    
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error(`Model methods API error for ${model}:`, error);
    return NextResponse.json(
      { error: `Failed to fetch methods for model ${model}` }, 
      { status: 500 }
    );
  }
}
