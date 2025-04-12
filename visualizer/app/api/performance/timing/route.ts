import { NextResponse } from 'next/server';

// Base API URL from server environment variable
const API_BASE_URL = process.env.API_URL || 'http://localhost:8000';

/**
 * API route handler for /api/performance/timing
 * Gets timing results
 */
export async function GET() {
  try {
    const response = await fetch(`${API_BASE_URL}/performance/timing`);
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }
    
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Timing results API error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch timing results' }, 
      { status: 500 }
    );
  }
}
