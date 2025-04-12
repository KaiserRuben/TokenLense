import { NextResponse } from 'next/server';

// Base API URL from server environment variable
const API_BASE_URL = process.env.API_URL || 'http://localhost:8000';

/**
 * API route handler for /api/performance/system
 * Gets system information
 */
export async function GET() {
  try {
    const response = await fetch(`${API_BASE_URL}/performance/system`);
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }
    
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('System info API error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch system information' }, 
      { status: 500 }
    );
  }
}
