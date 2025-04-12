"use client"

import React from 'react';
import ComparisonSelector from '@/components/comparison/ComparisonSelector';

export default function ComparisonSelectorPage() {
    return (
        <div className="space-y-8">
            <div>
                <h1 className="text-4xl font-bold">Compare Attribution Visualizations</h1>
                <p className="mt-2 text-muted-foreground">
                    Select what you want to compare to see token attribution side by side
                </p>
            </div>

            <ComparisonSelector />
        </div>
    );
}