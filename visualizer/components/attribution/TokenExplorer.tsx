"use client"

import React, { useState } from 'react';
import TokenCloud from './TokenCloud';
import { AnalysisResult } from '@/lib/types';

interface TokenExplorerProps {
  analysis: AnalysisResult;
}

export const TokenExplorer: React.FC<TokenExplorerProps> = ({ analysis }) => {
  const [maxConnections, setMaxConnections] = useState(3);
  const [useRelativeStrength, setUseRelativeStrength] = useState(true);
  const [visualizationMode, setVisualizationMode] = useState<'background' | 'connections'>('background');

  return (
      <div className="space-y-4">
        <div className="p-4 space-y-6 rounded-lg border">
          {/* Visualization Mode selection - primary control */}
          <div className="space-y-3">
            <h3 className="text-sm font-medium">Visualization Mode</h3>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <input
                    type="radio"
                    id="backgroundMode"
                    name="visualizationMode"
                    checked={visualizationMode === 'background'}
                    onChange={() => setVisualizationMode('background')}
                    className="w-4 h-4"
                />
                <label htmlFor="backgroundMode" className="text-sm">
                  Background Highlighting
                </label>
              </div>

              <div className="flex items-center gap-2">
                <input
                    type="radio"
                    id="connectionsMode"
                    name="visualizationMode"
                    checked={visualizationMode === 'connections'}
                    onChange={() => setVisualizationMode('connections')}
                    className="w-4 h-4"
                />
                <label htmlFor="connectionsMode" className="text-sm">
                  Connection Lines
                </label>
              </div>
            </div>
          </div>

          {/* Conditionally render max connections slider when in connections mode */}
          {visualizationMode === 'connections' && (
              <div className="space-y-2 pl-4 border-l-2 border-gray-100 dark:border-gray-800">
                <label className="text-sm font-medium block">
                  Max Connections per Token: {maxConnections}
                </label>
                <input
                    type="range"
                    value={maxConnections}
                    onChange={(e) => setMaxConnections(parseInt(e.target.value))}
                    min={1}
                    max={10}
                    step={1}
                    className="w-full"
                />
                <p className="text-xs text-muted-foreground">
                  Controls how many connection lines to show for each token
                </p>
              </div>
          )}

          {/* Normalization toggle - applies to both modes */}
          <div className="flex items-center gap-2">
            <input
                type="checkbox"
                id="normalizeToggle"
                checked={useRelativeStrength}
                onChange={() => setUseRelativeStrength(!useRelativeStrength)}
                className="w-4 h-4 rounded"
            />
            <div>
              <label htmlFor="normalizeToggle" className="text-sm font-medium">
                Normalize strengths
              </label>
              <p className="text-xs text-muted-foreground">
                Make strengths relative within each token&#39;s connections
              </p>
            </div>
          </div>
        </div>

        <TokenCloud
            analysis={analysis}
            maxConnections={maxConnections}
            useRelativeStrength={useRelativeStrength}
            showBackground={visualizationMode === 'background'}
            showConnections={visualizationMode === 'connections'}
        />
      </div>
  );
};

export default TokenExplorer;