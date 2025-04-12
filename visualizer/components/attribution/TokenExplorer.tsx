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
  const [showBackground, setShowBackground] = useState(true);
  const [showConnections, setShowConnections] = useState(true);
  const [showImportanceBars, setShowImportanceBars] = useState(true);
  const [contextSize, setContextSize] = useState(10);

  return (
    <div className="space-y-4">
      <div className="p-4 space-y-6 rounded-lg border">
        {/* Connection controls */}
        <div>
          <label className="text-sm font-medium mb-2 block">
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
        </div>
        
        {/* Context window size */}
        <div>
          <label className="text-sm font-medium mb-2 block">
            Context Window Size: {contextSize}
          </label>
          <input
            type="range"
            value={contextSize}
            onChange={(e) => setContextSize(parseInt(e.target.value))}
            min={5}
            max={20}
            step={1}
            className="w-full"
          />
        </div>

        {/* Visualization toggles */}
        <div className="flex flex-wrap gap-6">
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="normalizeToggle"
              checked={useRelativeStrength}
              onChange={() => setUseRelativeStrength(!useRelativeStrength)}
              className="w-4 h-4 rounded"
            />
            <label htmlFor="normalizeToggle" className="text-sm">
              Normalize strengths
            </label>
          </div>
          
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="backgroundToggle"
              checked={showBackground}
              onChange={() => setShowBackground(!showBackground)}
              className="w-4 h-4 rounded"
            />
            <label htmlFor="backgroundToggle" className="text-sm">
              Show background
            </label>
          </div>
          
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="connectionsToggle"
              checked={showConnections}
              onChange={() => setShowConnections(!showConnections)}
              className="w-4 h-4 rounded"
            />
            <label htmlFor="connectionsToggle" className="text-sm">
              Show connections
            </label>
          </div>
          
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="importanceToggle"
              checked={showImportanceBars}
              onChange={() => setShowImportanceBars(!showImportanceBars)}
              className="w-4 h-4 rounded"
            />
            <label htmlFor="importanceToggle" className="text-sm">
              Show importance bars
            </label>
          </div>
        </div>
      </div>

      <TokenCloud
        analysis={analysis}
        maxConnections={maxConnections}
        useRelativeStrength={useRelativeStrength}
        showBackground={showBackground}
        showConnections={showConnections}
        showImportanceBars={showImportanceBars}
        contextSize={contextSize}
      />
    </div>
  );
};

export default TokenExplorer;