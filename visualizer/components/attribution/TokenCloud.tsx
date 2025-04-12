"use client"

import React, { useRef, useState, useEffect } from 'react';
import { Info } from 'lucide-react';
import { calculateTokenImportance, findTopInfluences } from '@/lib/utils';
import { AnalysisResult } from '@/lib/types';

interface TokenCloudProps {
  analysis: AnalysisResult;
  maxConnections?: number;
  useRelativeStrength?: boolean;
  showBackground?: boolean;
  showConnections?: boolean;
  showImportanceBars?: boolean;
  contextSize?: number;
}

interface Connection {
  path: string;
  opacity: number;
  strokeWidth: number;
}

const TokenCloud: React.FC<TokenCloudProps> = ({
  analysis,
  maxConnections = 3,
  useRelativeStrength = true,
  showBackground = true,
  showConnections = true,
  showImportanceBars = true,
  contextSize = 10
}) => {
  const [activeWordIndex, setActiveWordIndex] = useState<number | null>(null);
  const [lockedWordIndex, setLockedWordIndex] = useState<number | null>(null);
  const [connections, setConnections] = useState<Connection[]>([]);
  const wordRefs = useRef<(HTMLSpanElement | null)[]>([]);
  const containerRef = useRef<HTMLDivElement>(null);
  const [tokenImportance, setTokenImportance] = useState<number[]>([]);

  // Calculate token importance when analysis data changes
  useEffect(() => {
    if (analysis?.data) {
      const importance = calculateTokenImportance(
        analysis.data.input_tokens,
        analysis.data.normalized_association
      );
      setTokenImportance(importance);
    }
  }, [analysis]);

  // Function to create a visual connection between tokens
  const createConnection = (
    startIndex: number,
    endIndex: number,
    strength: number,
    influences: Array<{index: number, value: number}>
  ): Connection | null => {
    const startEl = wordRefs.current[startIndex];
    const endEl = wordRefs.current[endIndex];
    const container = containerRef.current;
    if (!startEl || !endEl || !container) return null;

    const containerRect = container.getBoundingClientRect();
    const startRect = startEl.getBoundingClientRect();
    const endRect = endEl.getBoundingClientRect();

    const x1 = startRect.left + startRect.width / 2 - containerRect.left;
    const y1 = startRect.top + startRect.height / 2 - containerRect.top;
    const x2 = endRect.left + endRect.width / 2 - containerRect.left;
    const y2 = endRect.top + endRect.height / 2 - containerRect.top;

    const cy = (y1 + y2) / 2;
    const cp1y = y1 + (cy - y1) * 0.5;
    const cp2y = y2 - (y2 - cy) * 0.5;

    const path = `M ${x1} ${y1} C ${x1} ${cp1y}, ${x2} ${cp2y}, ${x2} ${y2}`;

    let normalizedStrength = strength;
    if (useRelativeStrength && influences.length > 0) {
      const maxValue = Math.max(...influences.map(inf => inf.value));
      normalizedStrength = strength / maxValue;
    }

    return {
      path,
      opacity: normalizedStrength,
      strokeWidth: Math.max(1, normalizedStrength * 2)
    };
  };

  // Update connections when a token is hovered/clicked
  const updateConnections = (index: number) => {
    if (!analysis?.data) return;
    
    const newConnections: Connection[] = [];
    const inputLength = analysis.data.input_tokens.length;
    
    // Find top influences for this token
    const influences = findTopInfluences(
      index,
      inputLength,
      analysis.data.normalized_association,
      maxConnections
    );
    
    // Create connection for each influence
    for (const {index: influenceIndex, value} of influences) {
      if (value > 0) {
        const startIndex = Math.min(index, influenceIndex);
        const endIndex = Math.max(index, influenceIndex);
        
        const connection = createConnection(
          startIndex,
          endIndex,
          value,
          influences
        );
        
        if (connection) newConnections.push(connection);
      }
    }

    setConnections(newConnections);
  };

  // Event handlers for token interaction
  const handleWordHover = (index: number) => {
    if (lockedWordIndex !== null) return;
    if (index === activeWordIndex) return;

    setActiveWordIndex(index);
    updateConnections(index);
  };

  const handleWordClick = (index: number) => {
    if (lockedWordIndex === index) {
      setLockedWordIndex(null);
      setActiveWordIndex(null);
      setConnections([]);
    } else {
      setLockedWordIndex(index);
      setActiveWordIndex(index);
      updateConnections(index);
    }
  };

  const handleWordLeave = () => {
    if (lockedWordIndex !== null) return;
    setActiveWordIndex(null);
    setConnections([]);
  };

  // Calculate background highlight color for tokens
  const getTokenStyle = (index: number): React.CSSProperties => {
    if (!showBackground || activeWordIndex === null || !analysis?.data) return {};

    const inputLength = analysis.data.input_tokens.length;
    const isInputToken = index < inputLength;
    let strength = 0;

    if (activeWordIndex < inputLength) {
      if (!isInputToken) {
        const outputIndex = index - inputLength;
        if (outputIndex < analysis.data.normalized_association.length) {
          const row = analysis.data.normalized_association[outputIndex];
          strength = row[activeWordIndex] || 0;
        }
      }
    } else {
      const activeOutputIndex = activeWordIndex - inputLength;
      if (activeOutputIndex < analysis.data.normalized_association.length) {
        const row = analysis.data.normalized_association[activeOutputIndex];
        
        if (isInputToken && index < row.length) {
          strength = row[index] || 0;
        } else if (index < activeWordIndex && index - inputLength < row.length) {
          strength = row[index - inputLength] || 0;
        }
      }
    }

    if (index === activeWordIndex) {
      return {};
    }

    return {
      backgroundColor: `rgba(234, 88, 12, ${strength})`,
      opacity: 0.5 + (strength * 0.5)
    };
  };

  // Render a legend for the visualization
  const renderLegend = () => (
    <div className="absolute top-4 right-4 flex items-center gap-4 text-sm">
      <div className="flex items-center gap-2">
        <div className="w-3 h-3 rounded-full bg-orange-500" />
        <span>Input Tokens</span>
      </div>
      <div className="flex items-center gap-2">
        <div className="w-3 h-3 rounded-full bg-blue-500" />
        <span>Output Tokens</span>
      </div>
      <div className="flex items-center gap-2">
        <Info size={16} className="text-muted-foreground cursor-help" 
          // title="Hover over tokens to see relationships. Click to lock selection. Connection strength shown by line thickness."
        />
      </div>
    </div>
  );

  // Render a single token with its hover effects and importance bar
  const renderToken = (token: { clean_token: string, token_id: number }, index: number, globalIndex: number) => (
    <div key={index} className="group">
      <div className="flex flex-col items-center gap-1">
        <div className="relative">
          {/* @ts-ignore */}
          <span ref={el => wordRefs.current[globalIndex] = el}
            className="px-3 py-1.5 rounded-lg cursor-pointer transition-all duration-300 ease-in-out
                hover:shadow-lg hover:scale-105"
            style={getTokenStyle(globalIndex)}
            onMouseEnter={() => handleWordHover(globalIndex)}
            onMouseLeave={handleWordLeave}
            onClick={() => handleWordClick(globalIndex)}
            title={`Token: ${token.clean_token}, ID: ${token.token_id}`}
          >
            {token.clean_token || '<empty>'}
          </span>
        </div>
        {showImportanceBars && (
          <div className="w-full bg-muted rounded-full h-1.5">
            <div
              className="bg-foreground rounded-full h-1.5 transition-all duration-300"
              style={{
                width: `${(tokenImportance[globalIndex] || 0) * 100}%`
              }}
            />
          </div>
        )}
      </div>
    </div>
  );

  if (!analysis?.data) {
    return <div className="min-h-[400px] flex items-center justify-center">No analysis data available</div>;
  }

  return (
    <div className="space-y-4">
      <div
        ref={containerRef}
        className="relative min-h-[400px] rounded-xl border backdrop-blur-sm"
      >
        {renderLegend()}

        <div className="relative p-8 border-b">
          <h3 className="absolute -top-3 left-4 px-2 bg-background text-sm text-orange-500">
            Input Tokens
          </h3>
          <div className="flex flex-wrap gap-4">
            {analysis.data.input_tokens.map((token, index) =>
              renderToken(token, index, index)
            )}
          </div>
        </div>

        {showConnections && (
          <svg className="absolute inset-0 pointer-events-none overflow-visible">
            <defs>
              <linearGradient id="connectionGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="rgb(234, 88, 12)" />
                <stop offset="100%" stopColor="rgb(59, 130, 246)" />
              </linearGradient>
            </defs>
            <g className="connection-lines">
              {connections.map((connection, index) => (
                <path
                  key={index}
                  d={connection.path}
                  fill="none"
                  stroke="url(#connectionGradient)"
                  strokeWidth={connection.strokeWidth}
                  strokeOpacity={connection.opacity}
                  className="transition-all duration-300"
                />
              ))}
            </g>
          </svg>
        )}

        <div className="relative p-8">
          <h3 className="absolute -top-3 left-4 px-2 bg-background text-sm text-blue-500">
            Output Tokens
          </h3>
          <div className="flex flex-wrap gap-4">
            {analysis.data.output_tokens.map((token, index) =>
              renderToken(token, index, index + analysis.data.input_tokens.length)
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TokenCloud;