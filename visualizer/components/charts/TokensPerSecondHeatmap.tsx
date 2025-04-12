import { useMemo, useState } from 'react';
import { TokensPerSecondData } from '@/lib/types';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Info } from 'lucide-react';

type Props = {
  data: TokensPerSecondData[];
};

const formatMethodName = (method: string) => {
  if (!method) return 'Unknown';

  // Remove 'method_' prefix if it exists
  let name = method.replace('method_', '');

  // Replace underscores with spaces
  name = name.replace(/_/g, ' ');

  // Capitalize first letter of each word
  return name
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
};

export default function TokensPerSecondHeatmap({ data }: Props) {
  const [scaleMode, setScaleMode] = useState<'linear' | 'log'>('linear');

  // Extract unique methods and models for axis labels
  const methods = useMemo(() => {
    if (!data || data.length === 0) return [];
    return Array.from(new Set(data.map(d => d.method))).sort();
  }, [data]);

  const models = useMemo(() => {
    if (!data || data.length === 0) return [];
    return Array.from(new Set(data.map(d => d.model))).sort();
  }, [data]);

  // Process data for the heatmap
  const processedData = useMemo(() => {
    const result = new Map();

    if (!data || data.length === 0) return result;

    // Initialize the map with all possible model-method combinations
    models.forEach(model => {
      methods.forEach(method => {
        const key = `${model}-${method}`;
        result.set(key, { model, method, value: null });
      });
    });

    // Fill in actual values
    data.forEach(d => {
      if (d.tokens_per_second !== undefined && d.tokens_per_second !== null && !isNaN(d.tokens_per_second)) {
        const key = `${d.model}-${d.method}`;
        result.set(key, {
          model: d.model,
          method: d.method,
          value: d.tokens_per_second,
          formattedMethod: formatMethodName(d.method)
        });
      }
    });

    return result;
  }, [data, models, methods]);

  // Calculate value range for the heatmap
  const { minValue, maxValue } = useMemo(() => {
    if (!data || data.length === 0) return { minValue: 0, maxValue: 1 };

    const validValues = data
        .map(d => d.tokens_per_second)
        .filter((val): val is number => typeof val === 'number' && !isNaN(val));

    return {
      minValue: validValues.length > 0 ? Math.min(...validValues) : 0,
      maxValue: validValues.length > 0 ? Math.max(...validValues) : 1
    };
  }, [data]);

  // Function to calculate color based on value
  const getColor = (value: number | null) => {
    if (value === null) return 'rgb(240, 240, 240)'; // Light gray for no data

    // For log scale, adjust the normalization
    let normalized;
    if (scaleMode === 'log') {
      // Add a small number to avoid log(0)
      const logMin = Math.log(minValue + 0.1);
      const logMax = Math.log(maxValue + 0.1);
      const logValue = Math.log(value + 0.1);

      normalized = (logValue - logMin) / (logMax - logMin);
    } else {
      normalized = (value - minValue) / (maxValue - minValue);
    }

    normalized = Math.max(0, Math.min(normalized, 1)); // Ensure it's between 0 and 1

    // Generate color from blue (cold) to red (hot)
    const r = Math.floor(255 * normalized);
    const g = Math.floor(100 * (1 - normalized));
    const b = Math.floor(255 * (1 - normalized));

    return `rgb(${r}, ${g}, ${b})`;
  };

  // Handle empty data
  if (!data || data.length === 0 || methods.length === 0 || models.length === 0) {
    return (
        <div className="flex flex-col items-center justify-center h-[500px] bg-gray-50 border rounded-md">
          <p className="text-muted-foreground mb-2">No tokens per second data available</p>
          <p className="text-xs text-muted-foreground">Try adjusting your filters</p>
        </div>
    );
  }

  // Calculate cell size based on available space
  const cellWidth = Math.min(100, 800 / models.length);
  const cellHeight = 40;

  return (
      <div className="flex flex-col">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <span className="text-sm">Scale:</span>
            <Select
                value={scaleMode}
                onValueChange={(value) => setScaleMode(value as 'linear' | 'log')}
            >
              <SelectTrigger className="w-[150px]">
                <SelectValue placeholder="Scale" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="linear">Linear</SelectItem>
                <SelectItem value="log">Logarithmic</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger className="flex items-center gap-1">
                <Info size={16} />
                <span className="text-sm text-muted-foreground">How to read</span>
              </TooltipTrigger>
              <TooltipContent>
                <p className="text-sm">Darker red cells indicate higher tokens/second values.</p>
                <p className="text-sm mt-1">Gray cells indicate no data available.</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>

        <div className="overflow-x-auto">
          <div className="relative" style={{ height: 80, marginLeft: 140 }}>
            {models.map((model, index) => (
                <div
                    key={`header-${model}`}
                    className="absolute font-medium text-xs"
                    style={{
                      left: cellWidth * index + (cellWidth / 2),
                      width: 0,
                      bottom: 0,
                      transformOrigin: 'left bottom',
                      transform: 'rotate(-45deg)',
                      whiteSpace: 'nowrap'
                    }}
                >
                  {model}
                </div>
            ))}
          </div>

          <div className="mt-2">
            {methods.map(method => (
                <div key={`row-${method}`} className="flex">
                  <div
                      className="flex-shrink-0 font-medium text-xs flex items-center"
                      style={{
                        width: 140,
                        height: cellHeight,
                        padding: '0 8px',
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis'
                      }}
                  >
                    {formatMethodName(method)}
                  </div>
                  <div className="flex">
                    {models.map(model => {
                      const key = `${model}-${method}`;
                      const cell = processedData.get(key);
                      const value = cell?.value;

                      // Display the actual value in the cell for better readability
                      const displayValue = value !== null && value !== undefined
                          ? (value > 99 ? Math.round(value) : value.toFixed(1))
                          : '';

                      return (
                          <TooltipProvider key={key}>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <div
                                    className="flex-shrink-0 border border-gray-100 flex items-center justify-center text-xs font-medium"
                                    style={{
                                      width: cellWidth,
                                      height: cellHeight,
                                      backgroundColor: getColor(value),
                                      cursor: 'pointer',
                                      color: value && value > maxValue * 0.7 ? 'white' : 'black'
                                    }}
                                >
                                  {displayValue}
                                </div>
                              </TooltipTrigger>
                              <TooltipContent side="top">
                                <div className="text-sm font-semibold">{model}</div>
                                <div className="text-sm">{formatMethodName(method)}</div>
                                <div className="text-sm font-bold mt-1">
                                  {value !== null ? `${value.toFixed(2)} tokens/sec` : 'No data'}
                                </div>
                              </TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                      );
                    })}
                  </div>
                </div>
            ))}
          </div>
        </div>

        <div className="flex justify-center items-center mt-6 gap-2">
          <div className="w-4 h-4 rounded-full" style={{ backgroundColor: getColor(minValue) }} />
          <span className="text-xs">Low ({minValue.toFixed(1)})</span>
          <div className="w-24 h-2 bg-gradient-to-r from-blue-500 via-purple-500 to-red-500 rounded mx-2" />
          <span className="text-xs">High ({maxValue.toFixed(1)})</span>
          <div className="w-4 h-4 rounded-full" style={{ backgroundColor: getColor(maxValue) }} />
        </div>
      </div>
  );
}