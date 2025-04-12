"use client"

import { useMemo, useState } from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  LabelList
} from 'recharts';
import { HardwareComparisonData } from '@/lib/types';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';

type Props = {
  data: HardwareComparisonData[];
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

// Shorten hardware name for better display
const formatHardwareName = (hardware: string) => {
  if (!hardware) return 'Unknown';

  if (hardware.includes('Tesla')) return 'Tesla T4';
  if (hardware.includes('H200')) return 'NVIDIA H200';
  if (hardware.includes('8259CL')) return 'Xeon 8259CL';
  if (hardware.includes('8468')) return 'Xeon 8468';
  if (hardware.includes('Apple M')) return hardware.split(' ').slice(0, 2).join(' ');

  // If it's too long, truncate it
  return hardware.length > 15 ? `${hardware.substring(0, 12)}...` : hardware;
};

export default function HardwareComparisonChart({ data }: Props) {
  const [selectedMethod, setSelectedMethod] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'grouped' | 'stacked'>('grouped');

  // Get unique methods and hardware types
  const methods = useMemo(() => {
    if (!data || data.length === 0) return [];
    return Array.from(new Set(data.map(d => d.method))).sort();
  }, [data]);

  const hardwareTypes = useMemo(() => {
    if (!data || data.length === 0) return [];
    return Array.from(new Set(data.map(d => d.hardware))).sort();
  }, [data]);

  // Initialize selectedMethod if not set
  useMemo(() => {
    if (methods.length > 0 && !selectedMethod) {
      setSelectedMethod(methods[0]);
    }
  }, [methods, selectedMethod]);

  // Filter data by selected method if needed
  const filteredData = useMemo(() => {
    if (!selectedMethod) return data;
    return data.filter(d => d.method === selectedMethod);
  }, [data, selectedMethod]);

  // Define interfaces for processed data
  interface ProcessedChartDataItem {
    [key: string]: string | number;
    displayName: string;
  }

  interface ModelChartData extends ProcessedChartDataItem {
    model: string;
  }

  interface MethodChartData extends ProcessedChartDataItem {
    method: string;
  }

  // Process data for the chart - group by method
  const processedData = useMemo((): ProcessedChartDataItem[] => {
    // For single method view, group by model
    if (selectedMethod) {
      // Get unique models
      const models = Array.from(new Set(filteredData.map(d => d.model))).sort();

      return models.map(model => {
        // Filter data for this model
        const modelData = filteredData.filter(d => d.model === model);

        // Create a record for this model with times for each hardware
        const result: ModelChartData = {
          model,
          displayName: model
        };

        // Add hardware-specific data
        hardwareTypes.forEach(hardware => {
          const hwData = modelData.filter(d => d.hardware === hardware);
          if (hwData.length > 0) {
            // Calculate average time if multiple entries exist
            const avgTime = hwData.reduce((sum, d) => sum + d.avg_time, 0) / hwData.length;
            const hwKey = formatHardwareName(hardware);
            result[hwKey] = avgTime;
          }
        });

        return result;
      });
    }
    // For multiple methods, group by method
    else {
      return methods.map(method => {
        // Filter data for this method
        const methodData = data.filter(d => d.method === method);

        // Create a record for this method with times for each hardware
        const result: MethodChartData = {
          method,
          displayName: formatMethodName(method)
        };

        // Add hardware-specific data
        hardwareTypes.forEach(hardware => {
          const hwData = methodData.filter(d => d.hardware === hardware);
          if (hwData.length > 0) {
            // Calculate average time if multiple entries exist
            const avgTime = hwData.reduce((sum, d) => sum + d.avg_time, 0) / hwData.length;
            const hwKey = formatHardwareName(hardware);
            result[hwKey] = avgTime;
          }
        });

        return result;
      });
    }
  }, [methods, hardwareTypes, selectedMethod, filteredData, data]);

  // Create bars for each hardware type
  const hardwareBars = useMemo(() => {
    const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff8042', '#0088fe', '#00C49F'];

    return hardwareTypes.map((hardware, index) => {
      const hwKey = formatHardwareName(hardware);
      return (
          <Bar
              key={hardware}
              dataKey={hwKey}
              name={hwKey}
              fill={colors[index % colors.length]}
              radius={[4, 4, 0, 0]}
              stackId={viewMode === 'stacked' ? 'a' : undefined}
          >
            {viewMode === 'stacked' && (
                <LabelList
                    dataKey={hwKey}
                    position="middle"
                    fill="#fff"
                    formatter={(value: number) => (value > 0.5 ? `${value.toFixed(1)}s` : '')}
                />
            )}
          </Bar>
      );
    });
  }, [hardwareTypes, viewMode]);

  // Define proper tooltip types
  interface HardwareTooltipProps {
    active?: boolean;
    payload?: Array<{
      name: string;
      value: number;
      color: string;
    }>;
    label?: string;
  }

  // Format tooltip values to show time in seconds
  const CustomTooltip = ({ active, payload, label }: HardwareTooltipProps) => {
    if (active && payload && payload.length) {
      return (
          <div className="bg-white p-3 border border-gray-200 shadow-md rounded-md">
            <p className="font-semibold mb-1">{label}</p>
            {payload.map((entry, index) => (
                <div key={`tooltip-${index}`} className="flex items-center gap-2 text-sm">
                  <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: entry.color }}
                  />
                  <span>{entry.name}: </span>
                  <span className="font-semibold">{entry.value.toFixed(2)}s</span>
                </div>
            ))}
          </div>
      );
    }
    return null;
  };

  // Handle empty data
  if (!data || data.length === 0 || methods.length === 0 || hardwareTypes.length === 0) {
    return (
        <div className="flex flex-col items-center justify-center h-[500px] bg-gray-50 border rounded-md">
          <p className="text-muted-foreground mb-2">No hardware comparison data available</p>
          <p className="text-xs text-muted-foreground">Try adjusting your filters</p>
        </div>
    );
  }

  return (
      <div className="flex flex-col">
        <div className="flex flex-col sm:flex-row sm:justify-between mb-4 gap-4">
          <div className="flex flex-col gap-2">
            <span className="text-sm">Method Filter:</span>
            <div className="flex flex-wrap gap-2">
              <Badge
                  variant={!selectedMethod ? "default" : "outline"}
                  className="cursor-pointer"
                  onClick={() => setSelectedMethod(null)}
              >
                All Methods
              </Badge>
              {methods.map(method => (
                  <Badge
                      key={method}
                      variant={selectedMethod === method ? "default" : "outline"}
                      className="cursor-pointer"
                      onClick={() => setSelectedMethod(method)}
                  >
                    {formatMethodName(method)}
                  </Badge>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-sm">View Mode:</span>
            <Select
                value={viewMode}
                onValueChange={(value) => setViewMode(value as 'grouped' | 'stacked')}
            >
              <SelectTrigger className="w-[150px]">
                <SelectValue placeholder="View Mode" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="grouped">Grouped</SelectItem>
                <SelectItem value="stacked">Stacked</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <ResponsiveContainer width="100%" height={500}>
          <BarChart
              data={processedData}
              margin={{ top: 20, right: 30, left: 20, bottom: 70 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
                dataKey="displayName"
                angle={-45}
                textAnchor="end"
                height={70}
                interval={0}
                tick={{ fontSize: 12 }}
            />
            <YAxis
                label={{ value: 'Average time (seconds)', angle: -90, position: 'insideLeft' }}
                tickFormatter={(value) => `${value.toFixed(1)}s`}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
                wrapperStyle={{ bottom: 0 }}
                formatter={(value) => <span style={{ color: '#666', fontSize: 14 }}>{value}</span>}
            />
            {hardwareBars}
          </BarChart>
        </ResponsiveContainer>

        {selectedMethod && (
            <div className="mt-4 p-3 bg-muted rounded-md">
              <h4 className="font-medium mb-1">Method Details:</h4>
              <p className="text-sm text-muted-foreground">
                Showing performance of <span className="font-semibold">{formatMethodName(selectedMethod)}</span> across different models and hardware configurations.
              </p>
            </div>
        )}
      </div>
  );
}