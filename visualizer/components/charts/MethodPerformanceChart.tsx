"use client"

import { useMemo, useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { MethodPerformanceData } from '@/lib/types';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

type Props = {
  data: MethodPerformanceData[];
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

export default function MethodPerformanceChart({ data }: Props) {
  const [groupBy, setGroupBy] = useState<'model' | 'method'>('model');
  const [sortBy, setSortBy] = useState<'alphabetical' | 'performance'>('performance');

  // Define color palette with more distinct colors
  const colorPalette = [
    '#8884d8', '#82ca9d', '#ffc658', '#ff8042', '#0088fe',
    '#00C49F', '#FFBB28', '#FF8042', '#a4de6c', '#d0ed57',
    '#8dd1e1', '#83a6ed', '#8884d8', '#c49c94', '#e377c2'
  ];

  interface ProcessedDataItem {
    [key: string]: string | number;
    displayName: string;
    avgPerformance: number;
  }

// Process data to group by selected dimension and handle sorting
  const processedData = useMemo(() => {
    // Ensure we have data to process
    if (!data || data.length === 0) return [] as ProcessedDataItem[];

    // Get unique categories based on groupBy selection
    const categories = Array.from(new Set(data.map(d => d[groupBy])));

    // Get unique secondary categories (what we're not grouping by)
    const secondaryCategories = Array.from(new Set(data.map(d => d[groupBy === 'model' ? 'method' : 'model'])));

    // Create category mapping for colors
    const categoryColorMap: Record<string, string> = {};
    secondaryCategories.forEach((category, index) => {
      categoryColorMap[category] = colorPalette[index % colorPalette.length];
    });

    // Create aggregated data by groupBy selection
    const groupedData = categories.map(category => {
      const categoryData = data.filter(d => d[groupBy] === category);

      // Create a record for this category with average times
      const result: ProcessedDataItem = {
        [groupBy]: category,
        displayName: '',
        avgPerformance: 0
      };

      // Add a displayName property for better readability
      result.displayName = groupBy === 'method' ? formatMethodName(category as string) : category;

      // Add data for each secondary category
      secondaryCategories.forEach(secondaryCategory => {
        const filteredData = categoryData.filter(
            d => d[groupBy === 'model' ? 'method' : 'model'] === secondaryCategory
        );

        if (filteredData.length > 0) {
          // Calculate average time
          const avgTime = filteredData.reduce((sum, d) => sum + d.avg_time, 0) / filteredData.length;
          result[secondaryCategory] = avgTime;
        }
      });

      // For sorting by performance, add an avgPerformance property
      result.avgPerformance = Object.keys(result)
          .filter(key => key !== groupBy && key !== 'displayName' && key !== 'avgPerformance')
          .reduce((sum, key) => sum + (Number(result[key]) || 0), 0) / secondaryCategories.length;

      return result;
    });

    // Sort data based on sortBy selection
    if (sortBy === 'alphabetical') {
      return groupedData.sort((a, b) => {
        return a.displayName.localeCompare(b.displayName);
      });
    } else { // performance
      return groupedData.sort((a, b) => {
        return a.avgPerformance - b.avgPerformance;
      });
    }
  }, [data, groupBy, sortBy]);

  // Get secondary categories (for the bars)
  const secondaryCategories = useMemo(() => {
    if (!data || data.length === 0) return [];
    return Array.from(new Set(data.map(d => d[groupBy === 'model' ? 'method' : 'model'])));
  }, [data, groupBy]);

  // Check for empty data
  if (!data || data.length === 0 || !processedData || processedData.length === 0) {
    return (
        <div className="flex flex-col items-center justify-center h-[500px] bg-gray-50 border rounded-md">
          <p className="text-muted-foreground mb-2">No performance data available</p>
          <p className="text-xs text-muted-foreground">Try adjusting your filters</p>
        </div>
    );
  }

  // Prepare bars for the chart
  const renderBars = () => {
    return secondaryCategories.map((category, index) => {
      const color = colorPalette[index % colorPalette.length];
      const displayName = groupBy === 'method' ? category : formatMethodName(category as string);

      return (
          <Bar
              key={category}
              dataKey={category}
              name={displayName}
              fill={color}
              radius={[4, 4, 0, 0]}
          />
      );
    });
  };

  // Define proper tooltip types
  interface TooltipProps {
    active?: boolean;
    payload?: {
      name: string;
      value: number;
      color: string;
    }[];
    label?: string;
  }

  // Format tooltip to show time in seconds
  const CustomTooltip = ({ active, payload, label }: TooltipProps) => {
    if (active && payload && payload.length) {
      return (
          <div className="bg-white p-3 border border-gray-200 shadow-md rounded-md">
            <p className="font-bold mb-1">{label}</p>
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

  return (
      <div className="flex flex-col">
        <div className="flex flex-col sm:flex-row sm:justify-between mb-4 gap-2">
          <div className="flex items-center gap-2">
            <span className="text-sm">Group by:</span>
            <Select
                value={groupBy}
                onValueChange={(value) => setGroupBy(value as 'model' | 'method')}
            >
              <SelectTrigger className="w-[150px]">
                <SelectValue placeholder="Group by" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="model">Model</SelectItem>
                <SelectItem value="method">Method</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-sm">Sort by:</span>
            <Select
                value={sortBy}
                onValueChange={(value) => setSortBy(value as 'alphabetical' | 'performance')}
            >
              <SelectTrigger className="w-[150px]">
                <SelectValue placeholder="Sort by" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="alphabetical">Alphabetical</SelectItem>
                <SelectItem value="performance">Performance</SelectItem>
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
                height={100}
                tick={{ fontSize: 12 }}
                interval={0}
            />
            <YAxis
                label={{
                  value: 'Average time (seconds)',
                  angle: -90,
                  position: 'insideLeft',
                  style: { textAnchor: 'middle' }
                }}
                tickFormatter={(value) => `${value.toFixed(1)}s`}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
                wrapperStyle={{ paddingTop: 20 }}
                formatter={(value) => <span style={{ color: '#666', fontSize: 14 }}>{value}</span>}
            />
            {renderBars()}
          </BarChart>
        </ResponsiveContainer>
      </div>
  );
}