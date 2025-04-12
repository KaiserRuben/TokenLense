# TokenLense Implementation Blueprint

## API Endpoints and Corresponding Visualizations

### 1. Models Endpoint (`/models/`)

**URL:** `/models/`  
**Component:** `ModelSelector`

The model selection interface will:
- Display available models in a grid of cards
- Provide search/filter functionality
- Show model metadata when available
- Use a responsive layout with shadcn/ui components

```tsx
// Simplified component structure
export function ModelSelector() {
  const [models, setModels] = useState<string[]>([]);
  
  useEffect(() => {
    async function loadModels() {
      const modelData = await getModels();
      setModels(modelData);
    }
    
    loadModels();
  }, []);
  
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {models.map(model => (
        <Card key={model}>
          <CardHeader>
            <CardTitle>{model}</CardTitle>
          </CardHeader>
          <CardContent>Model details here</CardContent>
          <CardFooter>
            <Button asChild>
              <Link href={`/models/${model}`}>Explore Methods</Link>
            </Button>
          </CardFooter>
        </Card>
      ))}
    </div>
  );
}
```

### 2. Methods Endpoint (`/models/{model}/methods`)

**URL:** `/models/{model}/methods`  
**Component:** `MethodSelector`

The method selection component will:
- List available attribution methods for the selected model
- Provide brief descriptions of each method
- Allow users to select a method to view attribution data
- Support method comparison selection

```tsx
// Simplified component structure
export function MethodSelector({ model }: { model: string }) {
  const [methods, setMethods] = useState<string[]>([]);
  
  useEffect(() => {
    async function loadMethods() {
      const methodData = await getModelMethods(model);
      setMethods(methodData);
    }
    
    loadMethods();
  }, [model]);
  
  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold">Attribution Methods for {model}</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {methods.map(method => (
          <Card key={method}>
            <CardHeader>
              <CardTitle>{method}</CardTitle>
            </CardHeader>
            <CardContent>
              <p>Method description here</p>
            </CardContent>
            <CardFooter>
              <Button asChild>
                <Link href={`/models/${model}/methods/${method}/files`}>
                  View Files
                </Link>
              </Button>
            </CardFooter>
          </Card>
        ))}
      </div>
    </div>
  );
}
```

### 3. Files Endpoint (`/models/{model}/methods/{method}/files`)

**URL:** `/models/{model}/methods/{method}/files`  
**Component:** `FileSelector`

The file selection component will:
- List available attribution files for the selected model and method
- Show file metadata (prompts, timestamps)
- Allow users to select a file to view attribution data

```tsx
// Simplified component structure
export function FileSelector({ 
  model, 
  method 
}: { 
  model: string;
  method: string;
}) {
  const [files, setFiles] = useState<any[]>([]);
  
  useEffect(() => {
    async function loadFiles() {
      const response = await fetch(`/api/models/${model}/methods/${method}/files`);
      const data = await response.json();
      setFiles(data.files);
    }
    
    loadFiles();
  }, [model, method]);
  
  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold">Attribution Files</h2>
      <div className="space-y-2">
        {files.map((file, index) => (
          <Card key={index}>
            <CardContent className="p-4">
              <div className="flex justify-between">
                <div>
                  <h3 className="font-medium">{file.prompt}</h3>
                  <p className="text-sm text-muted-foreground">{file.timestamp}</p>
                </div>
                <Button asChild>
                  <Link href={`/attribution/${model}/${method}/${index}`}>
                    View Attribution
                  </Link>
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
```

### 4. Attribution Endpoint (`/attribution/{model}/{method}/{file_id}`)

**URL:** `/attribution/{model}/{method}/{file_id}`  
**Component:** `TokenCloud` (enhanced `WordCloud`)

The main attribution visualization will:
- Display the source and target tokens
- Show attribution relationships with interactive connections
- Support different aggregation methods
- Allow adjusting the context window
- Provide token importance visualization

```tsx
// Simplified component structure
export function TokenCloud({
  model,
  method,
  fileId,
  aggregation = "sum"
}: {
  model: string;
  method: string;
  fileId: number;
  aggregation?: string;
}) {
  const [attribution, setAttribution] = useState<AttributionData | null>(null);
  const [contextSize, setContextSize] = useState(10);
  
  useEffect(() => {
    async function loadAttribution() {
      const response = await fetch(
        `/api/attribution/${model}/${method}/${fileId}?aggregation=${aggregation}`
      );
      const data = await response.json();
      setAttribution(data);
    }
    
    loadAttribution();
  }, [model, method, fileId, aggregation]);
  
  if (!attribution) return <div>Loading...</div>;
  
  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Token Attribution</h2>
        <div className="flex items-center gap-2">
          <span>Context Size:</span>
          <Slider
            value={[contextSize]}
            onValueChange={([value]) => setContextSize(value)}
            min={5}
            max={20}
            step={1}
          />
          <span>{contextSize}</span>
        </div>
      </div>
      
      {/* Token visualization component */}
      <div className="border rounded-lg p-4">
        {/* Source tokens section */}
        <div className="mb-4">
          <h3 className="text-sm font-medium text-muted-foreground mb-2">
            Source Tokens
          </h3>
          <div className="flex flex-wrap gap-2">
            {attribution.source_tokens.map((token, index) => (
              <div key={index} className="px-2 py-1 bg-muted rounded">
                {token}
              </div>
            ))}
          </div>
        </div>
        
        {/* Target tokens section */}
        <div>
          <h3 className="text-sm font-medium text-muted-foreground mb-2">
            Target Tokens
          </h3>
          <div className="flex flex-wrap gap-2">
            {attribution.target_tokens.map((token, index) => (
              <div key={index} className="px-2 py-1 bg-primary/10 rounded">
                {token}
              </div>
            ))}
          </div>
        </div>
        
        {/* SVG for connections would be rendered here */}
      </div>
    </div>
  );
}
```

### 5. Method Comparison Component

**URL:** `/attribution/compare`  
**Component:** `MethodComparison`

The method comparison component will:
- Allow toggling between multiple attribution methods
- Show combined token importance with color differentiation
- Display differences in attribution patterns

```tsx
// Simplified component structure
export function MethodComparison({
  model,
  fileId,
  methods = []
}: {
  model: string;
  fileId: number;
  methods: string[];
}) {
  const [attributions, setAttributions] = useState<Record<string, AttributionData>>({});
  
  useEffect(() => {
    async function loadAttributions() {
      const results: Record<string, AttributionData> = {};
      
      for (const method of methods) {
        const response = await fetch(
          `/api/attribution/${model}/${method}/${fileId}`
        );
        results[method] = await response.json();
      }
      
      setAttributions(results);
    }
    
    if (methods.length > 0) {
      loadAttributions();
    }
  }, [model, fileId, methods]);
  
  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold">Method Comparison</h2>
      
      {/* Method selection */}
      <div className="flex flex-wrap gap-2">
        {methods.map(method => (
          <Badge key={method} variant="outline">
            {method}
          </Badge>
        ))}
      </div>
      
      {/* Visualization would render here */}
      <div className="border rounded-lg p-4">
        {/* Combined token visualization component */}
        {/* This would render a combined view with color-coded attributions */}
      </div>
    </div>
  );
}
```

### 6. Model Comparison Component

**URL:** `/attribution/compare`  
**Component:** `ModelComparison`

The model comparison component will:
- Show a split view of attribution patterns between models
- Highlight differences in attribution
- Allow synchronized navigation through tokens

```tsx
// Simplified component structure
export function ModelComparison({
  models,
  method,
  fileId
}: {
  models: string[];
  method: string;
  fileId: number;
}) {
  const [attributions, setAttributions] = useState<Record<string, AttributionData>>({});
  
  useEffect(() => {
    async function loadAttributions() {
      const results: Record<string, AttributionData> = {};
      
      for (const model of models) {
        const response = await fetch(
          `/api/attribution/${model}/${method}/${fileId}`
        );
        results[model] = await response.json();
      }
      
      setAttributions(results);
    }
    
    if (models.length > 0) {
      loadAttributions();
    }
  }, [models, method, fileId]);
  
  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold">Model Comparison</h2>
      
      {/* Model selection */}
      <div className="flex flex-wrap gap-2">
        {models.map(model => (
          <Badge key={model} variant="outline">
            {model}
          </Badge>
        ))}
      </div>
      
      {/* Split view visualization */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {models.map(model => (
          <div key={model} className="border rounded-lg p-4">
            <h3 className="font-medium mb-2">{model}</h3>
            {/* Individual model visualization would render here */}
          </div>
        ))}
      </div>
    </div>
  );
}
```

## Data Fetching and State Management Patterns

### API Client

We'll implement a comprehensive API client to handle all API requests:

```tsx
// lib/api.ts
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Fetch wrapper with error handling
async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, options);
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }
    
    return await response.json() as T;
  } catch (error) {
    console.error('API fetch error:', error);
    throw error;
  }
}

export function getModels() {
  return fetchAPI<{models: string[]}>('/models/');
}

export function getModelMethods(model: string) {
  return fetchAPI<{model: string, methods: string[]}>(`/models/${model}/methods`);
}

export function getModelMethodFiles(model: string, method: string) {
  return fetchAPI<{model: string, method: string, files: string[]}>(`/models/${model}/methods/${method}/files`);
}

export function getAttribution(model: string, method: string, fileId: number, aggregation = 'sum') {
  return fetchAPI<AttributionData>(`/attribution/${model}/${method}/${fileId}?aggregation=${aggregation}`);
}

export function getDetailedAttribution(model: string, method: string, fileId: number, aggregation = 'sum') {
  return fetchAPI<AttributionDetailedData>(`/attribution/${model}/${method}/${fileId}/detailed?aggregation=${aggregation}`);
}

export function compareAttributions(files: string[], aggregation = 'sum') {
  const params = new URLSearchParams();
  files.forEach(file => params.append('files', file));
  params.append('aggregation', aggregation);
  
  return fetchAPI<AttributionComparisonData>(`/attribution/compare?${params.toString()}`);
}

export function getTokenImportance(files: string[], tokenIndex: number, isTarget = true, aggregation = 'sum') {
  const params = new URLSearchParams();
  files.forEach(file => params.append('files', file));
  params.append('token_index', tokenIndex.toString());
  params.append('is_target', isTarget.toString());
  params.append('aggregation', aggregation);
  
  return fetchAPI<TokenImportanceData>(`/attribution/token_importance?${params.toString()}`);
}

export function getSystemInfo() {
  return fetchAPI<SystemInfo>('/performance/system');
}

export function getTimingResults() {
  return fetchAPI<TimingResults>('/performance/timing');
}
```

### React Query for Data Fetching

For more efficient data fetching and caching, we'll use React Query:

```tsx
// hooks/useAttribution.ts
import { useQuery } from '@tanstack/react-query';
import { getAttribution } from '@/lib/api';
import type { AttributionData } from '@/lib/types';

export function useAttribution(model: string, method: string, fileId: number, aggregation = 'sum') {
  return useQuery<AttributionData>({
    queryKey: ['attribution', model, method, fileId, aggregation],
    queryFn: () => getAttribution(model, method, fileId, aggregation),
    enabled: !!model && !!method && fileId !== undefined,
  });
}
```

## WordCloud Migration Plan

For migrating the `WordCloud` component:

1. **Create TypeScript interfaces** for all relevant data structures
2. **Port the core rendering logic** with proper types 
3. **Connect to the API** using React Query hooks
4. **Add the enhanced features** one by one

### Implementation approach:

1. Start with a basic version that displays tokens with proper styling
2. Add the interactive connection visualization
3. Implement token importance calculations and visualization
4. Add context window adjustment
5. Implement method selection and color differentiation
6. Add split view for model comparison

## TypeScript Interfaces

We'll define comprehensive TypeScript interfaces for all API responses:

```tsx
// lib/types.ts

// Basic token representation
export interface Token {
  token: string;
  id: number;
}

// Enhanced token with additional metadata
export interface TokenWithId {
  id: number;
  token: string;
}

// Available aggregation methods
export type AggregationMethod = 'sum' | 'mean' | 'l2_norm' | 'abs_sum' | 'max';

// Attribution matrix information
export interface AttributionMatrixInfo {
  shape: number[];
  dtype: string;
  is_attention: boolean;
  tensor_type: string;
}

// Attribution response from API
export interface AttributionResponse {
  model: string;
  method: string;
  file_id: number;
  prompt: string;
  generation: string;
  source_tokens: string[];
  target_tokens: string[];
  attribution_matrix: number[][];
  aggregation: string;
}

// Detailed attribution with token IDs
export interface AttributionDetailedResponse extends Omit<AttributionResponse, 'source_tokens' | 'target_tokens'> {
  source_tokens: TokenWithId[];
  target_tokens: TokenWithId[];
  matrix_info: AttributionMatrixInfo;
  exec_time: number;
  original_attribution_shape: number[];
}

// Data for token importance visualization
export interface TokenImportance {
  token: string;
  token_index: number;
  importance: number;
  method: string;
  model: string;
}

// Processed data for visualization
export interface VisualizationData {
  sourceTokens: {
    token: string;
    id: number;
    importance: number;
  }[];
  targetTokens: {
    token: string;
    id: number;
    importance: number;
  }[];
  attributionMatrix: number[][];
  normalizedMatrix: number[][];
  model: string;
  method: string;
}
```

## Implementation Concerns and Questions

1. **Data transformation**: How to efficiently transform API response data to match the expected format for visualizations?
2. **Performance**: How to handle potentially large attribution matrices without impacting UI performance?
3. **Interactivity**: What's the best approach to implement the interactive token exploration while maintaining good performance?
4. **Context window**: How to implement the context window adjustment that works with the existing visualization?
5. **Method comparison**: What's the most effective way to visually distinguish between different attribution methods?

These concerns will be addressed during implementation, with specific solutions tailored to each challenge.