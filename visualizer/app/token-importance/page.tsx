"use client"

import { useState, useEffect } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import { AlertCircle, FileText, Info } from 'lucide-react'
import { 
  getModels, 
  getModelMethods, 
  getAggregationMethods, 
  getTokenImportanceAcrossFiles,
  getModelMethodFiles
} from '@/lib/api'
import { AggregationMethod, TokenImportanceData } from '@/lib/types'
import { cleanTokenText } from '@/lib/utils'

export default function TokenImportancePage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  
  const [models, setModels] = useState<string[]>([])
  const [methods, setMethods] = useState<string[]>([])
  const [aggregationMethods, setAggregationMethods] = useState<AggregationMethod[]>([])
  const [fileCount, setFileCount] = useState<number>(0)
  
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [selectedMethod, setSelectedMethod] = useState<string>('')
  const [selectedAggregation, setSelectedAggregation] = useState<AggregationMethod>('sum')
  
  const [tokenImportance, setTokenImportance] = useState<TokenImportanceData[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Load initial data
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const [modelsData, aggregationsData] = await Promise.all([
          getModels(),
          getAggregationMethods()
        ])
        
        setModels(modelsData)
        setAggregationMethods(aggregationsData.methods)
        setSelectedAggregation(aggregationsData.default)
        
        // Get URL params
        const modelParam = searchParams.get('model')
        const methodParam = searchParams.get('method')
        const aggregationParam = searchParams.get('aggregation') as AggregationMethod
        
        // Set initial selections from URL params if available
        if (modelParam && modelsData.includes(modelParam)) {
          setSelectedModel(modelParam)
          const methodsForModel = await getModelMethods(modelParam)
          setMethods(methodsForModel)
          
          if (methodParam && methodsForModel.includes(methodParam)) {
            setSelectedMethod(methodParam)
            
            if (aggregationParam && aggregationsData.methods.includes(aggregationParam)) {
              setSelectedAggregation(aggregationParam)
            }
            
            // Load token importance data
            loadTokenImportanceData(modelParam, methodParam, aggregationParam || aggregationsData.default)
          }
        } else if (modelsData.length > 0) {
          setSelectedModel(modelsData[0])
          const methodsForModel = await getModelMethods(modelsData[0])
          setMethods(methodsForModel)
        }
      } catch (e) {
        setError("Failed to load initial data. Please check if the API server is running.")
        console.error("Initial data load error:", e)
      }
    }
    
    fetchInitialData()
  }, [searchParams])
  
  // When model selection changes, update methods
  useEffect(() => {
    if (selectedModel) {
      const fetchMethods = async () => {
        try {
          const methodsData = await getModelMethods(selectedModel)
          setMethods(methodsData)
          
          if (methodsData.length > 0 && !methodsData.includes(selectedMethod)) {
            setSelectedMethod(methodsData[0])
          }
        } catch (e) {
          setError(`Failed to load methods for model ${selectedModel}.`)
          console.error("Methods load error:", e)
        }
      }
      
      fetchMethods()
    }
  }, [selectedModel])
  
  // Load token importance data when model, method, or aggregation changes
  useEffect(() => {
    if (selectedModel && selectedMethod && selectedAggregation) {
      loadTokenImportanceData(selectedModel, selectedMethod, selectedAggregation)
      
      // Update URL params
      const params = new URLSearchParams()
      params.set('model', selectedModel)
      params.set('method', selectedMethod)
      params.set('aggregation', selectedAggregation)
      router.push(`/token-importance?${params.toString()}`)
    }
  }, [selectedModel, selectedMethod, selectedAggregation])
  
  const loadTokenImportanceData = async (model: string, method: string, aggregation: AggregationMethod) => {
    setIsLoading(true)
    setError(null)
    try {
      // Get file count for information display
      const filesData = await getModelMethodFiles(model, method)
      const fileCount = filesData.files?.length || 0
      setFileCount(fileCount)
      
      console.log(`Found ${fileCount} files for ${model}/${method}`, 
      { files: filesData.files, modelId: model, methodId: method })
      
      if (fileCount === 0) {
        setError("No files found for the selected model and method.")
        setIsLoading(false)
        return
      }
      
      // Get token importance data
      const data = await getTokenImportanceAcrossFiles(model, method, aggregation)
      console.log(`Processed ${data.length} unique tokens across files`)
      setTokenImportance(data)
      
      if (data.length === 0) {
        setError("No token importance data found. This could be because the files don't contain attribution data.")
      }
    } catch (error: any) {
      console.error('Error loading token importance data:', error)
      setError(`Failed to load token importance data: ${error.message || "Unknown error"}`)
    } finally {
      setIsLoading(false)
    }
  }
  
  // Clean and format token for display
  const formatToken = (token: string) => {
    const cleaned = cleanTokenText(token)
    if (!cleaned.trim()) return '<empty>'
    return cleaned
  }
  
  return (
    <div className="container py-8">
      <h1 className="text-3xl font-bold mb-6">Token Importance Across Files</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Model</CardTitle>
          </CardHeader>
          <CardContent>
            <Select
              value={selectedModel}
              onValueChange={setSelectedModel}
              disabled={isLoading || models.length === 0}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select a model" />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  {models.map(model => (
                    <SelectItem key={model} value={model}>
                      {model}
                    </SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Attribution Method</CardTitle>
          </CardHeader>
          <CardContent>
            <Select
              value={selectedMethod}
              onValueChange={setSelectedMethod}
              disabled={!selectedModel || isLoading || methods.length === 0}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select a method" />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  {methods.map(method => (
                    <SelectItem key={method} value={method}>
                      {method}
                    </SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Aggregation Method</CardTitle>
          </CardHeader>
          <CardContent>
            <Select
              value={selectedAggregation}
              onValueChange={(value) => setSelectedAggregation(value as AggregationMethod)}
              disabled={isLoading || aggregationMethods.length === 0}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select aggregation" />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  {aggregationMethods.map(method => (
                    <SelectItem key={method} value={method}>
                      {method}
                    </SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
          </CardContent>
        </Card>
      </div>
      
      {error && (
        <div className="mb-6 p-4 border rounded-md bg-red-50 text-red-800 flex items-start gap-3">
          <AlertCircle className="h-5 w-5 mt-0.5 flex-shrink-0" />
          <div>
            <p className="font-medium">Error</p>
            <p>{error}</p>
          </div>
        </div>
      )}
      
      {fileCount > 0 && !error && (
        <div className="mb-6 p-4 border rounded-md bg-blue-50 text-blue-800 flex items-start gap-3">
          <Info className="h-5 w-5 mt-0.5 flex-shrink-0" />
          <div>
            <p className="font-medium">Analysis Information</p>
            <p>Analyzing token importance across {fileCount} files using {selectedAggregation} aggregation.</p>
          </div>
        </div>
      )}
      
      <Card className="mb-8">
        <CardHeader>
          <CardTitle>Token Importance Visualization</CardTitle>
          <CardDescription>
            Shows tokens ranked by their importance across all analyzed files
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-4 py-4">
              {Array(10).fill(0).map((_, i) => (
                <div key={i} className="flex items-center gap-4">
                  <Skeleton className="h-5 w-[150px]" />
                  <Skeleton className="h-4 w-full" />
                </div>
              ))}
            </div>
          ) : tokenImportance.length > 0 ? (
            <div className="grid gap-2 max-h-[600px] overflow-y-auto pr-2">
              {tokenImportance.map((item, index) => (
                <div 
                  key={index} 
                  className="flex items-center gap-4 p-2 rounded hover:bg-muted transition-colors"
                >
                  <div className="flex-none w-[30px] text-center font-mono text-sm text-muted-foreground">
                    {index + 1}
                  </div>
                  <div className="flex-none w-[200px] overflow-hidden">
                    <div className="font-medium truncate" title={item.token}>
                      {formatToken(item.token)}
                    </div>
                    <div className="flex items-center text-xs text-muted-foreground gap-1">
                      <FileText className="h-3 w-3" /> 
                      <span>Found in {Object.keys(item.importances).length} files</span>
                    </div>
                  </div>
                  <div className="flex-1">
                    <div className="h-4 w-full bg-muted rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-orange-500" 
                        style={{ 
                          width: `${Math.min(100, (item.importance || 0) * 100)}%` 
                        }}
                      />
                    </div>
                    <div className="text-right text-xs mt-1">
                      {item.importance !== undefined ? 
                        (item.importance * 100).toFixed(2) : '0.00'}%
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : !error ? (
            <div className="h-96 flex items-center justify-center">
              <p className="text-center text-muted-foreground">
                No token importance data available for the selected configuration.<br />
                Please select a different model, method, or aggregation.
              </p>
            </div>
          ) : null}
        </CardContent>
      </Card>
    </div>
  )
}