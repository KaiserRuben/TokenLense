"use client"

import { useState, useEffect } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { 
  getModels, 
  getModelMethods, 
  getAggregationMethods, 
  getTokenImportanceAcrossFiles 
} from '@/lib/api'
import { AggregationMethod, TokenImportanceData } from '@/lib/types'

export default function TokenImportancePage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  
  const [models, setModels] = useState<string[]>([])
  const [methods, setMethods] = useState<string[]>([])
  const [aggregationMethods, setAggregationMethods] = useState<AggregationMethod[]>([])
  
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [selectedMethod, setSelectedMethod] = useState<string>('')
  const [selectedAggregation, setSelectedAggregation] = useState<AggregationMethod>('sum')
  
  const [tokenImportance, setTokenImportance] = useState<TokenImportanceData[]>([])
  const [isLoading, setIsLoading] = useState(false)

  // Load initial data
  useEffect(() => {
    const fetchInitialData = async () => {
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
    }
    
    fetchInitialData()
  }, [searchParams])
  
  // When model selection changes, update methods
  useEffect(() => {
    if (selectedModel) {
      const fetchMethods = async () => {
        const methodsData = await getModelMethods(selectedModel)
        setMethods(methodsData)
        
        if (methodsData.length > 0 && !methodsData.includes(selectedMethod)) {
          setSelectedMethod(methodsData[0])
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
    try {
      const data = await getTokenImportanceAcrossFiles(model, method, aggregation)
      setTokenImportance(data)
    } catch (error) {
      console.error('Error loading token importance data:', error)
    } finally {
      setIsLoading(false)
    }
  }
  
  return (
    <div className="container py-8">
      <h1 className="text-3xl font-bold mb-6">Token Importance Across Files</h1>
      
      <div className="grid grid-cols-3 gap-4 mb-8">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Model</CardTitle>
          </CardHeader>
          <CardContent>
            <Select
              value={selectedModel}
              onValueChange={setSelectedModel}
              disabled={isLoading}
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
              disabled={!selectedModel || isLoading}
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
              disabled={isLoading}
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
      
      <Card className="mb-8">
        <CardHeader>
          <CardTitle>Token Importance Visualization</CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="h-96 flex items-center justify-center">
              <p>Loading token importance data...</p>
            </div>
          ) : tokenImportance.length > 0 ? (
            <div className="grid gap-2 max-h-[600px] overflow-y-auto">
              {tokenImportance.map((item, index) => (
                <div 
                  key={index} 
                  className="flex items-center gap-2 p-2 rounded hover:bg-muted"
                >
                  <div className="flex-1">
                    <div className="font-medium">{item.token}</div>
                    <div className="text-sm text-muted-foreground">Found in {Object.keys(item.importances).length} files</div>
                  </div>
                  <div className="flex-1">
                    <div className="h-4 w-full bg-muted rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-orange-500" 
                        style={{ 
                          width: `${Math.min(100, item.importance * 100)}%` 
                        }}
                      />
                    </div>
                    <div className="text-right text-xs mt-1">{(item.importance * 100).toFixed(2)}%</div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="h-96 flex items-center justify-center">
              <p>No token importance data available for the selected configuration.</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}