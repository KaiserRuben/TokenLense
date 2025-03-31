// src/components/SettingsPanel.tsx
import React from 'react';
import { Settings2, X, Sliders, Eye, BarChart4, RefreshCw } from 'lucide-react';
import {
    Popover,
    PopoverContent,
    PopoverTrigger
} from '@/components/ui/popover';
import { Button } from '@/components/ui/button';
import {
    Accordion,
    AccordionContent,
    AccordionItem,
    AccordionTrigger
} from '@/components/ui/accordion';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import {
    TokenImportanceSettings,
    ImportanceMethod,
    TokenImportanceWeights,
    DEFAULT_IMPORTANCE_SETTINGS
} from '@/utils/types';

interface ImportanceMethodInfo {
    id: ImportanceMethod;
    name: string;
    description: string;
}

const importanceMethods: ImportanceMethodInfo[] = [
    {
        id: 'association',
        name: 'Association Summation',
        description: 'Sums connection strengths across the association matrix'
    },
    {
        id: 'weighted',
        name: 'Weighted Association',
        description: 'Emphasizes stronger connections with quadratic weighting'
    },
    {
        id: 'centrality',
        name: 'Network Centrality',
        description: 'Measures importance based on connectivity patterns'
    },
    {
        id: 'positional',
        name: 'Position-Enhanced',
        description: 'Considers token position in context'
    },
    {
        id: 'comprehensive',
        name: 'Comprehensive (Default)',
        description: 'Combines multiple methods for balanced evaluation'
    }
];

interface SettingsPanelProps {
    settings: TokenImportanceSettings;
    onSettingsChange: (newSettings: TokenImportanceSettings) => void;
}

const SettingsPanel: React.FC<SettingsPanelProps> = ({ settings, onSettingsChange }) => {
    const updateSetting = <K extends keyof TokenImportanceSettings>(
        key: K,
        value: TokenImportanceSettings[K]
    ) => {
        onSettingsChange({
            ...settings,
            [key]: value
        });
    };

    const updateWeight = <K extends keyof TokenImportanceWeights>(
        weightKey: K,
        value: number
    ) => {
        onSettingsChange({
            ...settings,
            weights: {
                ...settings.weights,
                [weightKey]: value
            }
        });
    };

    const resetSettings = () => {
        onSettingsChange(DEFAULT_IMPORTANCE_SETTINGS);
    };

    return (
        <Popover>
            <PopoverTrigger asChild>
                <Button variant="ghost" size="icon" className="h-9 w-9 rounded-full">
                    <Settings2 className="h-4 w-4" />
                    <span className="sr-only">Open settings</span>
                </Button>
            </PopoverTrigger>
            <PopoverContent className="w-80 p-0" align="end">
                <div className="flex items-center justify-between border-b px-4 py-2">
                    <h4 className="font-medium flex items-center gap-2">
                        <Sliders className="h-4 w-4" />
                        Visualization Settings
                    </h4>
                    <Button variant="ghost" size="icon" className="h-7 w-7">
                        <X className="h-4 w-4" />
                        <span className="sr-only">Close</span>
                    </Button>
                </div>

                <div className="p-4 overflow-y-auto max-h-[500px]">
                    <Accordion type="single" collapsible className="w-full">
                        {/* Importance Calculation Method */}
                        <AccordionItem value="importance-method">
                            <AccordionTrigger className="text-sm font-medium">
                                <div className="flex items-center gap-2">
                                    <BarChart4 className="h-4 w-4" />
                                    <span>Importance Calculation</span>
                                </div>
                            </AccordionTrigger>
                            <AccordionContent>
                                <div className="space-y-4 pt-2">
                                    <RadioGroup
                                        value={settings.method}
                                        onValueChange={(value) => updateSetting('method', value as ImportanceMethod)}
                                        className="space-y-3"
                                    >
                                        {importanceMethods.map(method => (
                                            <div key={method.id} className="flex items-start space-x-2">
                                                <RadioGroupItem
                                                    value={method.id}
                                                    id={`method-${method.id}`}
                                                    className="mt-1"
                                                />
                                                <div className="grid gap-1">
                                                    <Label
                                                        htmlFor={`method-${method.id}`}
                                                        className="text-sm font-medium leading-none"
                                                    >
                                                        {method.name}
                                                    </Label>
                                                    <p className="text-xs text-gray-500 dark:text-gray-400">
                                                        {method.description}
                                                    </p>
                                                </div>
                                            </div>
                                        ))}
                                    </RadioGroup>

                                    {settings.method === 'comprehensive' && (
                                        <div className="pt-2 space-y-4">
                                            <Separator />
                                            <p className="text-xs font-medium">Weight Distribution</p>

                                            <div className="space-y-4">
                                                <div className="space-y-2">
                                                    <div className="flex justify-between text-xs">
                                                        <span>Association Weight</span>
                                                        <span>{settings.weights.association.toFixed(1)}</span>
                                                    </div>
                                                    <Slider
                                                        value={[settings.weights.association]}
                                                        onValueChange={([value]) => updateWeight('association', parseFloat(value.toFixed(1)))}
                                                        min={0}
                                                        max={1}
                                                        step={0.1}
                                                    />
                                                </div>

                                                <div className="space-y-2">
                                                    <div className="flex justify-between text-xs">
                                                        <span>Centrality Weight</span>
                                                        <span>{settings.weights.centrality.toFixed(1)}</span>
                                                    </div>
                                                    <Slider
                                                        value={[settings.weights.centrality]}
                                                        onValueChange={([value]) => updateWeight('centrality', parseFloat(value.toFixed(1)))}
                                                        min={0}
                                                        max={1}
                                                        step={0.1}
                                                    />
                                                </div>

                                                <div className="space-y-2">
                                                    <div className="flex justify-between text-xs">
                                                        <span>Positional Weight</span>
                                                        <span>{settings.weights.positional.toFixed(1)}</span>
                                                    </div>
                                                    <Slider
                                                        value={[settings.weights.positional]}
                                                        onValueChange={([value]) => updateWeight('positional', parseFloat(value.toFixed(1)))}
                                                        min={0}
                                                        max={1}
                                                        step={0.1}
                                                    />
                                                </div>

                                                <div className="pt-1">
                                                    <p className="text-xs text-gray-500 dark:text-gray-400">
                                                        Note: These weights don't need to sum to 1. Results will be normalized.
                                                    </p>
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </AccordionContent>
                        </AccordionItem>

                        {/* Visualization Options */}
                        <AccordionItem value="visualization">
                            <AccordionTrigger className="text-sm font-medium">
                                <div className="flex items-center gap-2">
                                    <Eye className="h-4 w-4" />
                                    <span>Visualization Options</span>
                                </div>
                            </AccordionTrigger>
                            <AccordionContent className="space-y-4 pt-2">
                                {/* System Tokens Toggle */}
                                <div className="flex items-center justify-between">
                                    <div className="space-y-0.5">
                                        <Label className="text-sm">Show System Tokens</Label>
                                        <p className="text-xs text-gray-500 dark:text-gray-400">
                                            Display technical tokens like separators and markers
                                        </p>
                                    </div>
                                    <Switch
                                        checked={settings.showSystemTokens}
                                        onCheckedChange={(value) => updateSetting('showSystemTokens', value)}
                                    />
                                </div>

                                {/* Max Connections */}
                                <div className="space-y-2">
                                    <div className="flex justify-between">
                                        <Label className="text-sm">Max Connections</Label>
                                        <Badge variant="outline">{settings.maxConnections}</Badge>
                                    </div>
                                    <Slider
                                        value={[settings.maxConnections]}
                                        onValueChange={([value]) => updateSetting('maxConnections', value)}
                                        min={1}
                                        max={20}
                                        step={1}
                                    />
                                    <p className="text-xs text-gray-500 dark:text-gray-400">
                                        Maximum number of connections to show per token
                                    </p>
                                </div>

                                {/* Highlight Threshold */}
                                <div className="space-y-2">
                                    <div className="flex justify-between">
                                        <Label className="text-sm">Highlight Threshold</Label>
                                        <Badge variant="outline">{settings.highlightThreshold.toFixed(2)}</Badge>
                                    </div>
                                    <Slider
                                        value={[settings.highlightThreshold]}
                                        onValueChange={([value]) => updateSetting('highlightThreshold', parseFloat(value.toFixed(2)))}
                                        min={0}
                                        max={0.5}
                                        step={0.01}
                                    />
                                    <p className="text-xs text-gray-500 dark:text-gray-400">
                                        Minimum importance value required to show background highlighting
                                    </p>
                                </div>
                            </AccordionContent>
                        </AccordionItem>
                    </Accordion>

                    <div className="mt-4 pt-2 border-t flex justify-end">
                        <Button
                            size="sm"
                            variant="ghost"
                            onClick={resetSettings}
                            className="flex items-center gap-1 text-xs"
                        >
                            <RefreshCw className="h-3 w-3" />
                            Reset to Defaults
                        </Button>
                    </div>
                </div>
            </PopoverContent>
        </Popover>
    );
};

export default SettingsPanel;