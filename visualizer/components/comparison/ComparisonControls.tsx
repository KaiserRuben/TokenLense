"use client"

import React from 'react';

interface ControlsProps {
    comparisonType: string;
    leftSettings: {
        maxConnections: number;
        useRelativeStrength: boolean;
        showBackground: boolean;
        showConnections: boolean;
    };
    rightSettings: {
        maxConnections: number;
        useRelativeStrength: boolean;
        showBackground: boolean;
        showConnections: boolean;
    };
    setLeftSettings: React.Dispatch<React.SetStateAction<{
        maxConnections: number;
        useRelativeStrength: boolean;
        showBackground: boolean;
        showConnections: boolean;
    }>>;
    setRightSettings: React.Dispatch<React.SetStateAction<{
        maxConnections: number;
        useRelativeStrength: boolean;
        showBackground: boolean;
        showConnections: boolean;
    }>>;
}

const ComparisonControls: React.FC<ControlsProps> = ({
                                                         comparisonType,
                                                         leftSettings,
                                                         rightSettings,
                                                         setLeftSettings,
                                                         setRightSettings,
                                                     }) => {
    const leftLabel = comparisonType === 'models' ? 'Left Model' : 'Left Method';
    const rightLabel = comparisonType === 'models' ? 'Right Model' : 'Right Method';

    // Toggle settings for either side
    const toggleSettings = (side: 'left' | 'right', field: keyof typeof leftSettings) => {
        if (side === 'left') {
            setLeftSettings(prev => {
                // For visualization mode toggles, ensure only one is active
                if (field === 'showBackground' && !prev.showBackground) {
                    return { ...prev, showBackground: true, showConnections: false };
                }
                if (field === 'showConnections' && !prev.showConnections) {
                    return { ...prev, showConnections: true, showBackground: false };
                }
                // For other toggles, just flip the value
                return { ...prev, [field]: !prev[field] };
            });
        } else {
            setRightSettings(prev => {
                // For visualization mode toggles, ensure only one is active
                if (field === 'showBackground' && !prev.showBackground) {
                    return { ...prev, showBackground: true, showConnections: false };
                }
                if (field === 'showConnections' && !prev.showConnections) {
                    return { ...prev, showConnections: true, showBackground: false };
                }
                // For other toggles, just flip the value
                return { ...prev, [field]: !prev[field] };
            });
        }
    };

    // Update max connections
    const updateMaxConnections = (side: 'left' | 'right', value: number) => {
        if (side === 'left') {
            setLeftSettings(prev => ({ ...prev, maxConnections: value }));
        } else {
            setRightSettings(prev => ({ ...prev, maxConnections: value }));
        }
    };

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Left side controls */}
            <div className="p-4 space-y-4 rounded-lg border">
                <h3 className="text-base font-medium">{leftLabel} Visualization</h3>

                {/* Visualization Mode */}
                <div className="space-y-2">
                    <p className="text-sm font-medium">Visualization Mode</p>
                    <div className="flex gap-4">
                        <div className="flex items-center gap-2">
                            <input
                                type="radio"
                                id="leftBackgroundMode"
                                checked={leftSettings.showBackground}
                                onChange={() => toggleSettings('left', 'showBackground')}
                                className="w-4 h-4"
                            />
                            <label htmlFor="leftBackgroundMode" className="text-sm">
                                Background
                            </label>
                        </div>

                        <div className="flex items-center gap-2">
                            <input
                                type="radio"
                                id="leftConnectionsMode"
                                checked={leftSettings.showConnections}
                                onChange={() => toggleSettings('left', 'showConnections')}
                                className="w-4 h-4"
                            />
                            <label htmlFor="leftConnectionsMode" className="text-sm">
                                Connections
                            </label>
                        </div>
                    </div>
                </div>

                {/* Max Connections slider (only shown for connections mode) */}
                {leftSettings.showConnections && (
                    <div className="space-y-2">
                        <label className="text-sm font-medium block">
                            Max Connections: {leftSettings.maxConnections}
                        </label>
                        <input
                            type="range"
                            value={leftSettings.maxConnections}
                            onChange={(e) => updateMaxConnections('left', parseInt(e.target.value))}
                            min={1}
                            max={10}
                            step={1}
                            className="w-full"
                        />
                    </div>
                )}

                {/* Normalize strengths */}
                <div className="flex items-center gap-2">
                    <input
                        type="checkbox"
                        id="leftNormalizeToggle"
                        checked={leftSettings.useRelativeStrength}
                        onChange={() => toggleSettings('left', 'useRelativeStrength')}
                        className="w-4 h-4 rounded"
                    />
                    <label htmlFor="leftNormalizeToggle" className="text-sm">
                        Normalize strengths
                    </label>
                </div>
            </div>

            {/* Right side controls */}
            <div className="p-4 space-y-4 rounded-lg border">
                <h3 className="text-base font-medium">{rightLabel} Visualization</h3>

                {/* Visualization Mode */}
                <div className="space-y-2">
                    <p className="text-sm font-medium">Visualization Mode</p>
                    <div className="flex gap-4">
                        <div className="flex items-center gap-2">
                            <input
                                type="radio"
                                id="rightBackgroundMode"
                                checked={rightSettings.showBackground}
                                onChange={() => toggleSettings('right', 'showBackground')}
                                className="w-4 h-4"
                            />
                            <label htmlFor="rightBackgroundMode" className="text-sm">
                                Background
                            </label>
                        </div>

                        <div className="flex items-center gap-2">
                            <input
                                type="radio"
                                id="rightConnectionsMode"
                                checked={rightSettings.showConnections}
                                onChange={() => toggleSettings('right', 'showConnections')}
                                className="w-4 h-4"
                            />
                            <label htmlFor="rightConnectionsMode" className="text-sm">
                                Connections
                            </label>
                        </div>
                    </div>
                </div>

                {/* Max Connections slider (only shown for connections mode) */}
                {rightSettings.showConnections && (
                    <div className="space-y-2">
                        <label className="text-sm font-medium block">
                            Max Connections: {rightSettings.maxConnections}
                        </label>
                        <input
                            type="range"
                            value={rightSettings.maxConnections}
                            onChange={(e) => updateMaxConnections('right', parseInt(e.target.value))}
                            min={1}
                            max={10}
                            step={1}
                            className="w-full"
                        />
                    </div>
                )}

                {/* Normalize strengths */}
                <div className="flex items-center gap-2">
                    <input
                        type="checkbox"
                        id="rightNormalizeToggle"
                        checked={rightSettings.useRelativeStrength}
                        onChange={() => toggleSettings('right', 'useRelativeStrength')}
                        className="w-4 h-4 rounded"
                    />
                    <label htmlFor="rightNormalizeToggle" className="text-sm">
                        Normalize strengths
                    </label>
                </div>
            </div>
        </div>
    );
};

export default ComparisonControls;