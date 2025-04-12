"use client"

import { useEffect, useState } from 'react'
import {
    Sheet,
    SheetContent,
    SheetHeader,
    SheetTitle,
    SheetDescription,
    SheetFooter
} from '@/components/ui/sheet'
import { Button } from '@/components/ui/button'
import { ChevronRight, ChevronLeft, X } from 'lucide-react'

const ONBOARDING_KEY = 'tokenlense-onboarding-viewed'

interface OnboardingProps {
    path: string
}

interface StepInfo {
    title: string
    content: React.ReactNode
    image?: string
}

// Define onboarding categories
const ONBOARDING_CATEGORIES = {
    HOME: 'home',
    MODELS: 'models',
    PERFORMANCE: 'performance',
    COMPARE: 'compare',
    TOKEN_IMPORTANCE: 'token_importance',
    ATTRIBUTION: 'attribution', // Added attribution category
}

// Map paths to categories
const getOnboardingCategory = (path: string): string => {
    if (path === '/') return ONBOARDING_CATEGORIES.HOME
    if (path.startsWith('/models')) return ONBOARDING_CATEGORIES.MODELS
    if (path.startsWith('/performance')) return ONBOARDING_CATEGORIES.PERFORMANCE
    if (path.startsWith('/compare') || path.startsWith('/compare-selector')) return ONBOARDING_CATEGORIES.COMPARE
    if (path.startsWith('/token-importance')) return ONBOARDING_CATEGORIES.TOKEN_IMPORTANCE
    if (path.startsWith('/attribution')) return ONBOARDING_CATEGORIES.ATTRIBUTION // Added attribution path
    return path // Fallback to exact path if no category match
}

// Helper function to strip URL parameters for consistent path tracking
const getNormalizedPath = (path: string): string => {
    // Remove URL parameters if present
    return path.split('?')[0]
}

export default function Onboarding({ path }: OnboardingProps) {
    const [showOnboarding, setShowOnboarding] = useState(false)
    const [currentStep, setCurrentStep] = useState(0)
    const [isNewUser, setIsNewUser] = useState(false)
    const normalizedPath = getNormalizedPath(path)
    const category = getOnboardingCategory(normalizedPath)

    // Check if onboarding should be shown on this page
    useEffect(() => {
        // Skip onboarding if running on server-side
        if (typeof window === 'undefined') return;

        try {
            // Check if this is a first-time user (no onboarding viewed yet)
            const viewedCategories = JSON.parse(localStorage.getItem(ONBOARDING_KEY) || '{}')
            const isFirstTimeUser = Object.keys(viewedCategories).length === 0
            setIsNewUser(isFirstTimeUser)

            // Check if this category has been viewed before
            if (!viewedCategories[category]) {
                setShowOnboarding(true)
                // Reset to first step when opening
                setCurrentStep(0)
                // Don't mark as viewed yet - only do that when they finish or dismiss
            }
        } catch (error) {
            console.error('Error checking onboarding state:', error)
            // In case of an error, reset the localStorage
            localStorage.setItem(ONBOARDING_KEY, '{}')
        }
    }, [category])

    // Mark category as viewed when onboarding is closed
    const handleClose = () => {
        setShowOnboarding(false)

        // Mark this category as viewed in localStorage
        const viewedCategories = JSON.parse(localStorage.getItem(ONBOARDING_KEY) || '{}')
        const updatedViewedCategories = { ...viewedCategories, [category]: true }
        localStorage.setItem(ONBOARDING_KEY, JSON.stringify(updatedViewedCategories))
    }

    // Handle completing the current step
    const handleComplete = () => {
        handleClose()

        // For new users who complete the home page onboarding,
        // don't mark other categories as viewed to ensure they get contextual help
    }

    // Get steps based on current page category
    const getStepsForPage = (): StepInfo[] => {
        // Root/home page
        if (category === ONBOARDING_CATEGORIES.HOME) {
            return [
                {
                    title: 'Welcome to TokenLense',
                    content: (
                        <div className="space-y-4">
                            <p>
                                TokenLense helps you visualize and understand how language models attribute
                                importance between tokens in text generation.
                            </p>
                            <p>
                                This powerful tool reveals insights into model behavior that might not be
                                apparent when looking at individual outputs, helping researchers identify
                                patterns in how models process information.
                            </p>
                        </div>
                    ),
                    image: '/onboarding/welcome.svg'
                },
                {
                    title: 'Platform Features',
                    content: (
                        <div className="space-y-4">
                            <p>
                                TokenLense offers several key features to explore language model behavior:
                            </p>
                            <ul className="list-disc list-inside space-y-1">
                                <li><strong>Model Explorer</strong>: Select different language models</li>
                                <li><strong>Attribution Visualizations</strong>: See token relationships</li>
                                <li><strong>Performance Analysis</strong>: Compare method efficiency</li>
                                <li><strong>Token Importance</strong>: Identify globally important tokens</li>
                                <li><strong>Side-by-Side Comparison</strong>: Compare different models or methods</li>
                            </ul>
                        </div>
                    )
                },
                {
                    title: 'Getting Started',
                    content: (
                        <div className="space-y-4">
                            <p>
                                Start by selecting a language model from the grid below. Each card
                                represents a different model that you can explore.
                            </p>
                            <p>
                                After selecting a model, you'll choose an attribution method to visualize
                                token relationships and gain insights into how the model processes information.
                            </p>
                        </div>
                    ),
                    image: '/onboarding/select-model.svg'
                }
            ]
        }

        // Models page
        else if (category === ONBOARDING_CATEGORIES.MODELS) {
            return [
                {
                    title: 'Attribution Methods',
                    content: (
                        <div className="space-y-4">
                            <p>
                                Each model supports different attribution methods that reveal how
                                tokens influence each other during text generation.
                            </p>
                            <p>
                                These methods use different techniques to calculate the importance of input
                                tokens to output tokens, helping you understand model behavior from different
                                perspectives.
                            </p>
                        </div>
                    ),
                    image: '/onboarding/methods.svg'
                },
                {
                    title: 'Choosing a Method',
                    content: (
                        <div className="space-y-4">
                            <p>
                                Different attribution methods reveal different aspects of model behavior:
                            </p>
                            <ul className="list-disc list-inside space-y-1">
                                <li><strong>Attention</strong>: Shows attention patterns between tokens</li>
                                <li><strong>Integrated Gradients</strong>: Path-based attribution with high accuracy</li>
                                <li><strong>Input × Gradient</strong>: Fast approximation of token importance</li>
                                <li><strong>LIME</strong>: Local model-agnostic explanations</li>
                            </ul>
                            <p className="mt-2">
                                Select a method to explore its visualization and see how it attributes token importance.
                            </p>
                        </div>
                    )
                }
            ]
        }

        // Performance page
        else if (category === ONBOARDING_CATEGORIES.PERFORMANCE) {
            return [
                {
                    title: 'Performance Dashboard',
                    content: (
                        <div className="space-y-4">
                            <p>
                                The Performance dashboard shows you how different attribution methods
                                perform across models and hardware configurations.
                            </p>
                            <p>
                                This information helps you select methods that balance accuracy and
                                computational efficiency for your specific use case.
                            </p>
                        </div>
                    ),
                    image: '/onboarding/performance.svg'
                },
                {
                    title: 'Key Metrics',
                    content: (
                        <div className="space-y-4">
                            <p>
                                The dashboard shows several important metrics:
                            </p>
                            <ul className="list-disc list-inside space-y-1">
                                <li><strong>Execution Time</strong>: How long each method takes to run</li>
                                <li><strong>Tokens per Second</strong>: Processing throughput for each method</li>
                                <li><strong>Success Rate</strong>: Reliability of each method</li>
                                <li><strong>Hardware Comparison</strong>: How performance varies across devices</li>
                            </ul>
                        </div>
                    )
                },
                {
                    title: 'Interactive Filters',
                    content: (
                        <div className="space-y-4">
                            <p>
                                Use the filters at the top to select specific models and methods you want to
                                compare. Click on a badge to toggle selection.
                            </p>
                            <p>
                                The charts will update dynamically to show only your selected data, making it
                                easier to focus on the methods and models that matter to you.
                            </p>
                        </div>
                    )
                }
            ]
        }

        // Compare page
        else if (category === ONBOARDING_CATEGORIES.COMPARE) {
            return [
                {
                    title: 'Comparison Tool',
                    content: (
                        <div className="space-y-4">
                            <p>
                                The Comparison view allows you to directly contrast how different models or
                                attribution methods interpret the same prompt.
                            </p>
                            <p>
                                This side-by-side view is powerful for identifying subtle differences in how
                                models or methods attribute importance to tokens.
                            </p>
                        </div>
                    ),
                    image: '/onboarding/compare.svg'
                },
                {
                    title: 'Comparison Types',
                    content: (
                        <div className="space-y-4">
                            <p>
                                You can set up three types of comparisons:
                            </p>
                            <ul className="list-disc list-inside space-y-1">
                                <li><strong>Models</strong>: Same method, different models</li>
                                <li><strong>Methods</strong>: Same model, different methods</li>
                                <li><strong>Both</strong>: Different models and methods</li>
                            </ul>
                            <p className="mt-2">
                                Each comparison type reveals different insights about language model behavior.
                            </p>
                        </div>
                    )
                },
                {
                    title: 'Interactive Comparison',
                    content: (
                        <div className="space-y-4">
                            <p>
                                When you hover or click on a token in either view, the selection is synchronized
                                across both sides, making it easy to compare relationships.
                            </p>
                            <p>
                                Use the controls to adjust visualization settings for each side independently,
                                such as connection lines, background highlighting, and maximum connections.
                            </p>
                        </div>
                    )
                }
            ]
        }

        // Token Importance page
        else if (category === ONBOARDING_CATEGORIES.TOKEN_IMPORTANCE) {
            return [
                {
                    title: 'Token Importance Analysis',
                    content: (
                        <div className="space-y-4">
                            <p>
                                This view shows which tokens have the highest importance across multiple files
                                for a selected model and attribution method.
                            </p>
                            <p>
                                By analyzing patterns across different inputs, you can discover which tokens
                                consistently have high influence in the model's decision-making process.
                            </p>
                        </div>
                    ),
                    image: '/onboarding/importance.svg'
                },
                {
                    title: 'Global Insights',
                    content: (
                        <div className="space-y-4">
                            <p>
                                Token Importance analysis helps you answer questions like:
                            </p>
                            <ul className="list-disc list-inside space-y-1">
                                <li>Which tokens are consistently influential across different inputs?</li>
                                <li>Are there patterns in how the model assigns importance?</li>
                                <li>How does token importance vary across models or methods?</li>
                            </ul>
                            <p className="mt-2">
                                These insights can reveal fundamental aspects of model behavior and biases.
                            </p>
                        </div>
                    )
                },
                {
                    title: 'Aggregation Methods',
                    content: (
                        <div className="space-y-4">
                            <p>
                                Select different aggregation methods to change how token importance is calculated:
                            </p>
                            <ul className="list-disc list-inside space-y-1">
                                <li><strong>Sum</strong>: Adds all attribution values (default)</li>
                                <li><strong>Mean</strong>: Averages attribution values</li>
                                <li><strong>L2 Norm</strong>: Uses square root of sum of squares</li>
                                <li><strong>Abs Sum</strong>: Sums absolute values</li>
                                <li><strong>Max</strong>: Uses maximum attribution value</li>
                            </ul>
                            <p className="mt-2">
                                Different aggregation methods highlight different aspects of token importance.
                            </p>
                        </div>
                    )
                }
            ]
        }

        // Attribution visualization page
        else if (category === ONBOARDING_CATEGORIES.ATTRIBUTION) {
            return [
                {
                    title: 'Token Attribution Visualization',
                    content: (
                        <div className="space-y-4">
                            <p>
                                You're now viewing the core visualization of TokenLense: a detailed view of
                                token relationships and attribution scores between input and output tokens.
                            </p>
                            <p>
                                This visualization shows how different parts of the input influenced specific
                                outputs according to the selected attribution method.
                            </p>
                        </div>
                    ),
                    image: '/onboarding/attribution.svg'
                },
                {
                    title: 'Interactive Token Exploration',
                    content: (
                        <div className="space-y-4">
                            <p>
                                <strong>Hover over any token</strong> to see its relationships with other tokens.
                                Input tokens (orange) show which outputs they influenced, while output tokens (blue)
                                show which inputs contributed to them.
                            </p>
                            <p>
                                <strong>Click on a token</strong> to lock the selection and explore the relationships
                                in detail. Click again to unlock.
                            </p>
                        </div>
                    )
                },
                {
                    title: 'Visualization Controls',
                    content: (
                        <div className="space-y-4">
                            <p>
                                Use the controls to customize your visualization:
                            </p>
                            <ul className="list-disc list-inside space-y-1">
                                <li><strong>Max Connections</strong>: Limit the number of connections shown (1-10)</li>
                                <li><strong>Show Background</strong>: Use background highlighting to show relationships</li>
                                <li><strong>Show Connections</strong>: Use lines to connect related tokens</li>
                                <li><strong>Relative Strength</strong>: Normalize connection strengths for easier comparison</li>
                            </ul>
                        </div>
                    )
                },
                {
                    title: 'Understanding The Results',
                    content: (
                        <div className="space-y-4">
                            <p>
                                The visualization shows which tokens have the strongest attribution relationships:
                            </p>
                            <ul className="list-disc list-inside space-y-1">
                                <li><strong>Stronger connections</strong> (thicker lines or darker highlights) indicate
                                    tokens with greater influence on each other</li>
                                <li><strong>Input → Output</strong> connections show how input tokens influenced
                                    specific parts of the generation</li>
                                <li><strong>Key patterns</strong> like attention to specific words or phrases can
                                    reveal model reasoning</li>
                            </ul>
                            <p className="mt-2">
                                Look for patterns in which tokens have the strongest relationships to gain insights
                                into how the model processes information.
                            </p>
                        </div>
                    )
                }
            ]
        }

        // Default empty steps if page doesn't match
        return []
    }

    const steps = getStepsForPage()
    const hasSteps = steps.length > 0

    // If no steps for this page or onboarding shouldn't be shown, return null
    if (!hasSteps || !showOnboarding) {
        return null
    }

    // Safety check - ensure currentStep is within bounds
    const safeCurrentStep = Math.min(currentStep, steps.length - 1)
    // If step index changed, update state
    if (safeCurrentStep !== currentStep) {
        setCurrentStep(safeCurrentStep)
    }

    const currentStepInfo = steps[safeCurrentStep]
    // Additional safety check
    if (!currentStepInfo) {
        console.error('No step info found for index:', safeCurrentStep)
        handleClose() // Close onboarding if we can't find step info
        return null
    }

    const isLastStep = safeCurrentStep === steps.length - 1

    return (
        <Sheet open={showOnboarding} onOpenChange={handleClose}>
            <SheetContent
                side="bottom"
                className="h-auto max-h-[80vh] sm:max-w-xl sm:rounded-t-xl mx-auto px-6 sm:px-8"
            >
                <SheetHeader className="text-center sm:text-left">
                    <SheetTitle className="text-xl">{currentStepInfo.title}</SheetTitle>
                    <SheetDescription>
                        Step {currentStep + 1} of {steps.length}
                    </SheetDescription>
                </SheetHeader>

                <div className="py-6 flex flex-col md:flex-row gap-6 items-center">
                    {currentStepInfo.image && (
                        <div className="w-40 h-40 flex-shrink-0 bg-muted rounded-lg flex items-center justify-center overflow-hidden">
                            <img
                                src={currentStepInfo.image}
                                alt={currentStepInfo.title}
                                className="max-w-full max-h-full object-contain p-2"
                            />
                        </div>
                    )}

                    <div className="flex-1">
                        {currentStepInfo.content}
                    </div>
                </div>

                <SheetFooter className="flex justify-between sm:justify-between border-t pt-4">
                    <div className="flex items-center gap-2">
                        {currentStep > 0 ? (
                            <Button
                                variant="outline"
                                size="sm"
                                onClick={() => setCurrentStep(prev => prev - 1)}
                            >
                                <ChevronLeft className="h-4 w-4 mr-1" /> Previous
                            </Button>
                        ) : (
                            <Button
                                variant="outline"
                                size="sm"
                                onClick={handleClose}
                            >
                                <X className="h-4 w-4 mr-1" /> Skip
                            </Button>
                        )}
                    </div>

                    <div className="flex items-center gap-2">
                        {!isLastStep ? (
                            <Button
                                variant="default"
                                size="sm"
                                onClick={() => setCurrentStep(prev => prev + 1)}
                            >
                                Next <ChevronRight className="h-4 w-4 ml-1" />
                            </Button>
                        ) : (
                            <Button
                                variant="default"
                                size="sm"
                                onClick={handleComplete}
                            >
                                Got it <X className="h-4 w-4 ml-1" />
                            </Button>
                        )}
                    </div>
                </SheetFooter>
            </SheetContent>
        </Sheet>
    )
}