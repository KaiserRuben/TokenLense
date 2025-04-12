"use client"

import { Button } from '@/components/ui/button'
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
} from '@/components/ui/dialog'
import { RefreshCw } from 'lucide-react'
import { useState } from 'react'

const ONBOARDING_KEY = 'tokenlense-onboarding-viewed'

export default function ResetOnboarding() {
    const [open, setOpen] = useState(false)
    const [resetComplete, setResetComplete] = useState(false)

    const handleResetOnboarding = () => {
        try {
            // Clear onboarding state
            localStorage.removeItem(ONBOARDING_KEY)
            setResetComplete(true)

            // Reset status after a delay
            setTimeout(() => {
                setOpen(false)
                setTimeout(() => setResetComplete(false), 300) // After dialog close animation
            }, 1500)
        } catch (error) {
            console.error('Error resetting onboarding state:', error)
            // Fallback: set to empty object
            localStorage.setItem(ONBOARDING_KEY, '{}')
        }
    }

    return (
        <Dialog open={open} onOpenChange={setOpen}>
            <DialogTrigger asChild>
                <Button variant="outline" size="sm" className="flex items-center gap-2">
                    <RefreshCw className="h-4 w-4" />
                    <span>Reset Onboarding</span>
                </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-md">
                <DialogHeader>
                    <DialogTitle>Reset Onboarding Experience</DialogTitle>
                    <DialogDescription>
                        This will reset all onboarding guides so you can see them again on each page.
                    </DialogDescription>
                </DialogHeader>

                <div className="py-4">
                    {!resetComplete ? (
                        <p>
                            If you&#39;d like to see the onboarding guides again, you can reset them here.
                            This will clear your onboarding preferences and show the guides the next time
                            you visit each page.
                        </p>
                    ) : (
                        <div className="flex items-center justify-center py-2 text-green-600">
                            <p className="font-medium">Onboarding reset complete!</p>
                        </div>
                    )}
                </div>

                <DialogFooter>
                    <Button
                        variant="outline"
                        onClick={() => setOpen(false)}
                        disabled={resetComplete}
                    >
                        Cancel
                    </Button>
                    <Button
                        onClick={handleResetOnboarding}
                        disabled={resetComplete}
                    >
                        Reset Onboarding
                    </Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    )
}