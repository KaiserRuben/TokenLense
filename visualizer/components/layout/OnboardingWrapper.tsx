"use client"

import { usePathname } from "next/navigation"
import Onboarding from "./Onboarding"

export default function OnboardingWrapper() {
  const pathname = usePathname()
  return <Onboarding path={pathname} />
}