import { useEffect } from 'react'

// Analytics is fully disabled for this build: no PostHog initialization,
// no network calls, regardless of any stored consent/settings value.
export function AnalyticProvider() {
  useEffect(() => {
    console.log('Analytics disabled')
  }, [])

  // This component doesn't render anything
  return null
}