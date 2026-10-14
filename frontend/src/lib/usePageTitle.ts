import { useEffect } from 'react'

export function usePageTitle(title: string) {
  useEffect(() => {
    document.title = title ? `${title} - ReviewRadar` : 'ReviewRadar - Amazon Reviews 2023'
  }, [title])
}
