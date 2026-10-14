import { createContext, useContext, useMemo, useState, type ReactNode } from 'react'

interface UiState {
  /** Mobile navigation drawer. */
  navOpen: boolean
  setNavOpen: (open: boolean) => void
  /** On screens narrower than 1280px the assistant is a slide-over drawer. */
  assistantOpen: boolean
  setAssistantOpen: (open: boolean) => void
  /** On wide screens the assistant can be folded away to give charts more room. */
  assistantCollapsed: boolean
  setAssistantCollapsed: (collapsed: boolean) => void
}

const UiContext = createContext<UiState | null>(null)

export function UiProvider({ children }: { children: ReactNode }) {
  const [navOpen, setNavOpen] = useState(false)
  const [assistantOpen, setAssistantOpen] = useState(false)
  const [assistantCollapsed, setAssistantCollapsed] = useState(false)
  const value = useMemo(
    () => ({ navOpen, setNavOpen, assistantOpen, setAssistantOpen, assistantCollapsed, setAssistantCollapsed }),
    [navOpen, assistantOpen, assistantCollapsed],
  )
  return <UiContext.Provider value={value}>{children}</UiContext.Provider>
}

export function useUi(): UiState {
  const ctx = useContext(UiContext)
  if (!ctx) throw new Error('useUi must be used inside <UiProvider>')
  return ctx
}
