import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from 'react'
import { askAssistant } from '@/api/endpoints'
import type { AssistantChatResponse } from '@/api/types'
import { useFilters } from './FilterContext'

export interface ChatTurn {
  id: number
  question: string
  /** Category the question was scoped to when it was sent. */
  category: string | null
  topK: number
  state: 'pending' | 'done' | 'error'
  response?: AssistantChatResponse
  error?: string
}

interface AssistantState {
  turns: ChatTurn[]
  isBusy: boolean
  /** Status of the most recent answer, so other screens can say whether generated answers are on. */
  lastStatus: AssistantChatResponse['status'] | null
  ask: (question: string, options?: { topK?: number }) => void
  retry: (id: number) => void
  clear: () => void
}

const AssistantContext = createContext<AssistantState | null>(null)

export const DEFAULT_TOP_K = 4

/** Chat history lives here so the side panel and the full Assistant page show the same conversation. */
export function AssistantProvider({ children }: { children: ReactNode }) {
  const { category } = useFilters()
  const [turns, setTurns] = useState<ChatTurn[]>([])
  const nextId = useRef(1)
  // A ref mirrors "a request is running" so a fast double submit cannot slip past a stale closure.
  const busy = useRef(false)

  const patch = useCallback((id: number, change: Partial<ChatTurn>) => {
    setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, ...change } : t)))
  }, [])

  const run = useCallback(
    async (turn: ChatTurn) => {
      busy.current = true
      try {
        const response = await askAssistant({ question: turn.question, category: turn.category, top_k: turn.topK })
        patch(turn.id, { state: 'done', response, error: undefined })
      } catch (err) {
        patch(turn.id, { state: 'error', error: err instanceof Error ? err.message : 'Something went wrong.' })
      } finally {
        busy.current = false
      }
    },
    [patch],
  )

  const ask = useCallback(
    (question: string, options?: { topK?: number }) => {
      const trimmed = question.trim()
      if (trimmed.length < 2 || busy.current) return
      const turn: ChatTurn = {
        id: nextId.current++,
        question: trimmed,
        category,
        topK: options?.topK ?? DEFAULT_TOP_K,
        state: 'pending',
      }
      setTurns((prev) => [...prev, turn])
      void run(turn)
    },
    [category, run],
  )

  const retry = useCallback(
    (id: number) => {
      if (busy.current) return
      const turn = turns.find((t) => t.id === id)
      if (!turn) return
      patch(id, { state: 'pending', error: undefined })
      void run({ ...turn, state: 'pending' })
    },
    [patch, run, turns],
  )

  const clear = useCallback(() => {
    if (!busy.current) setTurns([])
  }, [])

  const isBusy = turns.some((t) => t.state === 'pending')
  const lastStatus = useMemo(() => [...turns].reverse().find((t) => t.response)?.response?.status ?? null, [turns])
  const value = useMemo(() => ({ turns, isBusy, lastStatus, ask, retry, clear }), [turns, isBusy, lastStatus, ask, retry, clear])

  return <AssistantContext.Provider value={value}>{children}</AssistantContext.Provider>
}

export function useAssistant(): AssistantState {
  const ctx = useContext(AssistantContext)
  if (!ctx) throw new Error('useAssistant must be used inside <AssistantProvider>')
  return ctx
}
