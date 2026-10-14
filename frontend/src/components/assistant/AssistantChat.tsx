import { Bot, Loader2, Send, Trash2, User } from 'lucide-react'
import { useEffect, useId, useMemo, useRef, useState, type FormEvent } from 'react'
import { DEFAULT_TOP_K, useAssistant, type ChatTurn } from '@/context/AssistantContext'
import { useFilters } from '@/context/FilterContext'
import { categoryLabel, fmtInt } from '@/lib/format'
import { Callout, Chip, Select } from '../ui/controls'
import { RetryButton } from '../ui/StateBlocks'
import { AnswerBody } from './AnswerBody'
import { EvidenceList, FactChips } from './Evidence'

/** Starter questions. Clicking one sends a real request to /assistant/chat. */
export const SUGGESTED_QUESTIONS = [
  'What are the biggest complaints customers have?',
  'What do customers like most about these products?',
  'Do reviewers mention quality or durability problems?',
  'How do customers feel about price and value?',
] as const

/** Why there is no written answer, in a few words. The full explanation comes from the backend in `response`. */
const STATUS_TITLES: Record<string, string> = {
  setup_required: 'No written answer: no Gemini API key is configured',
  invalid_key: 'No written answer: Gemini rejected the API key',
  rate_limited: 'No written answer: Gemini rate limit reached',
  timeout: 'No written answer: Gemini timed out',
  network_error: 'No written answer: Gemini could not be reached',
  service_error: 'No written answer: Gemini had a server error',
  blocked: 'No written answer: Gemini declined to answer',
  api_error: 'No written answer: the Gemini request failed',
}

function UserBubble({ turn }: { turn: ChatTurn }) {
  return (
    <div className="flex items-start justify-end gap-2">
      <div className="max-w-[88%] rounded-2xl rounded-tr-sm bg-brand/80 px-3 py-2 text-[13px] leading-snug text-white">
        {turn.question}
        <p className="mt-1 text-[10px] text-white/70">{turn.category ? categoryLabel(turn.category) : 'All categories'}</p>
      </div>
      <span aria-hidden="true" className="flex size-7 shrink-0 items-center justify-center rounded-full bg-violet/30 text-violet">
        <User size={14} />
      </span>
    </div>
  )
}

function AssistantBubble({ turn, onRetry, evidenceOpen }: { turn: ChatTurn; onRetry: () => void; evidenceOpen: boolean }) {
  const res = turn.response
  const evidenceRef = useRef<HTMLDetailsElement>(null)
  const evidenceIds = useMemo(() => new Set((res?.evidence_reviews ?? []).map((r) => r.review_id)), [res])

  // A citation in the answer opens the evidence list, scrolls to that review and highlights it for a moment.
  const showCitedReview = (reviewId: string) => {
    const details = evidenceRef.current
    if (!details) return
    details.open = true
    const card = Array.from(details.querySelectorAll<HTMLElement>('[data-review-id]')).find((el) => el.dataset.reviewId === reviewId)
    if (!card) return
    card.scrollIntoView?.({ block: 'nearest', behavior: 'smooth' })
    card.focus({ preventScroll: true })
    card.dataset.cited = 'true'
    window.setTimeout(() => delete card.dataset.cited, 2000)
  }

  return (
    <div className="flex items-start gap-2">
      <span aria-hidden="true" className="flex size-7 shrink-0 items-center justify-center rounded-full bg-teal/20 text-teal">
        <Bot size={14} />
      </span>
      <div className="min-w-0 flex-1 space-y-2.5 rounded-2xl rounded-tl-sm border border-line-soft bg-card-2/50 p-3">
        {turn.state === 'pending' && (
          <p role="status" className="flex items-center gap-2 text-[13px] text-muted">
            <Loader2 size={14} className="animate-spin" aria-hidden="true" /> Searching {turn.category ? categoryLabel(turn.category) : 'all'} reviews…
          </p>
        )}

        {turn.state === 'error' && (
          <div role="alert" className="space-y-2">
            <p className="text-[13px] text-neg">{turn.error}</p>
            <RetryButton onRetry={onRetry} />
          </div>
        )}

        {turn.state === 'done' && res && (
          <>
            {res.llm_used ? (
              <>
                <AnswerBody text={res.response} evidenceIds={evidenceIds} onCite={showCitedReview} />
                <p className="text-[11px] text-muted">
                  Written by <span className="font-medium text-ink-2">{res.llm_model ?? 'Gemini'}</span> from the numbers and reviews below.
                </p>
              </>
            ) : (
              <Callout tone="warn">
                <p className="mb-1 font-medium text-ink">{STATUS_TITLES[res.status] ?? 'No written answer'}</p>
                <AnswerBody text={res.response} className="text-xs" />
                {res.error_detail && (
                  <details className="mt-1.5 text-[11px] text-muted">
                    <summary className="cursor-pointer select-none hover:text-ink">Technical detail</summary>
                    <p className="mt-1 break-words">{res.error_detail}</p>
                  </details>
                )}
              </Callout>
            )}
            <div>
              <p className="mb-1.5 text-[11px] font-medium text-muted">Database numbers for {res.numerical_facts.scope ? categoryLabel(res.numerical_facts.scope) : 'this scope'}</p>
              <FactChips facts={res.numerical_facts} />
            </div>
            <p className="text-[11px] leading-snug text-muted">
              Searched {fmtInt(res.retrieval.indexed_reviews)} of {fmtInt(res.retrieval.eligible_reviews)} reviews ({res.retrieval.coverage_pct}%).
              {res.evidence_count === 0 ? ` No review was similar enough (minimum similarity ${res.retrieval.min_score}).` : ` Reviews below ${res.retrieval.min_score} similarity are left out.`}
            </p>
            <details ref={evidenceRef} open={evidenceOpen} className="group rounded-lg">
              <summary className="cursor-pointer text-xs font-medium text-brand-2 select-none hover:underline">
                Evidence from {res.evidence_count} {res.evidence_count === 1 ? 'review' : 'reviews'}
              </summary>
              <div className="mt-2">
                <EvidenceList reviews={res.evidence_reviews} />
              </div>
            </details>
          </>
        )}
      </div>
    </div>
  )
}

interface AssistantChatProps {
  variant: 'panel' | 'page'
}

export function AssistantChat({ variant }: AssistantChatProps) {
  const { turns, isBusy, ask, retry, clear } = useAssistant()
  const { category } = useFilters()
  const [draft, setDraft] = useState('')
  const [topK, setTopK] = useState(DEFAULT_TOP_K)
  const threadRef = useRef<HTMLDivElement>(null)
  const inputId = useId()
  const canSend = draft.trim().length >= 2 && !isBusy

  // Keep the newest message in view. While a question is pending the bottom is shown (question and "Searching");
  // when the answer arrives its top is shown, so the written answer is read before the evidence list below it.
  useEffect(() => {
    const thread = threadRef.current
    const last = turns[turns.length - 1]
    if (!thread || !last) return
    const turnEl = thread.querySelector<HTMLElement>(`[data-turn-id="${last.id}"]`)
    if (last.state === 'pending' || !turnEl) {
      thread.scrollTop = thread.scrollHeight
    } else {
      thread.scrollTop += turnEl.getBoundingClientRect().top - thread.getBoundingClientRect().top - 8
    }
  }, [turns])

  const submit = (e: FormEvent) => {
    e.preventDefault()
    if (!canSend) return
    ask(draft, { topK })
    setDraft('')
  }

  const sendSuggestion = (q: string) => {
    if (!isBusy) ask(q, { topK })
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div ref={threadRef} className="min-h-0 flex-1 space-y-3 overflow-y-auto px-4 py-3" aria-live="polite" aria-label="Conversation">
        {turns.length === 0 ? (
          <>
            <p className="text-[13px] leading-relaxed text-muted">
              Ask about the reviews in plain English. Each answer shows the database numbers and the most similar reviews it found.
              <span className="mt-1 block">
                Searching: <Chip tone="blue">{category ? categoryLabel(category) : 'All categories'}</Chip>
              </span>
            </p>
            <ul className="space-y-2" aria-label="Suggested questions">
              {SUGGESTED_QUESTIONS.map((q) => (
                <li key={q}>
                  <button
                    type="button"
                    disabled={isBusy}
                    onClick={() => sendSuggestion(q)}
                    className="flex w-full items-center gap-2.5 rounded-lg border border-line-soft bg-card-2/60 px-3 py-2.5 text-left text-[13px] leading-snug text-ink-2 transition-colors hover:border-brand-2/60 hover:text-ink disabled:opacity-50"
                  >
                    <span aria-hidden="true" className="size-4 shrink-0 rounded-full border-2 border-brand-2" />
                    {q}
                  </button>
                </li>
              ))}
            </ul>
          </>
        ) : (
          turns.map((turn) => (
            <div key={turn.id} data-turn-id={turn.id} className="space-y-3">
              <UserBubble turn={turn} />
              <AssistantBubble turn={turn} onRetry={() => retry(turn.id)} evidenceOpen={variant === 'page'} />
            </div>
          ))
        )}
      </div>

      <div className="border-t border-line-soft px-4 pt-3 pb-4">
        {turns.length > 0 && (
          <div className="mb-2 flex items-center justify-between gap-2">
            <details className="min-w-0 text-xs text-muted">
              <summary className="cursor-pointer select-none hover:text-ink">Suggested questions</summary>
              <ul className="mt-1.5 space-y-1">
                {SUGGESTED_QUESTIONS.map((q) => (
                  <li key={q}>
                    <button type="button" disabled={isBusy} onClick={() => sendSuggestion(q)} className="text-left text-ink-2 hover:text-ink hover:underline disabled:opacity-50">
                      {q}
                    </button>
                  </li>
                ))}
              </ul>
            </details>
            <button
              type="button"
              onClick={clear}
              disabled={isBusy}
              className="inline-flex shrink-0 items-center gap-1 text-xs text-muted hover:text-ink disabled:opacity-40"
            >
              <Trash2 size={12} aria-hidden="true" /> Clear chat
            </button>
          </div>
        )}
        <form onSubmit={submit} className="flex items-end gap-2">
          {variant === 'page' && (
            <Select
              label="Reviews to retrieve"
              value={topK}
              onChange={setTopK}
              options={[3, 4, 6, 8, 10].map((n) => ({ value: n, label: `${n} reviews` }))}
              className="w-28 shrink-0"
            />
          )}
          <div className="min-w-0 flex-1">
            <label htmlFor={inputId} className="sr-only">
              Ask a question about the reviews
            </label>
            <input
              id={inputId}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder={turns.length ? 'Ask another question…' : 'Ask about the reviews…'}
              autoComplete="off"
              maxLength={500}
              className="h-9 w-full rounded-lg border border-line bg-card-2 px-3 text-[13px] text-ink placeholder:text-muted hover:border-brand-2/60"
            />
          </div>
          <button
            type="submit"
            disabled={!canSend}
            aria-label="Send question"
            className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-brand text-white transition-colors hover:bg-brand-2 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {isBusy ? <Loader2 size={16} className="animate-spin" aria-hidden="true" /> : <Send size={16} aria-hidden="true" />}
          </button>
        </form>
        <p className="mt-2 text-[11px] leading-snug text-muted">Answers use real review text and database counts from this dataset.</p>
      </div>
    </div>
  )
}
