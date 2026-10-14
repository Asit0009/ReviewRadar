import { PageLayout } from '@/components/layout/PageLayout'
import { AssistantChat } from '@/components/assistant/AssistantChat'
import { CategorySelect } from '@/components/CategorySelect'
import { Callout, PageHeader } from '@/components/ui/controls'
import { useHealth } from '@/api/queries'
import { useAssistant } from '@/context/AssistantContext'
import { fmtInt } from '@/lib/format'
import { usePageTitle } from '@/lib/usePageTitle'

const STEPS = [
  ['Count', 'The backend counts reviews, ratings and sentiment for your category with SQL.'],
  ['Find', 'It looks up the reviews closest in meaning to your question in a search index.'],
  ['Answer', 'If a Gemini API key is set, the language model writes an answer from those two sources.'],
] as const

export default function AssistantPage() {
  usePageTitle('LLM Assistant')
  const { lastStatus } = useAssistant()
  const health = useHealth().data?.health
  return (
    <PageLayout showAssistant={false}>
      <PageHeader
        title="LLM Assistant"
        description="Ask a question about the reviews. Every answer comes with the database numbers and the reviews it was based on."
        actions={<CategorySelect className="w-56" />}
      />

      <div className="grid gap-4 @3xl:grid-cols-[minmax(0,1fr)_300px]">
        <section aria-label="Assistant chat" className="flex h-[calc(100dvh-var(--topbar-h)-170px)] min-h-[520px] min-w-0 flex-col overflow-hidden rounded-xl border border-line bg-card/90">
          <AssistantChat variant="page" />
        </section>

        <div className="space-y-4">
          <section aria-labelledby="how-it-works" className="rounded-xl border border-line bg-card/90 p-4">
            <h2 id="how-it-works" className="text-[15px] font-semibold text-ink">
              How answers are built
            </h2>
            <ol className="mt-3 space-y-3">
              {STEPS.map(([title, text], i) => (
                <li key={title} className="flex gap-3 text-[13px]">
                  <span aria-hidden="true" className="flex size-6 shrink-0 items-center justify-center rounded-full bg-brand/25 text-xs font-semibold text-brand-2">
                    {i + 1}
                  </span>
                  <span className="leading-relaxed text-ink-2">
                    <strong className="font-semibold text-ink">{title}.</strong> {text}
                  </span>
                </li>
              ))}
            </ol>
          </section>
          {lastStatus === 'success' ? (
            <Callout>Generated answers are on. The language model only sees the numbers and reviews listed under each answer.</Callout>
          ) : lastStatus ? (
            <Callout tone="warn">The last answer was not written by Gemini (reason: {lastStatus.replace(/_/g, ' ')}), so it shows the database numbers and similar reviews only.</Callout>
          ) : health && !health.assistant.llm_configured ? (
            <Callout tone="warn">No Gemini API key is configured on the server. Questions still return database numbers and the most similar reviews, but no written answer. Set GEMINI_API_KEY and restart the backend to turn written answers on.</Callout>
          ) : health ? (
            <Callout>Gemini is configured ({health.assistant.llm_model}). Answers are written by the model from the numbers and reviews listed under each answer.</Callout>
          ) : null}
          {health && (
            <Callout tone={health.retrieval.ready && health.retrieval.coverage_pct >= 100 ? 'info' : 'warn'}>
              {health.retrieval.ready
                ? `Review search covers ${fmtInt(health.retrieval.indexed_reviews)} of ${fmtInt(health.retrieval.eligible_reviews)} reviews (${health.retrieval.coverage_pct}%).`
                : 'The review search index is not available, so no reviews can be retrieved until it is rebuilt.'}
            </Callout>
          )}
        </div>
      </div>
    </PageLayout>
  )
}
