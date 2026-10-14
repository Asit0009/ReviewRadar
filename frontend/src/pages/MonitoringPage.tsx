import { CheckCircle2, RefreshCw, XCircle } from 'lucide-react'
import { API_BASE_URL } from '@/api/client'
import { useHealth, useMonitoring } from '@/api/queries'
import { DriftChart, PSI_ALERT, PSI_WATCH } from '@/components/charts/DriftChart'
import { PageLayout } from '@/components/layout/PageLayout'
import { Card } from '@/components/ui/Card'
import { Callout, Chip, PageHeader } from '@/components/ui/controls'
import { NotAvailableBlock, QueryBoundary } from '@/components/ui/StateBlocks'
import { fmtInt, fmtTime } from '@/lib/format'
import { usePageTitle } from '@/lib/usePageTitle'

const ARTIFACT_LABELS: Record<string, string> = {
  parquet_data: 'Review dataset (Parquet)',
  sentiment_models: 'Sentiment models',
  vector_search_faiss: 'Search index (FAISS)',
  nmf_topics: 'Topic model and review tags (NMF)',
  distilbert_model: 'DistilBERT model (optional)',
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3 border-t border-line-soft py-2 text-[13px] first:border-t-0">
      <dt className="text-muted">{label}</dt>
      <dd className="text-right font-medium text-ink">{children}</dd>
    </div>
  )
}

function BackendStatus() {
  const query = useHealth()
  return (
    <Card title="Backend status" subtitle="From GET /health">
      <QueryBoundary query={query} loadingLabel="Checking the backend" loadingRows={5}>
        {({ health, latencyMs, checkedAt }) => (
          <div className="space-y-4">
            <dl>
              <Row label="Status">
                <Chip tone={health.status === 'healthy' ? 'teal' : 'neutral'}>{health.status}</Chip>
              </Row>
              <Row label="API version">{health.version}</Row>
              <Row label="Response time (this browser)">{latencyMs} ms</Row>
              <Row label="Last checked">{fmtTime(new Date(checkedAt).toISOString())}</Row>
              <Row label="API address">
                <code className="font-mono text-xs text-ink-2">{API_BASE_URL}</code>
              </Row>
            </dl>
            <div>
              <h3 className="mb-2 text-xs font-medium text-muted">Loaded files</h3>
              <ul className="space-y-1.5">
                {Object.entries(health.artifacts_ready).map(([key, ready]) => (
                  <li key={key} className="text-[13px] text-ink-2">
                    <span className="flex items-center gap-2">
                      {ready ? <CheckCircle2 size={16} className="text-pos" aria-hidden="true" /> : <XCircle size={16} className="text-neg" aria-hidden="true" />}
                      {ARTIFACT_LABELS[key] ?? key.replace(/_/g, ' ')}
                      <span className="sr-only">{ready ? 'ready' : 'missing'}</span>
                    </span>
                    {!ready && health.setup_steps?.[key] && (
                      <code className="mt-1 ml-6 block rounded-md bg-bg/60 px-2 py-1 font-mono text-[11px] break-words text-muted">{health.setup_steps[key]}</code>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </QueryBoundary>
    </Card>
  )
}

function RetrievalCard() {
  const query = useHealth()
  return (
    <Card title="Search index and assistant" subtitle="Which reviews the assistant can search, and whether Gemini is set up">
      <QueryBoundary query={query} loadingLabel="Checking the search index" loadingRows={5}>
        {({ health }) => {
          const r = health.retrieval
          const full = r.ready && r.coverage_pct >= 100 && !r.limited_by
          return (
            <div className="space-y-3">
              <dl>
                <Row label="Reviews indexed">
                  {fmtInt(r.indexed_reviews)} of {fmtInt(r.eligible_reviews)}
                </Row>
                <Row label="Coverage">
                  <Chip tone={full ? 'teal' : 'neutral'}>{r.coverage_pct}%</Chip>
                </Row>
                <Row label="Embedding model">{r.embedding_model ?? 'n/a'}</Row>
                <Row label="Index built">{r.built_at ? r.built_at.replace('T', ' ').replace('+00:00', ' UTC') : 'n/a'}</Row>
                <Row label="Gemini API key">
                  <Chip tone={health.assistant.llm_configured ? 'teal' : 'neutral'}>{health.assistant.llm_configured ? `configured (${health.assistant.llm_model})` : 'not configured'}</Chip>
                </Row>
              </dl>
              {!r.ready && (
                <Callout tone="warn">
                  The search index is not usable{r.limited_by === 'stale_index' ? ' because it was built from an older version of the data' : ''}. Rebuild it with{' '}
                  <code className="font-mono">python -m reviewpulse.nlp.semantic_search</code>.
                </Callout>
              )}
              {r.ready && !full && <Callout tone="warn">The index is smaller than the dataset because of: {r.limited_by ?? 'a configured limit'}.</Callout>}
              {!health.assistant.llm_configured && (
                <Callout tone="warn">Without a Gemini API key the assistant returns database numbers and similar reviews only. Set GEMINI_API_KEY on the server and restart it.</Callout>
              )}
            </div>
          )
        }}
      </QueryBoundary>
    </Card>
  )
}

function DriftDetails() {
  const query = useMonitoring()
  return (
    <Card title="Data drift" subtitle="From GET /monitoring/status">
      <QueryBoundary query={query} loadingLabel="Running the drift check" loadingRows={6}>
        {(data) => (
          <div className="space-y-4">
            <div className="flex flex-wrap items-center gap-3">
              <Chip tone={data.status === 'healthy' ? 'teal' : 'neutral'}>{data.status === 'healthy' ? 'No drift found' : 'Drift detected'}</Chip>
              <p className="text-[13px] text-ink-2">
                {data.drifted_features_count} of {data.total_features_evaluated} features changed. Compared {fmtInt(data.reference_sample_size)} reference rows with {fmtInt(data.current_sample_size)} current rows.
              </p>
            </div>

            <DriftChart features={data.feature_metrics} height={170} />
            <p className="text-[11px] text-muted">
              <span className="text-neu">Dashed amber</span> line: PSI {PSI_WATCH} (moderate change). <span className="text-neg">Dashed red</span> line: PSI {PSI_ALERT} (large change).
            </p>

            <div className="overflow-x-auto">
              <table className="w-full min-w-[480px] border-separate border-spacing-0 text-[13px]">
                <caption className="sr-only">Drift results per feature</caption>
                <thead>
                  <tr className="text-left text-xs text-muted">
                    <th scope="col" className="px-3 py-2 font-medium">Feature</th>
                    <th scope="col" className="px-3 py-2 font-medium">Test</th>
                    <th scope="col" className="px-3 py-2 text-right font-medium">KS statistic</th>
                    <th scope="col" className="px-3 py-2 text-right font-medium">p-value</th>
                    <th scope="col" className="px-3 py-2 text-right font-medium">PSI</th>
                    <th scope="col" className="px-3 py-2 text-right font-medium">Result</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(data.feature_metrics).map(([name, f]) => (
                    <tr key={name}>
                      <th scope="row" className="border-t border-line-soft px-3 py-2 text-left font-medium text-ink">{name.replace(/_/g, ' ')}</th>
                      <td className="border-t border-line-soft px-3 py-2 text-ink-2">{f.metric}</td>
                      <td className="border-t border-line-soft px-3 py-2 text-right text-ink-2 tabular-nums">{f.ks_statistic?.toFixed(3) ?? <span className="text-muted" title="This test was not run for this feature">n/a</span>}</td>
                      <td className="border-t border-line-soft px-3 py-2 text-right text-ink-2 tabular-nums">{f.ks_pvalue?.toFixed(3) ?? <span className="text-muted" title="This test was not run for this feature">n/a</span>}</td>
                      <td className="border-t border-line-soft px-3 py-2 text-right text-ink-2 tabular-nums">{f.psi.toFixed(3)}</td>
                      <td className="border-t border-line-soft px-3 py-2 text-right">
                        <Chip tone={f.drift_detected ? 'neutral' : 'teal'}>{f.drift_detected ? 'Drift' : 'Stable'}</Chip>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <Callout>
              The oldest 70% of reviews{data.reference_period ? ` (${data.reference_period.from} to ${data.reference_period.to})` : ''} are the reference and the newest 30%
              {data.current_period ? ` (${data.current_period.from} to ${data.current_period.to})` : ''} are the current data. A feature is flagged when PSI is 0.2 or higher, or when the KS test is
              significant (p below 0.05) <em>and</em> the KS statistic is at least 0.1. Requiring a minimum size avoids false alarms, because with this many reviews even tiny differences are statistically significant.
            </Callout>
          </div>
        )}
      </QueryBoundary>
    </Card>
  )
}

export default function MonitoringPage() {
  usePageTitle('Deploy & Monitor')
  const health = useHealth()
  const monitoring = useMonitoring()
  const refreshing = health.isFetching || monitoring.isFetching

  return (
    <PageLayout>
      <PageHeader
        title="Deploy & Monitor"
        description="Check that the backend is running, which files it loaded, and whether the data has shifted."
        actions={
          <button
            type="button"
            onClick={() => {
              void health.refetch()
              void monitoring.refetch()
            }}
            disabled={refreshing}
            className="inline-flex h-9 items-center gap-2 rounded-lg border border-line bg-card-2 px-3 text-[13px] font-medium text-ink-2 hover:border-brand-2/60 hover:text-ink disabled:opacity-60"
          >
            <RefreshCw size={14} className={refreshing ? 'animate-spin' : ''} aria-hidden="true" /> Refresh
          </button>
        }
      />
      <div className="grid gap-4 @3xl:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
        <div className="space-y-4">
          <BackendStatus />
          <RetrievalCard />
        </div>
        <DriftDetails />
      </div>
      <Card title="Not available yet" subtitle="The backend has no endpoints for these, so they are not shown with made-up values">
        <div className="grid gap-2 @xl:grid-cols-2">
          {[
            ['Model accuracy over time', 'Needs an evaluation history, but only the latest training run is saved.'],
            ['Server-side request latency', 'Only the browser-measured round trip above is available.'],
          ].map(([title, hint]) => (
            <div key={title} className="rounded-xl border border-line-soft bg-bg/40">
              <NotAvailableBlock title={title} hint={hint} />
            </div>
          ))}
        </div>
      </Card>
    </PageLayout>
  )
}
