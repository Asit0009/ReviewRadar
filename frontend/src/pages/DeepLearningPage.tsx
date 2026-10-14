import { useMutation } from '@tanstack/react-query'
import { Loader2, Search } from 'lucide-react'
import { useId, useState, type FormEvent } from 'react'
import { Link } from 'react-router-dom'
import { askAssistant } from '@/api/endpoints'
import { useHealth, useModelMetrics } from '@/api/queries'
import type { DistilbertMetrics, ModelMetricsResponse } from '@/api/types'
import { CategorySelect } from '@/components/CategorySelect'
import { EvidenceList } from '@/components/assistant/Evidence'
import { PageLayout } from '@/components/layout/PageLayout'
import { Card } from '@/components/ui/Card'
import { Callout, PageHeader, Select } from '@/components/ui/controls'
import { ErrorBlock, NotAvailableBlock, QueryBoundary } from '@/components/ui/StateBlocks'
import { useFilters } from '@/context/FilterContext'
import { fmtInt, fmtRatio, fmtScore } from '@/lib/format'
import { usePageTitle } from '@/lib/usePageTitle'

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-line-soft bg-bg/40 px-3 py-2">
      <dt className="text-[11px] text-muted">{label}</dt>
      <dd className="mt-0.5 text-[15px] font-semibold break-words text-ink tabular-nums">{value}</dd>
    </div>
  )
}

const orNA = <T,>(v: T | undefined | null, format: (n: T) => string) => (v === undefined || v === null ? 'n/a' : format(v))

function DistilbertDetails({ d, lrMacroF1 }: { d: DistilbertMetrics; lrMacroF1: number | undefined }) {
  const t = d.training
  const test = d.test
  const subset = t?.train_samples !== undefined && t.train_split_size !== undefined && t.train_samples < t.train_split_size
  return (
    <div className="space-y-4">
      <div>
        <h3 className="mb-2 text-xs font-medium text-muted">Results on {orNA(d.evaluated_on?.test_samples, fmtInt)} held-out test reviews (the same reviews the other models were scored on)</h3>
        <dl className="grid grid-cols-2 gap-2 @xl:grid-cols-4">
          <Fact label="Accuracy" value={orNA(test?.accuracy, fmtRatio)} />
          <Fact label="Macro F1" value={orNA(test?.macro_f1, (n) => fmtScore(n, 3))} />
          <Fact label="Negative F1" value={orNA(test?.negative_f1, (n) => fmtScore(n, 3))} />
          <Fact label="Neutral F1" value={orNA(test?.neutral_f1, (n) => fmtScore(n, 3))} />
        </dl>
        {lrMacroF1 !== undefined && test?.macro_f1 !== undefined && (
          <p className="mt-2 text-xs text-muted">
            For comparison, TF-IDF + Logistic Regression scored a macro F1 of {fmtScore(lrMacroF1, 3)} on the same reviews.
          </p>
        )}
      </div>
      <div>
        <h3 className="mb-2 text-xs font-medium text-muted">How it was trained</h3>
        <dl className="grid grid-cols-2 gap-2 @xl:grid-cols-4">
          <Fact label="Base model" value={d.model_name ?? t?.base_model ?? 'n/a'} />
          <Fact label="Training reviews" value={t?.train_samples !== undefined ? `${fmtInt(t.train_samples)} of ${orNA(t.train_split_size, fmtInt)}` : 'n/a'} />
          <Fact label="Epochs" value={orNA(t?.epochs, String)} />
          <Fact label="Device" value={t?.device ?? 'n/a'} />
          <Fact label="Batch (x accumulation)" value={t?.batch_size !== undefined ? `${t.batch_size} x ${t.gradient_accumulation_steps ?? 1}` : 'n/a'} />
          <Fact label="Max review length" value={t?.max_length !== undefined ? `${t.max_length} tokens` : 'n/a'} />
          <Fact label="Learning rate" value={orNA(t?.learning_rate, (n) => n.toExponential(0))} />
          <Fact label="Training time" value={orNA(t?.training_seconds, (n) => `${Math.round(n / 60)} min`)} />
        </dl>
        {t?.loss_per_epoch && t.loss_per_epoch.length > 0 && <p className="mt-2 text-xs text-muted">Mean training loss per epoch: {t.loss_per_epoch.map((l) => l.toFixed(3)).join(' → ')}.</p>}
      </div>
      {subset && (
        <Callout tone="warn">
          DistilBERT was fine-tuned on a subset of the training split ({fmtInt(t.train_samples!)} of {fmtInt(t.train_split_size!)} reviews){t?.frozen_word_embeddings ? ' with the word-embedding table frozen' : ''} so it could train on a CPU with limited memory.
          Training on the full split, ideally on a GPU, would probably score higher.
        </Callout>
      )}
    </div>
  )
}

function DistilbertCard({ data }: { data: ModelMetricsResponse }) {
  const d = data.metrics.models?.distilbert
  if (!d) {
    return (
      <NotAvailableBlock
        title="DistilBERT has not been trained yet"
        hint="Run python -m reviewpulse.modeling.train_distilbert on the machine that hosts the backend. See the README for the CPU-sized and full-data options."
      />
    )
  }
  return <DistilbertDetails d={d} lrMacroF1={data.metrics.models?.logistic_regression?.test?.macro_f1} />
}

function SemanticSearch() {
  const { category } = useFilters()
  const [text, setText] = useState('')
  const [topK, setTopK] = useState(5)
  const [error, setError] = useState<string | null>(null)
  const inputId = useId()
  const search = useMutation({
    mutationFn: (question: string) => askAssistant({ question, category, top_k: topK }),
  })

  const submit = (e: FormEvent) => {
    e.preventDefault()
    if (text.trim().length < 2) {
      setError('Enter at least 2 characters.')
      return
    }
    setError(null)
    search.mutate(text.trim())
  }

  return (
    <div className="space-y-4">
      <form onSubmit={submit} noValidate className="grid gap-3 @xl:grid-cols-[1fr_140px_auto] @xl:items-end">
        <div>
          <label htmlFor={inputId} className="mb-1 block text-xs font-medium text-muted">
            Describe what you are looking for
          </label>
          <input
            id={inputId}
            value={text}
            onChange={(e) => {
              setText(e.target.value)
              setError(null)
            }}
            aria-invalid={error ? true : undefined}
            placeholder="e.g. stopped working after a few weeks"
            className={`h-9 w-full rounded-lg border bg-card-2 px-3 text-[13px] text-ink placeholder:text-muted hover:border-brand-2/60 ${error ? 'border-neg' : 'border-line'}`}
          />
        </div>
        <Select label="Results" value={topK} onChange={setTopK} options={[3, 5, 8, 10].map((n) => ({ value: n, label: `${n} reviews` }))} />
        <button type="submit" disabled={search.isPending} className="inline-flex h-9 items-center justify-center gap-2 rounded-lg bg-brand px-4 text-[13px] font-medium text-white hover:bg-brand-2 disabled:opacity-60">
          {search.isPending ? <Loader2 size={15} className="animate-spin" aria-hidden="true" /> : <Search size={15} aria-hidden="true" />}
          Search
        </button>
      </form>
      {error && (
        <p role="alert" className="-mt-2 text-xs text-neg">
          {error}
        </p>
      )}

      {search.isPending && (
        <p role="status" className="flex items-center gap-2 text-[13px] text-muted">
          <Loader2 size={14} className="animate-spin" aria-hidden="true" /> Searching…
        </p>
      )}
      {search.isError && <ErrorBlock error={search.error} onRetry={() => search.mutate(text.trim())} />}
      {search.data && (
        <div className="space-y-2">
          <p className="text-xs text-muted">
            {search.data.evidence_count} {search.data.evidence_count === 1 ? 'review' : 'reviews'} found
            {search.data.evidence_count > 0 ? ', most similar first' : ` with similarity of at least ${search.data.retrieval.min_score}. Try different words or another category.`}. Searched {fmtInt(search.data.retrieval.indexed_reviews)} indexed reviews.
          </p>
          <EvidenceList reviews={search.data.evidence_reviews} />
        </div>
      )}
    </div>
  )
}

function IndexCoverage() {
  const { data } = useHealth()
  const r = data?.health.retrieval
  if (!r) return null
  const full = r.coverage_pct >= 100 && !r.limited_by
  return (
    <Callout tone={full ? 'info' : 'warn'} className="mb-4">
      {r.ready ? (
        <>
          The search index covers <strong className="text-ink">{fmtInt(r.indexed_reviews)}</strong> of {fmtInt(r.eligible_reviews)} reviews (<strong className="text-ink">{r.coverage_pct}%</strong>), embedded with {r.embedding_model ?? 'an unknown model'}.
          {full ? ' Every review with text is searchable.' : ` It is smaller than the dataset because of: ${r.limited_by ?? 'a configured limit'}.`}
        </>
      ) : (
        <>The search index is not available{r.limited_by === 'stale_index' ? ' because it was built from an older version of the data. Rebuild it with python -m reviewpulse.nlp.semantic_search' : '. Build it with python -m reviewpulse.nlp.semantic_search'}.</>
      )}
    </Callout>
  )
}

export default function DeepLearningPage() {
  usePageTitle('Deep Learning (NLP)')
  const metrics = useModelMetrics()
  return (
    <PageLayout>
      <PageHeader title="Deep Learning (NLP)" description="Neural models in the project: a fine-tuned DistilBERT sentiment classifier, and sentence embeddings that let you search reviews by meaning." />

      <Card
        title="DistilBERT sentiment model"
        subtitle="Numbers come from the saved training run"
        actions={
          <Link to="/predictions" className="text-xs font-medium text-brand-2 hover:underline">
            Try it on ML Predictions
          </Link>
        }
      >
        <QueryBoundary query={metrics} loadingLabel="Loading DistilBERT results" loadingRows={4}>
          {(data) => <DistilbertCard data={data} />}
        </QueryBoundary>
      </Card>

      <Card
        title="Search reviews by meaning"
        subtitle="Sentence embeddings and a FAISS index find reviews close in meaning, even without the same words."
        actions={<CategorySelect hideLabel className="w-48" />}
      >
        <IndexCoverage />
        <SemanticSearch />
      </Card>
    </PageLayout>
  )
}
