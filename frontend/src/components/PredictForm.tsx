import { Frown, Loader2, Meh, Smile, Wand2 } from 'lucide-react'
import { useId, useState, type FormEvent } from 'react'
import type { PredictModel, PredictResponse } from '@/api/types'
import { useHealth, usePredict } from '@/api/queries'
import { fmtRatio } from '@/lib/format'
import { SENTIMENT_META, SENTIMENTS, isSentiment } from '@/lib/theme'
import { Callout, ProgressBar } from './ui/controls'
import { ErrorBlock } from './ui/StateBlocks'

const MIN_LENGTH = 2 // matches PredictRequest.text min_length in the API schema
const MAX_LENGTH = 2000

const EXAMPLES = [
  'The blender stopped working after three uses and customer support never answered.',
  'Fits perfectly and works exactly as described. Great value for the price.',
  'It is okay. Does the job, but nothing special.',
]

const MODELS: Array<{ value: PredictModel; label: string; hint: string }> = [
  { value: 'logistic_regression', label: 'Logistic Regression', hint: 'TF-IDF features, shows probabilities' },
  { value: 'linear_svc', label: 'Linear SVC', hint: 'TF-IDF features, label only' },
  { value: 'distilbert', label: 'DistilBERT', hint: 'Fine-tuned transformer. Reads word order and context; slower than the other two' },
]

const ICONS = { positive: Smile, neutral: Meh, negative: Frown } as const

function PredictionResult({ result }: { result: PredictResponse }) {
  const sentiment = isSentiment(result.predicted_sentiment) ? result.predicted_sentiment : null
  const meta = sentiment ? SENTIMENT_META[sentiment] : null
  const Icon = sentiment ? ICONS[sentiment] : Meh
  return (
    <div className="space-y-3" aria-live="polite">
      <div className="flex items-center gap-3">
        <span className="flex size-11 items-center justify-center rounded-xl" style={{ background: `${meta?.color ?? '#8b98c2'}2e`, color: meta?.color ?? '#8b98c2' }} aria-hidden="true">
          <Icon size={24} />
        </span>
        <div>
          <p className="text-xs text-muted">Predicted sentiment</p>
          <p className="text-xl font-semibold" style={{ color: meta?.color }}>
            {meta?.label ?? result.predicted_sentiment}
          </p>
        </div>
        <p className="ml-auto text-right text-xs text-muted">
          Model
          <span className="block text-[13px] font-medium text-ink-2">{result.model_used}</span>
        </p>
      </div>

      {result.probabilities ? (
        <ul className="space-y-2">
          {SENTIMENTS.map((s) => {
            const p = result.probabilities?.[s] ?? 0
            return (
              <li key={s} className="grid grid-cols-[70px_1fr_52px] items-center gap-3 text-[13px]">
                <span className="text-ink-2">{SENTIMENT_META[s].label}</span>
                <ProgressBar value={p * 100} color={SENTIMENT_META[s].color} label={`${SENTIMENT_META[s].label} probability`} />
                <span className="text-right font-medium text-ink tabular-nums">{fmtRatio(p)}</span>
              </li>
            )
          })}
        </ul>
      ) : (
        <p className="text-xs text-muted">This model only returns a label. Choose Logistic Regression to see probabilities.</p>
      )}

      {result.is_rating_derived_label && (
        <Callout>The model was trained on labels taken from star ratings, so it estimates what rating-based label a review would get. It is not a human reading of the text.</Callout>
      )}
    </div>
  )
}

/** Sends the text to POST /predict and shows the label and class probabilities that come back. */
export function PredictForm() {
  const [text, setText] = useState('')
  const [model, setModel] = useState<PredictModel>('logistic_regression')
  const [error, setError] = useState<string | null>(null)
  const predict = usePredict()
  const health = useHealth()
  // DistilBERT only works once it has been trained; the backend says whether the saved model exists.
  const distilbertReady = health.data?.health?.artifacts_ready?.distilbert_model === true
  const textId = useId()
  const errorId = useId()

  const submit = (e: FormEvent) => {
    e.preventDefault()
    const trimmed = text.trim()
    if (trimmed.length < MIN_LENGTH) {
      setError(`Enter at least ${MIN_LENGTH} characters of review text.`)
      return
    }
    setError(null)
    predict.mutate({ text: trimmed, model_type: model })
  }

  return (
    <div className="grid gap-5 @3xl:grid-cols-2">
      <form onSubmit={submit} noValidate className="space-y-3">
        <div>
          <label htmlFor={textId} className="mb-1 block text-xs font-medium text-muted">
            Review text
          </label>
          <textarea
            id={textId}
            value={text}
            onChange={(e) => {
              setText(e.target.value)
              if (error) setError(null)
            }}
            rows={5}
            maxLength={MAX_LENGTH}
            aria-invalid={error ? true : undefined}
            aria-describedby={error ? errorId : undefined}
            placeholder="Paste or type a review…"
            className={`w-full resize-y rounded-lg border bg-card-2 px-3 py-2 text-[13px] leading-relaxed text-ink placeholder:text-muted hover:border-brand-2/60 ${error ? 'border-neg' : 'border-line'}`}
          />
          <div className="mt-1 flex items-start justify-between gap-3 text-[11px]">
            {error ? (
              <p id={errorId} role="alert" className="text-neg">
                {error}
              </p>
            ) : (
              <span />
            )}
            <span className="shrink-0 text-muted tabular-nums">
              {text.length} / {MAX_LENGTH}
            </span>
          </div>
        </div>

        <fieldset>
          <legend className="mb-1.5 text-xs font-medium text-muted">Model</legend>
          <div className="grid grid-cols-[repeat(auto-fit,minmax(170px,1fr))] gap-2">
            {MODELS.map((m) => {
              const unavailable = m.value === 'distilbert' && !distilbertReady
              return (
                <label
                  key={m.value}
                  className={`flex items-start gap-2.5 rounded-lg border px-3 py-2 text-[13px] has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-brand-2 ${unavailable ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'} ${model === m.value ? 'border-brand/70 bg-brand/15' : 'border-line bg-card-2/50 hover:border-brand-2/50'}`}
                >
                  <input type="radio" name="model" value={m.value} checked={model === m.value} disabled={unavailable} onChange={() => setModel(m.value)} className="mt-1 accent-[var(--color-brand-2)]" />
                  <span>
                    <span className="block font-medium text-ink">{m.label}</span>
                    <span className="block text-[11px] text-muted">{unavailable ? 'Not trained yet' : m.hint}</span>
                  </span>
                </label>
              )
            })}
          </div>
        </fieldset>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="submit"
            disabled={predict.isPending}
            className="inline-flex h-9 items-center gap-2 rounded-lg bg-brand px-4 text-[13px] font-medium text-white hover:bg-brand-2 disabled:opacity-60"
          >
            {predict.isPending ? <Loader2 size={15} className="animate-spin" aria-hidden="true" /> : <Wand2 size={15} aria-hidden="true" />}
            Predict sentiment
          </button>
          <span className="text-xs text-muted">or try an example:</span>
          {EXAMPLES.map((ex, i) => (
            <button
              key={ex}
              type="button"
              onClick={() => {
                setText(ex)
                setError(null)
              }}
              className="rounded-full border border-line bg-card-2/60 px-2.5 py-1 text-xs text-ink-2 hover:border-brand-2/60 hover:text-ink"
            >
              Example {i + 1}
            </button>
          ))}
        </div>
      </form>

      <div className="min-h-[180px] rounded-xl border border-line-soft bg-bg/40 p-4">
        {predict.isPending ? (
          <p role="status" className="flex items-center gap-2 text-[13px] text-muted">
            <Loader2 size={15} className="animate-spin" aria-hidden="true" /> Predicting…
          </p>
        ) : predict.isError ? (
          <ErrorBlock error={predict.error} onRetry={() => predict.mutate({ text: text.trim(), model_type: model })} />
        ) : predict.data && predict.variables && predict.data.model_type !== predict.variables.model_type ? (
          // Never show one model's answer as if it came from another: say so instead.
          <Callout tone="warn">
            You asked for {predict.variables.model_type}, but the server answered with {predict.data.model_type}. The result is not shown because it is not from the model you chose.
          </Callout>
        ) : predict.data ? (
          <PredictionResult result={predict.data} />
        ) : (
          <p className="text-[13px] leading-relaxed text-muted">The result will appear here. Type a review, pick a model and press "Predict sentiment".</p>
        )}
      </div>
    </div>
  )
}
