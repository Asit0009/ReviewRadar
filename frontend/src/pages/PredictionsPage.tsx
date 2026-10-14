import { useModelMetrics } from '@/api/queries'
import type { ModelMetricsResponse } from '@/api/types'
import { CategoryMetricsTable } from '@/components/charts/CategoryMetricsTable'
import { ConfusionMatrix } from '@/components/charts/ConfusionMatrix'
import { ModelTable } from '@/components/charts/ModelTable'
import { PageLayout } from '@/components/layout/PageLayout'
import { PredictForm } from '@/components/PredictForm'
import { Card } from '@/components/ui/Card'
import { Callout, PageHeader } from '@/components/ui/controls'
import { QueryBoundary } from '@/components/ui/StateBlocks'
import { buildModelRows } from '@/lib/models'
import { fmtInt } from '@/lib/format'
import { usePageTitle } from '@/lib/usePageTitle'

function ConfusionMatrices({ data }: { data: ModelMetricsResponse }) {
  const rows = buildModelRows(data.metrics, 'test').filter((r) => r.confusion)
  if (rows.length === 0) return <p className="text-[13px] text-muted">No confusion matrices were saved with the metrics.</p>
  return (
    <div className="grid gap-6 @2xl:grid-cols-2">
      {rows.map((r) => (
        <ConfusionMatrix key={r.key} name={r.name} matrix={r.confusion!} />
      ))}
    </div>
  )
}

function SplitSummary({ data }: { data: ModelMetricsResponse }) {
  const split = data.metrics.split_info
  if (!split) return <p className="text-[13px] text-muted">Split sizes were not saved with the metrics.</p>
  const items: Array<[string, number | undefined]> = [
    ['Reviews used (after removing duplicate text)', split.total_deduplicated_samples],
    ['Train', split.train_samples],
    ['Validation', split.val_samples],
    ['Test', split.test_samples],
  ]
  const test = split.class_distribution_test
  const reviewers = [split.train_reviewers, split.val_reviewers, split.test_reviewers]
  return (
    <div className="space-y-3">
      <dl className="grid grid-cols-2 gap-2 @xl:grid-cols-4">
        {items.map(([label, value], i) => (
          <div key={label} className="rounded-lg border border-line-soft bg-bg/40 px-3 py-2">
            <dt className="text-[11px] text-muted">{label}</dt>
            <dd className="text-lg font-semibold text-ink tabular-nums">{value === undefined ? 'n/a' : fmtInt(value)}</dd>
            {i > 0 && reviewers[i - 1] !== undefined && <dd className="text-[11px] text-muted">{fmtInt(reviewers[i - 1]!)} reviewers</dd>}
          </div>
        ))}
      </dl>
      <p className="text-xs leading-relaxed text-muted">
        {split.reviewers_shared_between_splits === 0
          ? 'The data is split by reviewer: no reviewer has reviews in more than one of these groups. People write consistently, so a random split would test the models on habits they had already seen and make the scores look better than they are.'
          : split.split_strategy
            ? `Split method: ${split.split_strategy}.`
            : null}
      </p>
      {test && (
        <p className="text-xs leading-relaxed text-muted">
          Test set mix: {fmtInt(test.positive)} positive, {fmtInt(test.neutral)} neutral, {fmtInt(test.negative)} negative. Because positive reviews dominate, accuracy looks high even for weak models, so
          compare macro F1 as well.
        </p>
      )}
    </div>
  )
}

export default function PredictionsPage() {
  usePageTitle('ML Predictions')
  const metrics = useModelMetrics()
  return (
    <PageLayout>
      <PageHeader
        title="ML Predictions"
        description="Try the trained sentiment models on your own text, then compare how well each model did on reviews it had never seen."
      />

      <Card title="Try the sentiment model" subtitle="Calls POST /predict on the backend">
        <PredictForm />
      </Card>

      <Card title="Model comparison" subtitle="Scores on reviews held out from training">
        <ModelTable />
        <Callout className="mt-3">Precision and recall are averaged over the three classes (macro). Two models use TF-IDF text features with a linear classifier; DistilBERT is a fine-tuned neural network (see Deep Learning for how it was trained).</Callout>
      </Card>

      <Card title="Results by category" subtitle="Where the same saved models do better or worse">
        <CategoryMetricsTable />
      </Card>

      <Card title="Where the models make mistakes" subtitle="Confusion matrices on the test set. Rows are the true class, columns are the prediction.">
        <QueryBoundary query={metrics} loadingLabel="Loading confusion matrices" loadingRows={5}>
          {(data) => <ConfusionMatrices data={data} />}
        </QueryBoundary>
      </Card>

      <Card title="How the data was split" subtitle="Models are trained on one part and scored on a part they have not seen">
        <QueryBoundary query={metrics} loadingLabel="Loading split sizes" loadingRows={3}>
          {(data) => <SplitSummary data={data} />}
        </QueryBoundary>
      </Card>
    </PageLayout>
  )
}
