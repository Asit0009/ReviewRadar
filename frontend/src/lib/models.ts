import type { ClassicalModelMetrics, ModelMetricsPayload, SplitMetrics } from '@/api/types'

export type EvalSplit = 'test' | 'validation'
export type ModelKey = 'logistic_regression' | 'linear_svc' | 'distilbert'

export interface ModelRow {
  key: ModelKey
  name: string
  /** Short caveat shown next to the name, e.g. DistilBERT was trained on a subset to fit this computer. */
  note: string | null
  accuracy: number | null
  macroF1: number | null
  weightedF1: number | null
  macroPrecision: number | null
  macroRecall: number | null
  confusion: number[][] | null
}

const mean = (values: Array<number | undefined>): number | null => {
  const present = values.filter((v): v is number => typeof v === 'number')
  return present.length === values.length && present.length > 0 ? present.reduce((a, b) => a + b, 0) / present.length : null
}

function row(key: ModelKey, name: string, note: string | null, m: SplitMetrics | undefined): ModelRow {
  return {
    key,
    name,
    note,
    accuracy: m?.accuracy ?? null,
    macroF1: m?.macro_f1 ?? null,
    weightedF1: m?.weighted_f1 ?? null,
    // Newer metrics files store the macro averages; older ones only have per-class values, whose plain mean is the macro average.
    macroPrecision: m?.macro_precision ?? mean([m?.negative_precision, m?.neutral_precision, m?.positive_precision]),
    macroRecall: m?.macro_recall ?? mean([m?.negative_recall, m?.neutral_recall, m?.positive_recall]),
    confusion: m?.confusion_matrix ?? null,
  }
}

const SPLIT_LABEL = { test: 'test', validation: 'validation' } as const

/** Turns the saved /models/metrics payload into one row per model. Anything the backend did not report stays null. */
export function buildModelRows(payload: ModelMetricsPayload | undefined, split: EvalSplit): ModelRow[] {
  const models = payload?.models
  if (!models) return []
  const rows: ModelRow[] = []
  const classical = (key: 'logistic_regression' | 'linear_svc', name: string, m: ClassicalModelMetrics | undefined) => {
    if (m) rows.push(row(key, name, null, m[SPLIT_LABEL[split]]))
  }
  classical('logistic_regression', 'Logistic Regression', models.logistic_regression)
  classical('linear_svc', 'Linear SVC', models.linear_svc)
  const d = models.distilbert
  if (d) {
    const t = d.training
    const note = t?.train_samples ? `trained on ${t.train_samples.toLocaleString('en-US')} reviews` : null
    rows.push(row('distilbert', 'DistilBERT', note, d[SPLIT_LABEL[split]]))
  }
  return rows
}

export function bestByMacroF1(rows: ModelRow[]): ModelRow['key'] | null {
  let best: ModelRow | null = null
  for (const r of rows) {
    if (r.macroF1 !== null && (best === null || r.macroF1 > (best.macroF1 ?? -1))) best = r
  }
  return best?.key ?? null
}
