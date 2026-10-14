import { fmtInt } from '@/lib/format'

const LABELS = ['Negative', 'Neutral', 'Positive'] as const

/**
 * 3x3 confusion matrix. Rows are the true class and columns the predicted class, in the same order the
 * backend uses for sentiment_encoded (0 = negative, 1 = neutral, 2 = positive).
 * Shading shows each cell as a share of its row, so the rare neutral class is still readable.
 */
export function ConfusionMatrix({ matrix, name }: { matrix: number[][]; name: string }) {
  return (
    <figure>
      <figcaption className="mb-2 text-[13px] font-medium text-ink">{name}</figcaption>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[300px] border-separate border-spacing-1 text-center text-xs" aria-label={`${name} confusion matrix on the test set`}>
          <thead>
            <tr>
              <td />
              <th scope="colgroup" colSpan={3} className="pb-1 font-medium text-muted">
                Predicted
              </th>
            </tr>
            <tr>
              <td />
              {LABELS.map((l) => (
                <th key={l} scope="col" className="font-medium text-muted">
                  {l}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => {
              const total = row.reduce((a, b) => a + b, 0) || 1
              return (
                <tr key={i}>
                  <th scope="row" className="pr-1 text-right font-medium whitespace-nowrap text-muted">
                    <span className="hidden sm:inline">Actual </span>
                    {LABELS[i]}
                  </th>
                  {row.map((value, j) => {
                    const share = value / total
                    const onDiagonal = i === j
                    return (
                      <td
                        key={j}
                        className="rounded-md px-1 py-2.5 font-medium tabular-nums text-ink"
                        style={{
                          background: onDiagonal ? `rgb(34 197 139 / ${0.12 + share * 0.55})` : `rgb(244 80 122 / ${share * 0.7})`,
                          border: '1px solid var(--color-line-soft)',
                        }}
                        title={`${fmtInt(value)} reviews (${(share * 100).toFixed(0)}% of actual ${LABELS[i]!.toLowerCase()})`}
                      >
                        {fmtInt(value)}
                      </td>
                    )
                  })}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </figure>
  )
}
