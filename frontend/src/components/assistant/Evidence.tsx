import { forwardRef, useState } from 'react'
import type { EvidenceReview, NumericalFacts } from '@/api/types'
import { categoryLabel, fmtDate, fmtInt } from '@/lib/format'
import { cleanReviewText } from '@/lib/text'
import { Chip, SentimentBadge, StarRating } from '../ui/controls'

const EvidenceItem = forwardRef<HTMLLIElement, { review: EvidenceReview }>(function EvidenceItem({ review }, ref) {
  const [expanded, setExpanded] = useState(false)
  const text = cleanReviewText(review.full_text || review.text)
  const title = cleanReviewText(review.title)
  return (
    <li
      ref={ref}
      tabIndex={-1}
      data-review-id={review.review_id}
      className="rounded-lg border border-line-soft bg-bg/50 p-2.5 transition-shadow outline-none data-[cited=true]:ring-2 data-[cited=true]:ring-brand-2"
    >
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <StarRating rating={review.rating} size={11} />
        <SentimentBadge sentiment={review.sentiment} />
        <Chip tone="blue">{categoryLabel(review.category)}</Chip>
        <span
          className="ml-auto text-[11px] text-muted tabular-nums"
          title="Similarity between your question and this review (cosine, 0 to 1). Higher means closer in meaning."
        >
          match {review.score.toFixed(2)}
        </span>
      </div>
      <p className="mt-1.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] text-muted">
        <span title={`Review id ${review.review_id}. The answer cites reviews by this id.`}>
          ID <code className="font-mono text-ink-2">{review.review_id.slice(0, 8)}</code>
        </span>
        {review.verified_purchase && <span>Verified purchase</span>}
        {review.helpful_vote > 0 && <span>{fmtInt(review.helpful_vote)} found helpful</span>}
      </p>
      {title && <p className="mt-1 text-xs font-semibold text-ink">{title}</p>}
      <p className={`mt-1 text-xs leading-relaxed text-ink-2 ${expanded ? '' : 'line-clamp-3'}`}>{text}</p>
      <div className="mt-1 flex items-center justify-between text-[11px] text-muted">
        {text.length > 180 ? (
          <button type="button" className="font-medium text-brand-2 hover:underline" aria-expanded={expanded} onClick={() => setExpanded((v) => !v)}>
            {expanded ? 'Show less' : 'Show more'}
          </button>
        ) : (
          <span />
        )}
        <time dateTime={review.timestamp_utc}>{fmtDate(review.timestamp_utc.slice(0, 10))}</time>
      </div>
    </li>
  )
})

export function EvidenceList({ reviews }: { reviews: EvidenceReview[] }) {
  if (reviews.length === 0) {
    return <p className="text-xs text-muted">No similar reviews were found for this question.</p>
  }
  return (
    <ul className="space-y-2">
      {reviews.map((r) => (
        <EvidenceItem key={r.review_id} review={r} />
      ))}
    </ul>
  )
}

/** The exact counts the backend computed with SQL for the chosen category. */
export function FactChips({ facts }: { facts: NumericalFacts }) {
  const items: Array<[string, string]> = []
  if (facts.total_reviews !== undefined) items.push(['Reviews', fmtInt(facts.total_reviews)])
  if (facts.average_rating !== undefined && facts.average_rating !== null) items.push(['Avg rating', facts.average_rating.toFixed(2)])
  if (facts.positive_pct !== undefined) items.push(['Positive', `${facts.positive_pct}%`])
  if (facts.neutral_pct !== undefined) items.push(['Neutral', `${facts.neutral_pct}%`])
  if (facts.negative_pct !== undefined) items.push(['Negative', `${facts.negative_pct}%`])
  if (facts.verified_purchase_pct !== undefined) items.push(['Verified', `${facts.verified_purchase_pct}%`])
  if (items.length === 0) return null
  return (
    <dl className="grid grid-cols-3 gap-1.5">
      {items.map(([label, value]) => (
        <div key={label} className="rounded-md border border-line-soft bg-bg/50 px-2 py-1.5">
          <dt className="text-[10px] tracking-wide text-muted uppercase">{label}</dt>
          <dd className="text-[13px] font-semibold text-ink tabular-nums">{value}</dd>
        </div>
      ))}
    </dl>
  )
}
