import { Frown, Meh, Smile, ThumbsUp } from 'lucide-react'
import { Fragment, useState } from 'react'
import { Link } from 'react-router-dom'
import type { ReviewItem } from '@/api/types'
import { categoryLabel, fmtDate } from '@/lib/format'
import { cleanReviewText, splitByTerm } from '@/lib/text'
import { SENTIMENT_META, isSentiment } from '@/lib/theme'
import { Chip, SentimentBadge, StarRating } from '../ui/controls'

const ICONS = { positive: Smile, neutral: Meh, negative: Frown } as const

function Highlighted({ text, term }: { text: string; term?: string }) {
  if (!term) return <>{text}</>
  return (
    <>
      {splitByTerm(text, term).map((part, i) =>
        part.match ? (
          <mark key={i} className="rounded-sm bg-neu/30 px-0.5 text-ink">
            {part.text}
          </mark>
        ) : (
          <Fragment key={i}>{part.text}</Fragment>
        ),
      )}
    </>
  )
}

interface ReviewCardProps {
  review: ReviewItem
  /** Search term to highlight inside the text. */
  highlight?: string
  /** Short layout for dashboard panels: two lines of text and fewer details. */
  compact?: boolean
}

export function ReviewCard({ review, highlight, compact = false }: ReviewCardProps) {
  const [expanded, setExpanded] = useState(false)
  const sentiment = isSentiment(review.sentiment) ? review.sentiment : null
  const Icon = sentiment ? ICONS[sentiment] : Meh
  const color = sentiment ? SENTIMENT_META[sentiment].color : '#8b98c2'
  const text = cleanReviewText(review.text)
  const title = cleanReviewText(review.title)
  const isLong = text.length > 240
  const clamp = compact ? 'line-clamp-2' : expanded ? '' : 'line-clamp-3'

  return (
    <article className="flex gap-3 rounded-lg border border-line-soft bg-card-2/40 p-3">
      <div aria-hidden="true" className="flex size-8 shrink-0 items-center justify-center rounded-full" style={{ background: `${color}26`, color }}>
        <Icon size={17} />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-start justify-between gap-x-3 gap-y-1">
          {title ? (
            <h3 className="min-w-0 text-[13px] leading-snug font-semibold text-ink">
              <Highlighted text={title} term={highlight} />
            </h3>
          ) : (
            <span />
          )}
          {!compact && <SentimentBadge sentiment={review.sentiment} />}
        </div>
        {text && (
          <p className={`mt-1 text-[13px] leading-relaxed text-ink-2 ${clamp}`}>
            <Highlighted text={text} term={highlight} />
          </p>
        )}
        {!compact && isLong && (
          <button type="button" onClick={() => setExpanded((v) => !v)} className="mt-1 text-xs font-medium text-brand-2 hover:underline" aria-expanded={expanded}>
            {expanded ? 'Show less' : 'Show more'}
          </button>
        )}
        {!compact && review.product_title && <p className="mt-1.5 truncate text-xs text-muted">Product: {review.product_title}</p>}
        <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1.5 text-xs text-muted">
          <StarRating rating={review.rating} size={12} />
          <time dateTime={review.review_date}>{fmtDate(review.review_date)}</time>
          <Chip tone="blue">{categoryLabel(review.category)}</Chip>
          {review.topic_id !== null && review.topic_label && (
            <Link
              to={`/reviews?topic=${review.topic_id}`}
              title={`Main topic found by the NMF topic model${review.topic_score !== null ? ` (${Math.round(review.topic_score * 100)}% of this review's topic weight)` : ''}. Click to list reviews with this topic.`}
            >
              <Chip tone="violet">{review.topic_label}</Chip>
            </Link>
          )}
          {review.verified_purchase && <Chip tone="teal">Verified purchase</Chip>}
          {!compact && review.helpful_vote > 0 && (
            <span className="inline-flex items-center gap-1" title="Helpful votes from other customers">
              <ThumbsUp size={12} aria-hidden="true" /> {review.helpful_vote}
            </span>
          )}
        </div>
      </div>
    </article>
  )
}
