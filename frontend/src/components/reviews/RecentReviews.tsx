import { ArrowRight } from 'lucide-react'
import { useState } from 'react'
import { Link } from 'react-router-dom'
import type { Sentiment } from '@/api/types'
import { useOverview, useReviews } from '@/api/queries'
import { useFilters } from '@/context/FilterContext'
import { fmtInt } from '@/lib/format'
import { Card } from '../ui/Card'
import { QueryBoundary } from '../ui/StateBlocks'
import { Tabs } from '../ui/controls'
import { ReviewCard } from './ReviewCard'

/** The newest reviews for one sentiment, scoped to the selected category. */
export function RecentReviews({ count = 3, className }: { count?: number; className?: string }) {
  const { scope } = useFilters()
  const [sentiment, setSentiment] = useState<Sentiment>('positive')
  const overview = useOverview(scope)
  const reviews = useReviews({ ...scope, sentiment, limit: count, page: 1 })
  const counts = overview.data?.sentiment_distribution

  const params = new URLSearchParams({ sentiment })

  return (
    <Card
      className={className}
      title="Recent Reviews"
      actions={
        <Link to={`/reviews?${params}`} className="inline-flex items-center gap-1 text-xs font-medium text-brand-2 hover:underline">
          View all <ArrowRight size={12} aria-hidden="true" />
        </Link>
      }
    >
      <div className="mb-3">
        <Tabs
          label="Review sentiment"
          size="sm"
          value={sentiment}
          onChange={setSentiment}
          options={[
            { value: 'positive', label: 'Positive', badge: counts ? fmtInt(counts.positive) : undefined },
            { value: 'neutral', label: 'Neutral', badge: counts ? fmtInt(counts.neutral) : undefined },
            { value: 'negative', label: 'Negative', badge: counts ? fmtInt(counts.negative) : undefined },
          ]}
        />
      </div>
      <QueryBoundary query={reviews} isEmpty={(d) => d.reviews.length === 0} emptyTitle="No reviews in this group" emptyHint="Try another sentiment or category." loadingLabel="Loading recent reviews" loadingRows={4}>
        {(data) => (
          <ul className="space-y-2">
            {data.reviews.map((r) => (
              <li key={r.review_id}>
                <ReviewCard review={r} compact />
              </li>
            ))}
          </ul>
        )}
      </QueryBoundary>
    </Card>
  )
}
