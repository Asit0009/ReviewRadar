import { useCategories, useOverview } from '@/api/queries'
import { CategorySelect } from '@/components/CategorySelect'
import { DateRangeFilter } from '@/components/DateRangeFilter'
import { CategoryCompare } from '@/components/charts/CategoryCompare'
import { SentimentDonut } from '@/components/charts/SentimentDonut'
import { PageLayout } from '@/components/layout/PageLayout'
import { RatingPanel, TrendPanel, useScopeLabel } from '@/components/overview/Panels'
import { RecentReviews } from '@/components/reviews/RecentReviews'
import { Card } from '@/components/ui/Card'
import { Callout, PageHeader } from '@/components/ui/controls'
import { QueryBoundary } from '@/components/ui/StateBlocks'
import { useFilters } from '@/context/FilterContext'
import { fmtInt, fmtPct } from '@/lib/format'
import { SENTIMENT_META, SENTIMENTS } from '@/lib/theme'
import { usePageTitle } from '@/lib/usePageTitle'

function SentimentSplit() {
  const { scope } = useFilters()
  const query = useOverview(scope)
  const label = useScopeLabel()
  return (
    <Card title="Sentiment Split" subtitle={label} className="h-full">
      <QueryBoundary query={query} loadingLabel="Loading sentiment split" loadingRows={5}>
        {(data) => (
          <div className="space-y-4">
            <SentimentDonut overview={data} size={150} />
            <table className="w-full text-[13px]">
              <caption className="sr-only">Review counts by sentiment</caption>
              <thead>
                <tr className="text-left text-xs text-muted">
                  <th scope="col" className="py-1 font-medium">Sentiment</th>
                  <th scope="col" className="py-1 text-right font-medium">Reviews</th>
                  <th scope="col" className="py-1 text-right font-medium">Share</th>
                </tr>
              </thead>
              <tbody>
                {SENTIMENTS.map((s) => (
                  <tr key={s}>
                    <th scope="row" className="border-t border-line-soft py-1.5 text-left font-medium" style={{ color: SENTIMENT_META[s].color }}>
                      {SENTIMENT_META[s].label}
                    </th>
                    <td className="border-t border-line-soft py-1.5 text-right text-ink-2 tabular-nums">{fmtInt(data.sentiment_distribution[s])}</td>
                    <td className="border-t border-line-soft py-1.5 text-right text-ink-2 tabular-nums">{fmtPct(data.sentiment_percentages[s])}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </QueryBoundary>
    </Card>
  )
}

export default function SentimentPage() {
  usePageTitle('Sentiment Analysis')
  const categories = useCategories()
  return (
    <PageLayout>
      <PageHeader
        title="Sentiment Analysis"
        description="How positive or negative the reviews are, how that changes month by month, and how ratings differ between categories."
        actions={
          <>
            <CategorySelect className="w-52" />
            <DateRangeFilter className="w-64" />
          </>
        }
      />
      <Callout>Sentiment here is not judged from the text. It is taken from the star rating: 1-2 stars is negative, 3 is neutral and 4-5 is positive.</Callout>
      <div className="grid gap-4 @3xl:grid-cols-2">
        <SentimentSplit />
        <RatingPanel />
      </div>
      <TrendPanel />
      <Card title="Compare ratings across categories" subtitle="Whole dataset. Click a category to filter the dashboard (dates are not applied to this table)">
        <QueryBoundary query={categories} isEmpty={(d) => d.categories.length === 0} emptyTitle="No categories found" loadingLabel="Loading category statistics" loadingRows={6}>
          {(data) => <CategoryCompare categories={data.categories} />}
        </QueryBoundary>
      </Card>
      <RecentReviews count={4} />
    </PageLayout>
  )
}
