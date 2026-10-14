import { PageLayout } from '@/components/layout/PageLayout'
import { Hero } from '@/components/overview/Hero'
import { KpiRow } from '@/components/overview/Kpis'
import { FilterCard, ModelsPanel, MonitoringPanel, RatingPanel, SentimentCard, TopicsPanel, TrendPanel } from '@/components/overview/Panels'
import { RecentReviews } from '@/components/reviews/RecentReviews'
import { usePageTitle } from '@/lib/usePageTitle'

export default function OverviewPage() {
  usePageTitle('Overview')
  return (
    <PageLayout
      rail={
        <>
          <FilterCard />
          <SentimentCard />
        </>
      }
    >
      <Hero />
      {/* The filter and sentiment cards live in the right column on wide screens; below 1280px they sit inline. */}
      <div className="xl:hidden">
        <FilterCard />
      </div>
      <KpiRow />
      <div className="xl:hidden">
        <SentimentCard />
      </div>
      <div className="grid gap-4 @3xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <TrendPanel />
        <RatingPanel />
      </div>
      <div className="grid gap-4 @3xl:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
        <TopicsPanel />
        <ModelsPanel />
      </div>
      <div className="grid gap-4 @3xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <RecentReviews />
        <MonitoringPanel />
      </div>
    </PageLayout>
  )
}
