import { lazy, Suspense } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { AppShell } from './components/layout/AppShell'
import { LoadingBlock } from './components/ui/StateBlocks'
import OverviewPage from './pages/OverviewPage'

// Overview is the landing page, so it ships in the main bundle. The rest load on first visit.
const ReviewsPage = lazy(() => import('./pages/ReviewsPage'))
const SentimentPage = lazy(() => import('./pages/SentimentPage'))
const TopicsPage = lazy(() => import('./pages/TopicsPage'))
const PredictionsPage = lazy(() => import('./pages/PredictionsPage'))
const DeepLearningPage = lazy(() => import('./pages/DeepLearningPage'))
const AssistantPage = lazy(() => import('./pages/AssistantPage'))
const MonitoringPage = lazy(() => import('./pages/MonitoringPage'))

export default function App() {
  return (
    <Suspense fallback={<LoadingBlock label="Loading page" rows={6} className="p-6" />}>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<OverviewPage />} />
          <Route path="reviews" element={<ReviewsPage />} />
          <Route path="sentiment" element={<SentimentPage />} />
          <Route path="topics" element={<TopicsPage />} />
          <Route path="predictions" element={<PredictionsPage />} />
          <Route path="deep-learning" element={<DeepLearningPage />} />
          <Route path="assistant" element={<AssistantPage />} />
          <Route path="monitoring" element={<MonitoringPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </Suspense>
  )
}
