import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { shouldRetry } from '@/api/queries'
import App from './App'
import { AssistantProvider } from './context/AssistantContext'
import { FilterProvider } from './context/FilterContext'
import { UiProvider } from './context/UiContext'
import './index.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: shouldRetry, refetchOnWindowFocus: false },
  },
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <FilterProvider>
          <UiProvider>
            <AssistantProvider>
              <App />
            </AssistantProvider>
          </UiProvider>
        </FilterProvider>
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
)
