import { fireEvent, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError } from '@/api/client'
import type { AssistantChatResponse, ReviewItem } from '@/api/types'
import MonitoringPage from '@/pages/MonitoringPage'
import ReviewsPage from '@/pages/ReviewsPage'
import { healthyBody, mockApi, renderWithProviders } from '@/test/helpers'
import { AnswerBody } from './assistant/AnswerBody'
import { AssistantChat, SUGGESTED_QUESTIONS } from './assistant/AssistantChat'
import { ModelTableView } from './charts/ModelTable'
import { PredictForm } from './PredictForm'
import { ReviewCard } from './reviews/ReviewCard'
import { Pagination } from './ui/Pagination'
import { lastMonthIsIncomplete } from './overview/Panels'
import { ErrorBlock } from './ui/StateBlocks'

afterEach(() => vi.unstubAllGlobals())

const review = (over: Partial<ReviewItem> = {}): ReviewItem => ({
  review_id: 'r1',
  category: 'Video_Games',
  rating: 2,
  sentiment: 'negative',
  title: 'Battery died',
  text: 'Stopped holding charge<br />after a week.',
  product_title: '',
  review_date: '2023-03-13',
  verified_purchase: true,
  helpful_vote: 3,
  topic_id: null,
  topic_label: null,
  topic_score: null,
  ...over,
})

const chatResponse = (over: Partial<AssistantChatResponse> = {}): AssistantChatResponse => ({
  response: 'No written answer was generated because no Gemini API key is configured on the server.',
  llm_used: false,
  mode: 'retrieval_only',
  status: 'setup_required',
  llm_model: null,
  error_detail: null,
  numerical_facts: { scope: 'All Categories', total_reviews: 29950, average_rating: 4.31, positive_pct: 84.7, neutral_pct: 6.3, negative_pct: 9.0 },
  evidence_reviews: [
    {
      review_id: 'e1',
      vector_id: 7,
      score: 0.52,
      rating: 1,
      sentiment: 'negative',
      category: 'Electronics',
      title: 'Stick drift',
      text: 'The stick drifts',
      full_text: 'The stick drifts after three weeks.',
      product_title: '',
      timestamp_utc: '2023-02-12T17:45:29Z',
      verified_purchase: true,
      helpful_vote: 0,
    },
  ],
  evidence_count: 1,
  retrieval: { indexed_reviews: 29950, eligible_reviews: 29950, coverage_pct: 100, min_score: 0.4, scope: 'All Categories' },
  ...over,
})

describe('ErrorBlock', () => {
  it('shows "Not available yet" for a 503 and keeps the backend message', () => {
    renderWithProviders(<ErrorBlock error={new ApiError('Topic models not yet trained.', 503, 'unavailable')} />)
    expect(screen.getByText('Not available yet')).toBeInTheDocument()
    expect(screen.getByText('Topic models not yet trained.')).toBeInTheDocument()
  })

  it('tells the user when the API cannot be reached and offers a retry', async () => {
    const onRetry = vi.fn()
    renderWithProviders(<ErrorBlock error={new ApiError('Could not reach the API', null, 'network')} onRetry={onRetry} />)
    expect(screen.getByText("Can't reach the API")).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: /try again/i }))
    expect(onRetry).toHaveBeenCalledOnce()
  })
})

describe('AnswerBody', () => {
  it('renders bold text and lists', () => {
    renderWithProviders(<AnswerBody text={'**Battery** is the top issue.\n\n- Short life\n- Slow charging'} />)
    expect(screen.getByText('Battery').tagName).toBe('STRONG')
    expect(screen.getAllByRole('listitem')).toHaveLength(2)
  })

  it('never turns model output into HTML', () => {
    const { container } = renderWithProviders(<AnswerBody text={'<img src=x onerror=alert(1)> <script>alert(1)</script>'} />)
    expect(container.querySelector('img, script')).toBeNull()
    expect(container).toHaveTextContent('<img src=x onerror=alert(1)>')
  })

  it('shortens review-id citations', () => {
    renderWithProviders(<AnswerBody text="Customers complain [Review ID: 4139b50f735afc5a] about it." />)
    expect(screen.getByText('4139b50f')).toBeInTheDocument()
  })
})

describe('AnswerBody citations', () => {
  const full = '4139b50f735afc5a'
  const other = 'ffffffffffffffff'

  it('turns a cited review that is in the evidence into a button that reports the full id', async () => {
    const onCite = vi.fn()
    renderWithProviders(<AnswerBody text={`Fit problems [Review ID: ${full}] are common.`} evidenceIds={new Set([full])} onCite={onCite} />)
    await userEvent.click(screen.getByRole('button', { name: '4139b50f' }))
    expect(onCite).toHaveBeenCalledWith(full)
  })

  it('marks a cited id that is not in the evidence and does not make it clickable', () => {
    const onCite = vi.fn()
    renderWithProviders(<AnswerBody text={`Invented [Review ID: ${other}] claim.`} evidenceIds={new Set([full])} onCite={onCite} />)
    const mark = screen.getByText('ffffffff?')
    expect(mark.tagName).toBe('SPAN')
    expect(mark).toHaveAttribute('title', expect.stringMatching(/not among the evidence/))
    expect(screen.queryByRole('button')).not.toBeInTheDocument()
  })

  it('handles several ids in one citation', () => {
    renderWithProviders(<AnswerBody text={`Both agree [Review ID: ${full}, ${other}].`} evidenceIds={new Set([full, other])} onCite={() => {}} />)
    expect(screen.getAllByRole('button')).toHaveLength(2)
  })
})

describe('ReviewCard', () => {
  it('shows cleaned text, rating, date and chips', () => {
    renderWithProviders(<ReviewCard review={review()} />)
    expect(screen.getByText('Stopped holding charge after a week.')).toBeInTheDocument()
    expect(screen.getByRole('img', { name: '2 out of 5 stars' })).toBeInTheDocument()
    expect(screen.getByText('Mar 13, 2023')).toBeInTheDocument()
    expect(screen.getByText('Video Games')).toBeInTheDocument()
    expect(screen.getByText('Verified purchase')).toBeInTheDocument()
    expect(screen.getByText('Negative')).toBeInTheDocument()
  })

  it('highlights the search term', () => {
    const { container } = renderWithProviders(<ReviewCard review={review()} highlight="charge" />)
    expect(container.querySelector('mark')).toHaveTextContent('charge')
  })
})

describe('ModelTableView', () => {
  it('marks missing metrics as n/a instead of showing numbers', () => {
    renderWithProviders(
      <ModelTableView
        bestKey="logistic_regression"
        rows={[
          { key: 'logistic_regression', name: 'Logistic Regression', note: null, accuracy: 0.84, macroF1: 0.618, weightedF1: 0.85, macroPrecision: 0.58, macroRecall: 0.66, confusion: null },
          { key: 'distilbert', name: 'DistilBERT', note: '150-review CPU sample run', accuracy: 0.7867, macroF1: 0.2935, weightedF1: null, macroPrecision: null, macroRecall: null, confusion: null },
        ]}
      />,
    )
    const row = screen.getByRole('row', { name: /DistilBERT/ })
    expect(within(row).getAllByText('n/a')).toHaveLength(2)
    expect(within(row).getByText('150-review CPU sample run')).toBeInTheDocument()
    expect(screen.getByRole('row', { name: /Logistic Regression/ })).toHaveTextContent('0.618')
  })
})

describe('Pagination', () => {
  it('moves between pages and clamps the jump box', async () => {
    const onPageChange = vi.fn()
    renderWithProviders(<Pagination page={2} totalPages={5} onPageChange={onPageChange} />)
    await userEvent.click(screen.getByRole('button', { name: /next/i }))
    expect(onPageChange).toHaveBeenLastCalledWith(3)
    await userEvent.click(screen.getByRole('button', { name: /previous/i }))
    expect(onPageChange).toHaveBeenLastCalledWith(1)
    const box = screen.getByLabelText('Go to page')
    await userEvent.clear(box)
    await userEvent.type(box, '99{Enter}')
    expect(onPageChange).toHaveBeenLastCalledWith(5)
  })

  it('disables Previous on the first page', () => {
    renderWithProviders(<Pagination page={1} totalPages={3} onPageChange={() => {}} />)
    expect(screen.getByRole('button', { name: /previous/i })).toBeDisabled()
  })
})

describe('PredictForm', () => {
  it('validates the text before calling the API', async () => {
    const fetchMock = mockApi(() => ({ body: {} }))
    renderWithProviders(<PredictForm />)
    await userEvent.type(screen.getByLabelText('Review text'), ' a ')
    await userEvent.click(screen.getByRole('button', { name: /predict sentiment/i }))
    expect(screen.getByRole('alert')).toHaveTextContent('at least 2 characters')
    expect(fetchMock.mock.calls.filter(([url]) => String(url).includes('/predict'))).toHaveLength(0)
  })

  it('sends the text and chosen model, then shows the label and probabilities', async () => {
    const fetchMock = mockApi(() => ({
      body: {
        text: 'Terrible',
        model_type: 'logistic_regression',
        predicted_sentiment: 'negative',
        sentiment_encoded: 0,
        probabilities: { negative: 0.92, neutral: 0.03, positive: 0.05 },
        model_used: 'TF-IDF + LogisticRegression',
        is_rating_derived_label: true,
      },
    }))
    renderWithProviders(<PredictForm />)
    await userEvent.type(screen.getByLabelText('Review text'), 'Terrible product')
    await userEvent.click(screen.getByRole('button', { name: /predict sentiment/i }))
    await screen.findByText('Predicted sentiment')
    const predictCall = fetchMock.mock.calls.find(([url]) => String(url).includes('/predict'))
    expect(JSON.parse(String(predictCall?.[1]?.body))).toEqual({ text: 'Terrible product', model_type: 'logistic_regression' })
    expect(screen.getByText('92.0%')).toBeInTheDocument()
    expect(screen.getByText(/trained on labels taken from star ratings/i)).toBeInTheDocument()
  })

  it('explains that Linear SVC returns no probabilities', async () => {
    mockApi(() => ({
      body: { text: 'ok', model_type: 'linear_svc', predicted_sentiment: 'positive', sentiment_encoded: 2, probabilities: null, model_used: 'TF-IDF + LinearSVC', is_rating_derived_label: true },
    }))
    renderWithProviders(<PredictForm />)
    await userEvent.type(screen.getByLabelText('Review text'), 'Works great')
    await userEvent.click(screen.getByLabelText(/Linear SVC/))
    await userEvent.click(screen.getByRole('button', { name: /predict sentiment/i }))
    expect(await screen.findByText(/only returns a label/i)).toBeInTheDocument()
  })

  it('refuses to show a result that came from a different model than the one chosen', async () => {
    mockApi(() => ({
      body: { text: 'ok', model_type: 'logistic_regression', predicted_sentiment: 'positive', sentiment_encoded: 2, probabilities: null, model_used: 'TF-IDF + LogisticRegression', is_rating_derived_label: true },
    }))
    renderWithProviders(<PredictForm />)
    await userEvent.type(screen.getByLabelText('Review text'), 'Works great')
    await userEvent.click(screen.getByLabelText(/Linear SVC/))
    await userEvent.click(screen.getByRole('button', { name: /predict sentiment/i }))
    expect(await screen.findByText(/You asked for linear_svc, but the server answered with logistic_regression/)).toBeInTheDocument()
    expect(screen.queryByText('Predicted sentiment')).not.toBeInTheDocument()
  })

  it('shows the API error instead of a result', async () => {
    mockApi(() => ({ status: 503, body: { detail: 'Sentiment model artifacts not loaded. Train models first.' } }))
    renderWithProviders(<PredictForm />)
    await userEvent.type(screen.getByLabelText('Review text'), 'Works great')
    await userEvent.click(screen.getByRole('button', { name: /predict sentiment/i }))
    expect(await screen.findByText('Not available yet')).toBeInTheDocument()
    expect(screen.queryByText('Predicted sentiment')).not.toBeInTheDocument()
  })
})

describe('AssistantChat', () => {
  it('sends a suggested question to /assistant/chat and shows facts, evidence and the missing-LLM notice', async () => {
    const fetchMock = mockApi(() => ({ body: chatResponse() }))
    renderWithProviders(<AssistantChat variant="panel" />)

    await userEvent.click(screen.getByRole('button', { name: SUGGESTED_QUESTIONS[0] }))

    expect(await screen.findByText('No written answer: no Gemini API key is configured')).toBeInTheDocument()
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(new URL(url).pathname).toBe('/api/v1/assistant/chat')
    expect(JSON.parse(String(init.body))).toEqual({ question: SUGGESTED_QUESTIONS[0], category: null, top_k: 4 })

    expect(screen.getByText('29,950', { selector: 'dd' })).toBeInTheDocument()
    expect(screen.getByText(/Searched 29,950 of 29,950 reviews \(100%\)/)).toBeInTheDocument()
    expect(screen.getByText('Evidence from 1 review')).toBeInTheDocument()
    expect(screen.getByText(/match 0\.52/)).toBeInTheDocument()
  })

  it('shows the model answer when the LLM was used', async () => {
    mockApi(() => ({ body: chatResponse({ llm_used: true, mode: 'llm', status: 'success', llm_model: 'gemini-3.5-flash', response: 'Customers mention **stick drift**.' }) }))
    renderWithProviders(<AssistantChat variant="panel" />)
    await userEvent.type(screen.getByLabelText(/ask a question/i), 'What breaks first?')
    await userEvent.click(screen.getByRole('button', { name: /send question/i }))
    expect(await screen.findByText('stick drift')).toBeInTheDocument()
    expect(screen.getByText(/Written by/)).toHaveTextContent('Written by gemini-3.5-flash')
    expect(screen.queryByText(/No written answer/)).not.toBeInTheDocument()
  })

  it('shows an error with a working retry, and does not invent an answer', async () => {
    let calls = 0
    mockApi(() => (++calls === 1 ? { status: 503, body: { detail: 'Assistant service is not initialized.' } } : { body: chatResponse() }))
    renderWithProviders(<AssistantChat variant="panel" />)
    await userEvent.type(screen.getByLabelText(/ask a question/i), 'Any complaints?')
    await userEvent.click(screen.getByRole('button', { name: /send question/i }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Assistant service is not initialized.')
    expect(screen.queryByText(/Evidence from/)).not.toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: /try again/i }))
    expect(await screen.findByText('Evidence from 1 review')).toBeInTheDocument()
  })

  it('shows the review id and metadata on evidence, and a citation opens and highlights its card', async () => {
    const evidence = { ...chatResponse().evidence_reviews[0]!, review_id: 'abcdef0123456789', helpful_vote: 3, verified_purchase: true }
    mockApi(() => ({
      body: chatResponse({ llm_used: true, mode: 'llm', status: 'success', llm_model: 'gemini-test', response: 'The stick drifts [Review ID: abcdef0123456789].', evidence_reviews: [evidence] }),
    }))
    renderWithProviders(<AssistantChat variant="panel" />)
    await userEvent.type(screen.getByLabelText(/ask a question/i), 'What breaks first?')
    await userEvent.click(screen.getByRole('button', { name: /send question/i }))
    const citation = await screen.findByRole('button', { name: 'abcdef01' })
    await userEvent.click(screen.getByText('Evidence from 1 review')) // closed in the narrow panel: open it
    expect(screen.getByText('abcdef01', { selector: 'code' })).toBeInTheDocument()
    expect(screen.getByText('Verified purchase')).toBeInTheDocument()
    expect(screen.getByText('3 found helpful')).toBeInTheDocument()
    await userEvent.click(screen.getByText('Evidence from 1 review')) // close it again, then use the citation
    await userEvent.click(citation)
    const card = document.querySelector('[data-review-id="abcdef0123456789"]') as HTMLElement
    expect(card.closest('details')).toHaveProperty('open', true)
    expect(card.dataset.cited).toBe('true')
  })

  it('does not send questions shorter than 2 characters', async () => {
    const fetchMock = mockApi(() => ({ body: chatResponse() }))
    renderWithProviders(<AssistantChat variant="panel" />)
    await userEvent.type(screen.getByLabelText(/ask a question/i), 'a')
    expect(screen.getByRole('button', { name: /send question/i })).toBeDisabled()
    expect(fetchMock).not.toHaveBeenCalled()
  })
})

describe('ReviewsPage', () => {
  /** Mock API: /reviews calls are recorded so tests can check exactly which filters were sent. */
  const handler = (calls: URL[], options: { reviewsStatus?: number } = {}) => (url: URL) => {
    const path = url.pathname
    if (path.endsWith('/categories')) {
      return { body: { total_categories: 1, categories: [{ category: 'Video_Games', review_count: 1997, unique_products: 1718, mean_rating: 4.3, verified_pct: 68.8, mean_text_length: 467 }] } }
    }
    if (path.endsWith('/topics')) {
      const topic = { top_words: ['battery'], sample_reviews: [], prevalence_count: 10, prevalence_pct: 50, avg_rating: 3, sentiment_counts: { positive: 1, neutral: 1, negative: 1 } }
      return { body: { total_topics: 2, topics: [{ topic_id: 0, label: 'Battery & Charge', ...topic }, { topic_id: 1, label: 'Fit & Size', ...topic }], reviews_considered: 20, review_tags_available: true } }
    }
    if (path.endsWith('/overview')) {
      return { body: { total_reviews: 100, total_products: 10, total_users: 90, average_rating: 4.2, date_min: '2020-01-01', date_max: '2023-03-16', sentiment_distribution: { positive: 80, neutral: 10, negative: 10 }, sentiment_percentages: { positive: 80, neutral: 10, negative: 10 } } }
    }
    calls.push(url)
    if (options.reviewsStatus) return { status: options.reviewsStatus, body: { detail: 'Processed dataset is not available yet.' } }
    const empty = url.searchParams.get('search') === 'zzz'
    return {
      body: {
        total_records: empty ? 0 : 21,
        page: Number(url.searchParams.get('page') ?? 1),
        limit: 20,
        total_pages: empty ? 1 : 2,
        reviews: empty ? [] : [review({ topic_id: 0, topic_label: 'Battery & Charge', topic_score: 0.62 })],
      },
    }
  }
  const lastParams = (calls: URL[]) => Object.fromEntries(calls.at(-1)!.searchParams)

  it('turns URL filters into API parameters', async () => {
    const calls: URL[] = []
    mockApi(handler(calls))
    renderWithProviders(<ReviewsPage />, { route: '/reviews?q=battery&sentiment=negative&min=1&max=2&page=2&sort=rating-asc&topic=0&verified=1' })
    // The search term is wrapped in <mark>, so find the card by role rather than by its full title text.
    await screen.findByRole('article')
    expect(lastParams(calls)).toEqual({
      page: '2', limit: '20', sentiment: 'negative', min_rating: '1', max_rating: '2', search: 'battery',
      verified_only: 'true', topic_id: '0', sort: 'rating', order: 'asc',
    })
    expect(screen.getByText('Page 2 of 2')).toBeInTheDocument()
  })

  it('sends the selected category to the API', async () => {
    const calls: URL[] = []
    mockApi(handler(calls))
    renderWithProviders(<ReviewsPage />, { route: '/reviews' })
    await screen.findByRole('article')
    await userEvent.selectOptions(screen.getByLabelText('Category'), 'Video_Games')
    await waitFor(() => expect(lastParams(calls).category).toBe('Video_Games'))
  })

  it('applies the date range in the API request, not by trimming results on screen', async () => {
    const calls: URL[] = []
    mockApi(handler(calls))
    renderWithProviders(<ReviewsPage />, { route: '/reviews' })
    await screen.findByRole('article')
    expect(lastParams(calls)).not.toHaveProperty('date_from')
    fireEvent.change(screen.getByLabelText('From'), { target: { value: '2022-01-01' } })
    await waitFor(() => expect(lastParams(calls).date_from).toBe('2022-01-01'))
    fireEvent.change(screen.getByLabelText('To'), { target: { value: '2022-12-31' } })
    await waitFor(() => expect(lastParams(calls)).toMatchObject({ date_from: '2022-01-01', date_to: '2022-12-31' }))
    // Clearing returns to the unfiltered request, which react-query already has cached, so it is served without a new call.
    await userEvent.click(screen.getByRole('button', { name: 'Clear dates' }))
    await waitFor(() => expect(screen.getByLabelText('From')).toHaveValue(''))
    expect(screen.getByLabelText('To')).toHaveValue('')
    expect(screen.queryByRole('button', { name: 'Clear dates' })).not.toBeInTheDocument()
  })

  it('warns about a reversed date range and does not send it (the API would answer 422)', async () => {
    const calls: URL[] = []
    mockApi(handler(calls))
    renderWithProviders(<ReviewsPage />, { route: '/reviews' })
    await screen.findByRole('article')
    fireEvent.change(screen.getByLabelText('From'), { target: { value: '2023-01-01' } })
    fireEvent.change(screen.getByLabelText('To'), { target: { value: '2022-01-01' } })
    expect(await screen.findByText(/start date must be on or before the end date/i)).toBeInTheDocument()
    await waitFor(() => expect(lastParams(calls)).not.toHaveProperty('date_to'))
  })

  it('sends sort order, topic and verified-only choices', async () => {
    const calls: URL[] = []
    mockApi(handler(calls))
    renderWithProviders(<ReviewsPage />, { route: '/reviews' })
    await screen.findByRole('article')
    await userEvent.selectOptions(screen.getByLabelText('Sort by'), 'rating-asc')
    await waitFor(() => expect(lastParams(calls)).toMatchObject({ sort: 'rating', order: 'asc' }))
    await screen.findByRole('option', { name: 'Fit & Size' }) // the topic list loads from /topics
    await userEvent.selectOptions(screen.getByLabelText('Topic'), '1') // option value = topic id (innerHTML would be "Fit &amp; Size")
    await waitFor(() => expect(lastParams(calls).topic_id).toBe('1'))
    await userEvent.click(screen.getByLabelText('Verified only'))
    await waitFor(() => expect(lastParams(calls).verified_only).toBe('true'))
  })

  it("shows each review topic tag as a link to that topic", async () => {
    mockApi(handler([]))
    renderWithProviders(<ReviewsPage />, { route: '/reviews' })
    const tag = await screen.findByRole('link', { name: 'Battery & Charge' })
    expect(tag).toHaveAttribute('href', '/reviews?topic=0')
  })

  it('shows an empty state when nothing matches', async () => {
    mockApi(handler([]))
    renderWithProviders(<ReviewsPage />, { route: '/reviews?q=zzz' })
    expect(await screen.findByText('No reviews match these filters')).toBeInTheDocument()
  })

  it('does not call the reviews API for an impossible rating range', async () => {
    const calls: URL[] = []
    mockApi(handler(calls))
    renderWithProviders(<ReviewsPage />, { route: '/reviews?min=5&max=2' })
    expect(await screen.findByRole('alert')).toHaveTextContent('minimum rating cannot be higher')
    expect(calls).toHaveLength(0)
  })

  it('shows an error state when the API fails', async () => {
    mockApi(handler([], { reviewsStatus: 503 }))
    renderWithProviders(<ReviewsPage />, { route: '/reviews' })
    expect(await screen.findByText('Processed dataset is not available yet.')).toBeInTheDocument()
    expect(screen.getByText('Not available yet')).toBeInTheDocument()
  })
})

describe('MonitoringPage setup steps', () => {
  it('shows the command that builds a missing file, and nothing for files that exist', async () => {
    mockApi(
      (url) =>
        url.pathname.endsWith('/health')
          ? {
              body: {
                ...healthyBody,
                status: 'degraded',
                artifacts_ready: { ...healthyBody.artifacts_ready, nmf_topics: false },
                setup_steps: { nmf_topics: 'python -m reviewpulse.nlp.topic_model', distilbert_model: 'python -m reviewpulse.modeling.train_distilbert' },
              },
            }
          : { status: 503, body: { detail: 'not needed for this test' } },
      { serveHealth: false },
    )
    renderWithProviders(<MonitoringPage />)
    expect(await screen.findByText('python -m reviewpulse.nlp.topic_model')).toBeInTheDocument()
    expect(screen.getByText('python -m reviewpulse.modeling.train_distilbert')).toBeInTheDocument()
    expect(screen.queryByText(/semantic_search/)).not.toBeInTheDocument()
  })
})

describe('lastMonthIsIncomplete', () => {
  it('flags a final month that the data stops in the middle of', () => {
    expect(lastMonthIsIncomplete('2023-09', '2023-09-09')).toBe('2023-09-09')
  })
  it('does not flag a month that is complete, or when the newest review is in another month', () => {
    expect(lastMonthIsIncomplete('2023-09', '2023-09-30')).toBeNull()
    expect(lastMonthIsIncomplete('2024-02', '2024-02-29')).toBeNull() // leap year
    expect(lastMonthIsIncomplete('2023-08', '2023-09-09')).toBeNull()
  })
  it('says nothing when the newest date is unknown', () => {
    expect(lastMonthIsIncomplete('2023-09', null)).toBeNull()
    expect(lastMonthIsIncomplete('2023-09', undefined)).toBeNull()
  })
})
