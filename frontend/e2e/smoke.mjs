/**
 * End-to-end browser test of the production app.
 *
 *   1. npm run build                       (in frontend/)
 *   2. uvicorn reviewpulse.api.app:app     (from the repo root; serves the API AND frontend/dist)
 *   3. npm run e2e                         (this file)
 *
 * It drives a real Chrome through the intended entry point (E2E_BASE_URL, default http://127.0.0.1:8000), not the
 * Vite dev server. Set CHROME_PATH if Chrome is not in a standard location, and E2E_SCREENSHOT_DIR to save pictures.
 */
import { existsSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'
import puppeteer from 'puppeteer-core'

const BASE = (process.env.E2E_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')
const SHOTS = process.env.E2E_SCREENSHOT_DIR
const CHROME =
  process.env.CHROME_PATH ||
  [
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  ].find((p) => existsSync(p))
if (!CHROME) {
  console.error('Chrome not found. Set CHROME_PATH.')
  process.exit(2)
}
if (SHOTS) mkdirSync(SHOTS, { recursive: true })

const VIEWPORTS = [
  { name: 'desktop-1536', width: 1536, height: 1024 },
  { name: 'laptop-1280', width: 1280, height: 900 },
  { name: 'tablet-768', width: 768, height: 1024 },
  { name: 'phone-375', width: 375, height: 812, isMobile: true, hasTouch: true },
]
const ROUTES = [
  ['/', 'Amazon Reviews 2023'],
  ['/reviews', 'Reviews Analysis'],
  ['/sentiment', 'Sentiment Analysis'],
  ['/topics', 'Topic Modeling'],
  ['/predictions', 'ML Predictions'],
  ['/deep-learning', 'Deep Learning (NLP)'],
  ['/assistant', 'LLM Assistant'],
  ['/monitoring', 'Deploy & Monitor'],
]

const results = []
async function check(name, fn) {
  const started = Date.now()
  try {
    await fn()
    results.push({ name, ok: true })
    console.log(`  PASS  ${name} (${Date.now() - started} ms)`)
  } catch (err) {
    results.push({ name, ok: false, error: err.message })
    console.log(`  FAIL  ${name}\n        ${err.message.split('\n')[0]}`)
  }
}
const assert = (cond, message) => {
  if (!cond) throw new Error(message)
}
const api = async (path) => {
  const res = await fetch(`${BASE}/api/v1${path}`)
  if (!res.ok) throw new Error(`GET /api/v1${path} returned HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`)
  return res.json()
}

/** A page that records console errors, page errors and failed or error responses. */
async function newPage(browser, viewport) {
  const page = await browser.newPage()
  await page.setViewport({ width: viewport.width, height: viewport.height, isMobile: !!viewport.isMobile, hasTouch: !!viewport.hasTouch })
  page.problems = []
  page.requests = []
  page.on('console', (m) => m.type() === 'error' && page.problems.push(`console: ${m.text()}`))
  page.on('pageerror', (e) => page.problems.push(`pageerror: ${e.message}`))
  // net::ERR_ABORTED is what Chrome reports when the app itself cancels a request that a newer one has replaced
  // (for example when two filters change in quick succession). That is intended behaviour, not a failure.
  page.on('requestfailed', (r) => {
    if (!page.expectFailures && r.failure()?.errorText !== 'net::ERR_ABORTED') page.problems.push(`request failed: ${r.url()} ${r.failure()?.errorText}`)
  })
  page.on('response', (r) => {
    if (r.status() >= 400 && !page.expectFailures) page.problems.push(`HTTP ${r.status()}: ${r.url()}`)
  })
  page.on('request', (r) => page.requests.push({ url: r.url(), method: r.method(), body: r.postData() }))
  return page
}

/** Wait until the page has finished loading data: no skeleton placeholders and no outstanding requests. */
async function settled(page) {
  await page.waitForNetworkIdle({ idleTime: 600, timeout: 30000 }).catch(() => {})
  await page.waitForFunction(() => document.querySelectorAll('main [role=status]').length === 0, { timeout: 30000 })
}
const goto = async (page, path) => {
  await page.goto(`${BASE}${path}`, { waitUntil: 'domcontentloaded' })
  await page.waitForSelector('h1', { timeout: 20000 })
  await settled(page)
}
const shot = async (page, name) => SHOTS && page.screenshot({ path: join(SHOTS, `${name}.png`), fullPage: false })
const noProblems = (page) => assert(page.problems.length === 0, `browser problems: ${page.problems.slice(0, 3).join(' | ')}`)
const noErrorBlocks = async (page) => {
  const text = await page.$eval('main', (m) => m.innerText)
  assert(!/Couldn.t load this data|Can.t reach the API/.test(text), 'an error block is visible')
}
const noHorizontalScroll = async (page) => {
  const { scroll, inner } = await page.evaluate(() => ({ scroll: document.documentElement.scrollWidth, inner: window.innerWidth }))
  assert(scroll <= inner + 1, `horizontal overflow: page is ${scroll}px wide in a ${inner}px window`)
}
const setInput = (page, label, value) =>
  page.evaluate(
    (label, value) => {
      const el = [...document.querySelectorAll('label')].filter((l) => l.textContent.trim() === label).map((l) => document.getElementById(l.htmlFor)).find((e) => e && e.offsetParent !== null)
      if (!el) throw new Error(`no visible input labelled ${label}`)
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, value)
      el.dispatchEvent(new Event('input', { bubbles: true }))
    },
    label,
    value,
  )
const selectByLabel = async (page, label, value) => {
  const id = await page.evaluate((label) => [...document.querySelectorAll('label')].filter((l) => l.textContent.trim() === label).map((l) => l.htmlFor).find((id) => document.getElementById(id)?.offsetParent !== null), label)
  assert(id, `no visible select labelled ${label}`)
  await page.select(`[id="${id}"]`, value)
}
const waitForRequest = (page, predicate, timeout = 15000) =>
  page.waitForRequest((r) => predicate(new URL(r.url()), r), { timeout })

const browser = await puppeteer.launch({ executablePath: CHROME, headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'] })
console.log(`E2E against ${BASE} with ${CHROME}\n`)

try {
  // ---------- the entry point itself ----------
  console.log('Entry point')
  await check('GET / serves the React app and GET /api/v1/health answers', async () => {
    const html = await (await fetch(`${BASE}/`)).text()
    assert(html.includes('<div id="root">') && /assets\/index-.*\.js/.test(html), 'index.html is not the built React app')
    assert((await api('/health')).status === 'healthy', 'backend is not healthy')
  })
  await check('hashed assets are cached for a year; unknown /api paths are JSON 404s; client routes fall back to index.html', async () => {
    const html = await (await fetch(`${BASE}/`)).text()
    const asset = html.match(/\/assets\/index-[^"]+\.js/)[0]
    const res = await fetch(`${BASE}${asset}`)
    assert(res.ok && /immutable/.test(res.headers.get('cache-control') || ''), `asset cache header: ${res.headers.get('cache-control')}`)
    const missing = await fetch(`${BASE}/api/v1/nope`)
    assert(missing.status === 404 && (missing.headers.get('content-type') || '').includes('json'), 'unknown API path is not a JSON 404')
    assert((await (await fetch(`${BASE}/reviews`)).text()) === html, '/reviews does not fall back to index.html')
  })

  // ---------- layouts, console and network ----------
  for (const vp of VIEWPORTS) {
    console.log(`\nLayout: ${vp.name}`)
    const page = await newPage(browser, vp)
    await check(`overview loads with real data, no console/network errors, no horizontal scroll (${vp.name})`, async () => {
      await goto(page, '/')
      const overview = await api('/overview')
      const text = await page.$eval('main', (m) => m.innerText)
      assert(text.includes(overview.total_reviews.toLocaleString('en-US')), `total reviews ${overview.total_reviews} not shown`)
      assert(text.includes(`${overview.average_rating.toFixed(2)} / 5`), 'average rating not shown')
      await noErrorBlocks(page)
      await noHorizontalScroll(page)
      const charts = await page.$$eval('.recharts-wrapper', (els) => els.map((e) => [e.clientWidth, e.clientHeight]))
      assert(charts.length >= 3 && charts.every(([w, h]) => w > 0 && h > 0), `charts did not render: ${JSON.stringify(charts)}`)
      await shot(page, `overview-${vp.name}`)
      noProblems(page)
    })
    await check(`every page deep-links, renders and fits the screen (${vp.name})`, async () => {
      for (const [path, heading] of ROUTES) {
        await goto(page, path)
        const h1 = await page.$eval('h1', (e) => e.textContent)
        assert(h1.includes(heading), `${path}: heading "${h1}" (expected "${heading}")`)
        await noErrorBlocks(page).catch((e) => Promise.reject(new Error(`${path}: ${e.message}`)))
        await noHorizontalScroll(page).catch((e) => Promise.reject(new Error(`${path}: ${e.message}`)))
        await shot(page, `${path === '/' ? 'overview' : path.slice(1)}-${vp.name}`)
      }
      noProblems(page)
    })
    if (vp.width < 1024) {
      await check(`menu and assistant drawers open and close (${vp.name})`, async () => {
        await goto(page, '/')
        await page.click('button[aria-label="Open menu"]')
        await page.waitForSelector('aside a[href="/reviews"]', { visible: true, timeout: 5000 })
        await page.keyboard.press('Escape')
        await page.waitForSelector('aside a[href="/reviews"]', { hidden: true, timeout: 5000 })
        const open = [...(await page.$$('header button'))]
        let assistantButton
        for (const b of open) if ((await b.evaluate((e) => e.textContent)).includes('Assistant') || (await b.evaluate((e) => e.querySelector('svg.lucide-sparkles') !== null))) assistantButton = b
        assert(assistantButton, 'no Assistant button in the header')
        await assistantButton.click()
        await page.waitForSelector('section[aria-label="Review assistant"]', { visible: true, timeout: 5000 })
        await page.keyboard.press('Escape')
        await page.waitForSelector('section[aria-label="Review assistant"]', { hidden: true, timeout: 5000 })
      })
    }
    await page.close()
  }

  // ---------- behaviour on desktop ----------
  const desktop = VIEWPORTS[0]
  console.log('\nFilters')
  {
    const page = await newPage(browser, desktop)
    await goto(page, '/')
    const categories = (await api('/categories')).categories
    const pick = categories.find((c) => c.category === 'Electronics') ?? categories[0]

    await check('choosing a category sends category= to the API and the KPIs follow the backend', async () => {
      const waiting = waitForRequest(page, (u) => u.pathname.endsWith('/overview') && u.searchParams.get('category') === pick.category)
      await selectByLabel(page, 'Category', pick.category)
      await waiting
      await settled(page)
      const expected = await api(`/overview?category=${pick.category}`)
      const text = await page.$eval('main', (m) => m.innerText)
      assert(text.includes(expected.total_reviews.toLocaleString('en-US')) && text.includes(`${expected.average_rating.toFixed(2)} / 5`), 'KPIs do not match /overview for the category')
      assert((await page.evaluate(() => document.body.innerText)).includes(`${expected.total_reviews.toLocaleString('en-US')} reviews analyzed`), 'filter card count did not update')
      await shot(page, 'filtered-by-category')
    })
    await check('every one of the ten categories plus "All Categories" can be selected and returns its own count', async () => {
      const options = await page.$$eval('select option', (os) => os.map((o) => o.value))
      for (const c of categories) assert(options.includes(c.category), `missing option ${c.category}`)
      assert(categories.length === 10, `expected 10 categories, API has ${categories.length}`)
      for (const c of categories) {
        const waiting = waitForRequest(page, (u) => u.pathname.endsWith('/overview') && u.searchParams.get('category') === c.category)
        await selectByLabel(page, 'Category', c.category)
        await waiting.catch(() => {}) // already cached requests are not repeated
        await settled(page)
        const text = await page.$eval('main', (m) => m.innerText)
        assert(text.includes(c.review_count.toLocaleString('en-US')), `${c.category}: ${c.review_count} not shown`)
      }
      await selectByLabel(page, 'Category', '__all__')
      await settled(page)
    })
    await check('date range is applied by the backend (date_from / date_to in the request) and the numbers match the API', async () => {
      const waiting = waitForRequest(page, (u) => u.pathname.endsWith('/overview') && u.searchParams.get('date_from') === '2022-01-01' && u.searchParams.get('date_to') === '2022-12-31')
      await setInput(page, 'From', '2022-01-01')
      await setInput(page, 'To', '2022-12-31')
      await waiting
      await settled(page)
      const expected = await api('/overview?date_from=2022-01-01&date_to=2022-12-31')
      const text = await page.$eval('main', (m) => m.innerText)
      assert(expected.total_reviews > 0 && text.includes(expected.total_reviews.toLocaleString('en-US')), 'KPI total does not match the dated /overview')
      const trendCalls = page.requests.filter((r) => r.url.includes('/sentiment/trend') && r.url.includes('date_from=2022-01-01'))
      assert(trendCalls.length > 0, 'the trend chart request did not carry the date filter')
      await shot(page, 'filtered-by-date')
    })
    await check('a reversed date range is explained and not sent to the API', async () => {
      await setInput(page, 'From', '2023-01-01')
      await setInput(page, 'To', '2022-01-01')
      await page.waitForFunction(() => /start date must be on or before/i.test(document.body.innerText), { timeout: 5000 })
      await page.evaluate(() => [...document.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Clear dates')?.click())
      await settled(page)
    })
    noProblems(page)
    await page.close()
  }

  console.log('\nReviews')
  {
    const page = await newPage(browser, desktop)
    await check('typing a search term filters via the API, highlights matches and updates the URL', async () => {
      await goto(page, '/reviews')
      const waiting = waitForRequest(page, (u) => u.pathname.endsWith('/reviews') && u.searchParams.get('search') === 'refund')
      await page.type('input[type=search][placeholder*="battery"]', 'refund')
      await waiting
      await settled(page)
      const expected = await api('/reviews?search=refund&limit=20')
      const text = await page.$eval('main', (m) => m.innerText)
      assert(page.url().includes('q=refund'), 'URL does not contain q=refund')
      assert(expected.total_records > 0 && text.includes(`${expected.total_records.toLocaleString('en-US')} reviews match`), `count text does not show ${expected.total_records}`)
      assert((await page.$$('mark')).length > 0, 'matches are not highlighted')
    })
    await check('pagination: Next loads page 2 from the API and Page X of Y is right', async () => {
      const expected = await api('/reviews?search=refund&limit=20')
      assert(expected.total_pages >= 2, 'need at least two pages of results for this check')
      const waiting = waitForRequest(page, (u) => u.pathname.endsWith('/reviews') && u.searchParams.get('page') === '2')
      await page.evaluate(() => [...document.querySelectorAll('nav[aria-label=Pagination] button')].find((b) => b.textContent.includes('Next')).click())
      await waiting
      await settled(page)
      const pager = await page.$eval('nav[aria-label=Pagination] p', (p) => p.textContent)
      assert(pager.includes(`Page 2 of ${expected.total_pages}`), `pager says "${pager}"`)
      const first = await page.$eval('article h3', (h) => h.textContent)
      const second = await api('/reviews?search=refund&limit=20&page=2')
      assert(second.reviews[0].title.replace(/\s+/g, ' ').trim() === first.replace(/\s+/g, ' ').trim(), 'first review on page 2 is not the API\'s')
    })
    await check('sort order and sentiment filter reach the API', async () => {
      await goto(page, '/reviews')
      await selectByLabel(page, 'Sentiment', 'negative')
      await selectByLabel(page, 'Sort by', 'rating-asc')
      await settled(page)
      const call = page.requests.filter((r) => r.url.includes('/reviews?') && r.url.includes('sentiment=negative') && r.url.includes('sort=rating') && r.url.includes('order=asc'))
      assert(call.length > 0, 'no request carried sentiment=negative&sort=rating&order=asc')
      const stars = await page.$$eval('article [role=img][aria-label$="stars"]', (els) => els.map((e) => e.getAttribute('aria-label')))
      assert(stars.length > 0 && stars.every((s) => /^[12] out of 5 stars$/.test(s)), `negative reviews should be 1-2 stars: ${stars.slice(0, 5)}`)
    })
    await check('topic tags are shown and clicking one filters by topic', async () => {
      await goto(page, '/reviews')
      const tag = await page.$('article a[href^="/reviews?topic="]')
      assert(tag, 'no topic tag on any review')
      const href = await tag.evaluate((a) => a.getAttribute('href'))
      await tag.click()
      await page.waitForFunction((h) => location.pathname + location.search === h, { timeout: 10000 }, href)
      await settled(page)
      const id = new URL(href, BASE).searchParams.get('topic')
      assert(page.requests.some((r) => r.url.includes('/reviews?') && r.url.includes(`topic_id=${id}`)), 'topic_id was not sent')
    })
    await check('a search with no matches shows the empty state', async () => {
      await goto(page, '/reviews?q=zzqxjkvv')
      await page.waitForFunction(() => document.body.innerText.includes('No reviews match these filters'), { timeout: 10000 })
    })
    noProblems(page)
    await page.close()
  }

  console.log('\nPrediction and models')
  {
    const page = await newPage(browser, desktop)
    await goto(page, '/predictions')
    await check('empty prediction is rejected by the form without calling the API', async () => {
      await page.evaluate(() => [...document.querySelectorAll('button')].find((b) => b.textContent.includes('Predict sentiment')).click())
      await page.waitForFunction(() => /at least 2 characters/i.test(document.body.innerText), { timeout: 5000 })
      assert(!page.requests.some((r) => r.url.includes('/api/v1/predict')), 'the API was called with empty text')
    })
    await check('predicting a clearly negative review returns the real model result', async () => {
      await page.type('textarea', 'The blender stopped working after three uses and customer support never answered.')
      await page.evaluate(() => [...document.querySelectorAll('button')].find((b) => b.textContent.includes('Predict sentiment')).click())
      await page.waitForFunction(() => document.body.innerText.includes('Predicted sentiment'), { timeout: 15000 })
      const call = page.requests.find((r) => r.url.endsWith('/api/v1/predict'))
      assert(call && JSON.parse(call.body).model_type === 'logistic_regression', 'wrong /predict request')
      const text = await page.$eval('main', (m) => m.innerText)
      assert(/Negative/.test(text) && /\d+\.\d%/.test(text), 'no negative label / probabilities shown')
      await shot(page, 'prediction-result')
    })
    await check('DistilBERT is selectable exactly when it is trained, and then returns its own prediction', async () => {
      const health = await api('/health')
      const trained = health.artifacts_ready.distilbert_model === true
      const p = await newPage(browser, desktop)
      try {
        await goto(p, '/predictions')
        const disabled = await p.$eval('input[type=radio][value=distilbert]', (r) => r.disabled)
        assert(disabled === !trained, `DistilBERT radio disabled=${disabled} but trained=${trained}`)
        if (!trained) return
        await p.evaluate(() => document.querySelector('input[type=radio][value=distilbert]').click())
        await p.type('textarea', 'Love this, works perfectly and arrived early. Would buy again.')
        await p.evaluate(() => [...document.querySelectorAll('button')].find((b) => b.textContent.includes('Predict sentiment')).click())
        await p.waitForFunction(() => document.body.innerText.includes('Predicted sentiment'), { timeout: 60000 })
        const call = p.requests.find((r) => r.url.endsWith('/api/v1/predict'))
        assert(call && JSON.parse(call.body).model_type === 'distilbert', 'the request did not ask for distilbert')
        const text = await p.$eval('main', (m) => m.innerText)
        assert(/DistilBERT/.test(text) && /Positive/.test(text), 'result does not show a positive DistilBERT prediction')
        await shot(p, 'prediction-distilbert')
      } finally {
        await p.close()
      }
    })
    await check('the model table shows exactly the saved test metrics from /models/metrics', async () => {
      const metrics = await api('/models/metrics')
      const lr = metrics.metrics.models.logistic_regression.test
      const row = await page.$$eval('table tr', (rows) => rows.map((r) => r.innerText.replace(/\s+/g, ' ')).find((t) => t.includes('Logistic Regression')))
      assert(row && row.includes(`${(lr.accuracy * 100).toFixed(1)}%`) && row.includes(lr.macro_f1.toFixed(3)), `table row "${row}" does not match accuracy ${lr.accuracy} / macro F1 ${lr.macro_f1}`)
      // Only the comparison table counts here: the page text mentions DistilBERT in the form and descriptions regardless.
      const hasDistilbert = !!metrics.metrics.models.distilbert
      const rows = await page.$$eval('table tr', (trs) => trs.map((r) => r.innerText.replace(/\s+/g, ' ')))
      assert(rows.some((r) => r.startsWith('DistilBERT')) === hasDistilbert, `DistilBERT row in the table (${rows.some((r) => r.startsWith('DistilBERT'))}) must match metrics.json (${hasDistilbert})`)
      if (hasDistilbert) {
        const d = metrics.metrics.models.distilbert.test
        const row = rows.find((r) => r.startsWith('DistilBERT'))
        assert(row.includes(`${(d.accuracy * 100).toFixed(1)}%`) && row.includes(d.macro_f1.toFixed(3)), `DistilBERT row "${row}" does not match its saved metrics`)
      }
    })
    await check('results by category list every category', async () => {
      const table = await api('/models/metrics/by-category')
      const text = await page.$eval('main', (m) => m.innerText)
      for (const c of table.categories) assert(text.includes(c.replace(/_/g, ' ')), `${c} missing from the per-category table`)
    })
    noProblems(page)
    await page.close()
  }

  console.log('\nAssistant')
  {
    const page = await newPage(browser, desktop)
    await goto(page, '/')
    const health = await api('/health')
    await check('a suggested question sends a real request and the answer states honestly whether Gemini wrote it', async () => {
      await page.click('section[aria-label="Review assistant"] ul[aria-label="Suggested questions"] button')
      await page.waitForFunction(() => /Evidence from \d+ review/.test(document.body.innerText), { timeout: 120000 })
      const call = page.requests.find((r) => r.url.endsWith('/assistant/chat'))
      assert(call && JSON.parse(call.body).question.length > 5, 'no /assistant/chat request')
      const text = await page.$eval('section[aria-label="Review assistant"]', (s) => s.innerText)
      if (health.assistant.llm_configured) assert(/Written by/.test(text) || /No written answer/.test(text), 'neither an LLM answer nor an honest fallback is shown')
      else assert(/No written answer: no Gemini API key is configured/.test(text) && !/Written by/.test(text), 'retrieval-only answer is not labelled as such')
      assert(/Searched [\d,]+ of [\d,]+ reviews/.test(text), 'search coverage is not shown')
      await shot(page, 'assistant-answer')
    })
    await check('a failed assistant request shows an error with a retry that works', async () => {
      const p2 = await newPage(browser, desktop)
      await goto(p2, '/assistant')
      await p2.setRequestInterception(true)
      let failNext = true
      p2.on('request', (r) => {
        if (r.url().endsWith('/assistant/chat') && failNext) {
          failNext = false
          r.respond({ status: 503, contentType: 'application/json', body: JSON.stringify({ detail: 'Assistant service is not initialized.' }) })
        } else r.continue()
      })
      p2.expectFailures = true
      await p2.type('input[placeholder*="Ask"]', 'Any complaints about delivery?')
      await p2.keyboard.press('Enter')
      await p2.waitForSelector('[role=alert]', { timeout: 15000 })
      assert((await p2.$eval('[role=alert]', (e) => e.textContent)).includes('Assistant service is not initialized.'), 'error text missing')
      await p2.evaluate(() => [...document.querySelectorAll('button')].find((b) => b.textContent.includes('Try again')).click())
      await p2.waitForFunction(() => /Evidence from \d+ review/.test(document.body.innerText), { timeout: 60000 })
      await p2.close()
    })
    await check('semantic search on Deep Learning reports index coverage and returns scoped results', async () => {
      await goto(page, '/deep-learning')
      const text0 = await page.$eval('main', (m) => m.innerText)
      assert(text0.includes(`${health.retrieval.indexed_reviews.toLocaleString('en-US')} of ${health.retrieval.eligible_reviews.toLocaleString('en-US')} reviews`), 'coverage text does not match /health')
      await page.type('input[placeholder*="stopped working"]', 'battery drains quickly')
      await page.evaluate(() => [...document.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Search').click())
      await page.waitForFunction(() => /reviews? found/.test(document.body.innerText), { timeout: 60000 })
    })
    noProblems(page)
    await page.close()
  }

  console.log('\nFailure behaviour')
  {
    await check('with the API unreachable every panel explains it, shows retry and the header says offline', async () => {
      const page = await newPage(browser, desktop)
      page.expectFailures = true
      await page.setRequestInterception(true)
      page.on('request', (r) => (r.url().includes('/api/v1/') ? r.abort('connectionrefused') : r.continue()))
      await page.goto(`${BASE}/`, { waitUntil: 'domcontentloaded' })
      await page.waitForFunction(() => document.body.innerText.includes("Can't reach the API"), { timeout: 30000 })
      const text = await page.evaluate(() => document.body.innerText)
      assert(/API offline/.test(text), 'header does not say the API is offline')
      assert((await page.$$eval('button', (bs) => bs.filter((b) => b.textContent.includes('Try again')).length)) >= 3, 'retry buttons missing')
      await shot(page, 'api-down')
      await page.close()
    })
  }
} finally {
  await browser.close()
}

const failed = results.filter((r) => !r.ok)
console.log(`\n${results.length - failed.length} of ${results.length} end-to-end checks passed.`)
if (failed.length) {
  console.log('Failures:\n' + failed.map((f) => ` - ${f.name}: ${f.error.split('\n')[0]}`).join('\n'))
  process.exit(1)
}
