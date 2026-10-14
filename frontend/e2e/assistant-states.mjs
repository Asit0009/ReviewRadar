/**
 * Asks the assistant real questions through the UI and takes full-page screenshots of each answer state.
 *
 *   E2E_BASE_URL=http://127.0.0.1:8000 E2E_SCREENSHOT_DIR=../logs/visual-review/assistant node e2e/assistant-states.mjs
 *
 * Needs a running backend. With a Gemini key configured, "answer" and "insufficient" are real Gemini calls.
 * "provider-error" is only meaningful against a backend whose key is invalid (start one with a made-up
 * GEMINI_API_KEY in its environment): Google then really rejects the call and the page shows that failure.
 *
 *   node e2e/assistant-states.mjs answer insufficient      (default)
 *   node e2e/assistant-states.mjs provider-error
 */
import { existsSync, mkdirSync } from 'node:fs'
import puppeteer from 'puppeteer-core'

const BASE = (process.env.E2E_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')
const OUT = process.env.E2E_SCREENSHOT_DIR || '../logs/visual-review/assistant'
const CHROME =
  process.env.CHROME_PATH ||
  [
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium',
  ].find((p) => existsSync(p))
mkdirSync(OUT, { recursive: true })

const QUESTIONS = [
  { name: 'answer', text: 'What do customers complain about with phone cases?', category: 'Cell_Phones_and_Accessories' },
  { name: 'insufficient', text: 'What is the capital of Australia?', category: null },
  { name: 'provider-error', text: 'Why do some dog owners say their pet refused the food?', category: 'Pet_Supplies' },
]
const WANTED = process.argv.length > 2 ? process.argv.slice(2) : ['answer', 'insufficient']
const WIDTHS = [
  { name: '1536', width: 1536, height: 1024 },
  { name: '375', width: 375, height: 812, isMobile: true, hasTouch: true },
]

const browser = await puppeteer.launch({ executablePath: CHROME, headless: true, args: ['--no-sandbox'] })
for (const vp of WIDTHS) {
  for (const q of QUESTIONS.filter((x) => WANTED.includes(x.name))) {
    const page = await browser.newPage()
    await page.setViewport({ width: vp.width, height: vp.height, isMobile: !!vp.isMobile, hasTouch: !!vp.hasTouch })
    const errors = []
    page.on('pageerror', (e) => errors.push(String(e)))
    await page.goto(`${BASE}/assistant`, { waitUntil: 'networkidle0', timeout: 60000 })
    if (q.category) {
      await page.evaluate((cat) => {
        const sel = [...document.querySelectorAll('main select')].find((s) => [...s.options].some((o) => o.value === cat))
        if (sel) {
          const set = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set
          set.call(sel, cat)
          sel.dispatchEvent(new Event('change', { bubbles: true }))
        }
      }, q.category)
    }
    const input = await page.$('main input[type=text], main textarea, main input:not([type])')
    await input.type(q.text)
    const t0 = Date.now()
    await page.keyboard.press('Enter')
    // "Searched N of M reviews" is printed only under a finished answer, never on the empty page.
    await page.waitForFunction(() => /Searched [\d,]+ of [\d,]+ reviews/.test(document.body.innerText), { timeout: 120000 })
    await new Promise((r) => setTimeout(r, 800))
    const text = await page.$eval('main', (m) => m.innerText)
    const file = `${OUT}/assistant-${q.name}-${vp.name}.png`
    await page.screenshot({ path: file, fullPage: true })
    console.log(`${q.name.padEnd(15)} ${vp.name.padEnd(5)} ${((Date.now() - t0) / 1000).toFixed(1)}s  Written by: ${/Written by ([\w.-]+)/.exec(text)?.[1] ?? '-'} | evidence: ${/Evidence from (\d+)/.exec(text)?.[1] ?? '0'} | errors: ${errors.length}`)
    await page.close()
  }
}
await browser.close()
