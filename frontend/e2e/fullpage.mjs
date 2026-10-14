/**
 * Full-page screenshots of every page at the four review widths, for visual review.
 *
 *   E2E_BASE_URL=http://127.0.0.1:8000 E2E_SCREENSHOT_DIR=../logs/visual-review/full node e2e/fullpage.mjs [route ...]
 *
 * The page is scrolled to the bottom first so lazy content and charts are drawn before the picture is taken.
 * It also lists console errors, failed requests and horizontal overflow per page.
 */
import { existsSync, mkdirSync } from 'node:fs'
import puppeteer from 'puppeteer-core'

const BASE = (process.env.E2E_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')
const OUT = process.env.E2E_SCREENSHOT_DIR || '../logs/visual-review/full'
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
if (!CHROME) throw new Error('Chrome not found. Set CHROME_PATH.')
mkdirSync(OUT, { recursive: true })

const WIDTHS = [
  { name: '1536', width: 1536, height: 1024 },
  { name: '1280', width: 1280, height: 900 },
  { name: '768', width: 768, height: 1024 },
  { name: '375', width: 375, height: 812, isMobile: true, hasTouch: true },
]
const ROUTES = process.argv.length > 2 ? process.argv.slice(2) : ['/sentiment', '/topics', '/deep-learning', '/assistant', '/monitoring']

const browser = await puppeteer.launch({ executablePath: CHROME, headless: true, args: ['--no-sandbox'] })
let problems = 0
for (const route of ROUTES) {
  for (const vp of WIDTHS) {
    const page = await browser.newPage()
    await page.setViewport({ width: vp.width, height: vp.height, isMobile: !!vp.isMobile, hasTouch: !!vp.hasTouch, deviceScaleFactor: 1 })
    const consoleErrors = []
    const failed = []
    page.on('console', (m) => m.type() === 'error' && consoleErrors.push(m.text()))
    page.on('pageerror', (e) => consoleErrors.push(String(e)))
    page.on('response', (r) => r.status() >= 400 && failed.push(`${r.status()} ${r.url().replace(BASE, '')}`))
    await page.goto(`${BASE}${route}`, { waitUntil: 'networkidle0', timeout: 60000 })
    await new Promise((r) => setTimeout(r, 1200))
    await page.evaluate(async () => {
      for (let y = 0; y < document.body.scrollHeight; y += 600) {
        window.scrollTo(0, y)
        await new Promise((r) => setTimeout(r, 120))
      }
      window.scrollTo(0, 0)
    })
    await new Promise((r) => setTimeout(r, 600))
    const overflow = await page.evaluate(() => ({ scroll: document.documentElement.scrollWidth, client: document.documentElement.clientWidth }))
    const name = `${route.replace(/\//g, '') || 'overview'}-${vp.name}.png`
    await page.screenshot({ path: `${OUT}/${name}`, fullPage: true })
    const issues = []
    if (overflow.scroll > overflow.client + 1) issues.push(`horizontal overflow ${overflow.scroll}>${overflow.client}`)
    if (consoleErrors.length) issues.push(`console errors: ${consoleErrors.slice(0, 2).join(' | ')}`)
    if (failed.length) issues.push(`failed requests: ${failed.slice(0, 3).join(', ')}`)
    problems += issues.length
    console.log(`${name.padEnd(30)} ${issues.length ? 'ISSUES: ' + issues.join('; ') : 'ok'}`)
    await page.close()
  }
}
await browser.close()
console.log(problems ? `\n${problems} issue(s) found` : '\nno console errors, failed requests or overflow')
