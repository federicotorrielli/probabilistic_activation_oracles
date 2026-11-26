/**
 * Render scripts/figures/hero/hero.html to paper/figs/hero.pdf and .png.
 *
 * Uses the system Playwright: Linux /usr/lib/node_modules/playwright, or on macOS
 * the playwright-core bundled with `brew install playwright-cli` (browser via
 * `playwright-cli install-browser chromium`).
 */
import { existsSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';

const PLAYWRIGHT_CANDIDATES = [
  '/usr/lib/node_modules/playwright/index.mjs',
  '/opt/homebrew/opt/playwright-cli/libexec/lib/node_modules/@playwright/cli/node_modules/playwright-core/index.mjs',
];
const SYSTEM_CHROMIUM = '/usr/bin/chromium';

const found = PLAYWRIGHT_CANDIDATES.find(existsSync);
if (!found) throw new Error('Playwright not found in: ' + PLAYWRIGHT_CANDIDATES.join(', '));
const { chromium } = await import(pathToFileURL(found).href);

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(__dirname, '..', '..', '..');
const HTML = 'file://' + path.join(__dirname, 'hero.html');
const OUT_PDF = path.join(REPO, 'paper', 'figs', 'hero.pdf');
const OUT_PNG = path.join(REPO, 'paper', 'figs', 'hero.png');

// Linux: reuse the system Chromium. macOS: Playwright's own browser.
const browser = await chromium.launch(
  existsSync(SYSTEM_CHROMIUM) ? { executablePath: SYSTEM_CHROMIUM } : {},
);
const ctx = await browser.newContext({
  viewport: { width: 1538, height: 900 },
  deviceScaleFactor: 2,
});
const page = await ctx.newPage();
await page.goto(HTML, { waitUntil: 'networkidle' });

// Make sure custom fonts (if any) have settled.
await page.evaluate(() => document.fonts && document.fonts.ready);

const root = await page.$('#root');
const box = await root.boundingBox();

// Crisp PNG at 2x.
await root.screenshot({ path: OUT_PNG, omitBackground: false });

// Vector PDF sized to the figure’s actual height.
await page.emulateMedia({ media: 'print' });
await page.pdf({
  path: OUT_PDF,
  printBackground: true,
  width: Math.ceil(box.width) + 'px',
  height: Math.ceil(box.height) + 'px',
  margin: { top: 0, right: 0, bottom: 0, left: 0 },
  preferCSSPageSize: false,
});

console.log('wrote', OUT_PDF);
console.log('wrote', OUT_PNG);
await browser.close();
