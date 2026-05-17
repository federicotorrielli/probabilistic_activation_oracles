/**
 * Render scripts/figures/hero/hero.html to paper/figs/hero.pdf and .png.
 *
 * Uses the global Playwright install ( pnpm add -g @playwright/test ).
 */
import { chromium } from '/home/fedexmachina/.local/share/pnpm/global/5/node_modules/playwright/index.mjs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(__dirname, '..', '..', '..');
const HTML = 'file://' + path.join(__dirname, 'hero.html');
const OUT_PDF = path.join(REPO, 'paper', 'figs', 'hero.pdf');
const OUT_PNG = path.join(REPO, 'paper', 'figs', 'hero.png');

const browser = await chromium.launch({
  // Reuse the system Chromium so we don't need playwright-install.
  executablePath: '/usr/bin/chromium',
});
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
