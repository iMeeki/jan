#!/usr/bin/env node
/**
 * MixGPT RTL migration script.
 *
 * Converts physical Tailwind spacing/position classes (ml-, mr-, pl-, pr-,
 * left-, right-, rounded-l/r, border-l/r) to their logical RTL-aware
 * equivalents (ms-, me-, ps-, pe-, start-, end-, rounded-s/e, border-s/e)
 * across web-app/src, while skipping Radix animation utility classes
 * (slide-in-from-left/right, slide-out-to-left/right) which must stay
 * literal because they describe viewport-relative motion, not text
 * direction.
 *
 * Run from the repo root:
 *   node rtl-migrate.js
 */

const fs = require('fs')
const path = require('path')

const ROOT = path.join(process.cwd(), 'web-app', 'src')

if (!fs.existsSync(ROOT)) {
  console.error(
    `Could not find ${ROOT}. Run this script from the repo root (the folder containing "web-app").`
  )
  process.exit(1)
}

// Order matters: protect animate utilities first with placeholders,
// run the general replacements, then restore them.
const ANIMATE_GUARD = [
  [/-from-left-/g, '\u0000FROM_LEFT\u0000'],
  [/-from-right-/g, '\u0000FROM_RIGHT\u0000'],
  [/-to-left-/g, '\u0000TO_LEFT\u0000'],
  [/-to-right-/g, '\u0000TO_RIGHT\u0000'],
]

const REPLACEMENTS = [
  [/\bml-/g, 'ms-'],
  [/\bmr-/g, 'me-'],
  [/\bpl-/g, 'ps-'],
  [/\bpr-/g, 'pe-'],
  [/\bleft-/g, 'start-'],
  [/\bright-/g, 'end-'],
  [/\brounded-l-/g, 'rounded-s-'],
  [/\brounded-r-/g, 'rounded-e-'],
  [/\brounded-l\b/g, 'rounded-s'],
  [/\brounded-r\b/g, 'rounded-e'],
  [/\bborder-l-/g, 'border-s-'],
  [/\bborder-r-/g, 'border-e-'],
  [/\bborder-l\b/g, 'border-s'],
  [/\bborder-r\b/g, 'border-e'],
]

const RESTORE_ANIMATE = [
  [/\u0000FROM_LEFT\u0000/g, '-from-left-'],
  [/\u0000FROM_RIGHT\u0000/g, '-from-right-'],
  [/\u0000TO_LEFT\u0000/g, '-to-left-'],
  [/\u0000TO_RIGHT\u0000/g, '-to-right-'],
]

const SKIP_DIRS = new Set(['__tests__', 'locales', 'node_modules'])

function walk(dir, files) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.isDirectory()) {
      if (SKIP_DIRS.has(entry.name)) continue
      walk(path.join(dir, entry.name), files)
    } else if (/\.(tsx?|jsx?)$/.test(entry.name)) {
      files.push(path.join(dir, entry.name))
    }
  }
  return files
}

const files = walk(ROOT, [])
let changedCount = 0
let totalSubs = 0
const topChanged = []

for (const file of files) {
  const original = fs.readFileSync(file, 'utf8')
  let content = original
  let subs = 0

  for (const [pattern, placeholder] of ANIMATE_GUARD) {
    content = content.replace(pattern, placeholder)
  }
  for (const [pattern, repl] of REPLACEMENTS) {
    content = content.replace(pattern, (m) => {
      subs++
      return repl
    })
  }
  for (const [pattern, repl] of RESTORE_ANIMATE) {
    content = content.replace(pattern, repl)
  }

  if (content !== original) {
    fs.writeFileSync(file, content, 'utf8')
    changedCount++
    totalSubs += subs
    topChanged.push([path.relative(process.cwd(), file), subs])
  }
}

// Bonus: fix the hardcoded "Jan" brand text in the sidebar header, which
// doesn't go through the i18n system.
const sidebarFile = path.join(ROOT, 'components', 'left-sidebar', 'index.tsx')
if (fs.existsSync(sidebarFile)) {
  const before = fs.readFileSync(sidebarFile, 'utf8')
  const after = before.replace(
    /(<span className="ms-2 font-medium font-studio">)Jan(<\/span>)/,
    '$1MixGPT$2'
  )
  if (after !== before) {
    fs.writeFileSync(sidebarFile, after, 'utf8')
    console.log('Fixed hardcoded "Jan" brand text in left-sidebar/index.tsx')
  }
}

topChanged.sort((a, b) => b[1] - a[1])
console.log(`\nChanged ${changedCount} files, ${totalSubs} total class substitutions.\n`)
console.log('Top changed files:')
for (const [f, n] of topChanged.slice(0, 15)) {
  console.log(`  ${String(n).padStart(3)}  ${f}`)
}
console.log('\nDone. Review with "git diff", then test with "yarn dev".')
