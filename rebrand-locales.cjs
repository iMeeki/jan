#!/usr/bin/env node
// rebrand-locales.cjs
// Replace the literal brand word "Jan" with "MixGPT" inside every
// locale JSON file's VALUES (never touches keys), across ALL languages.

const fs = require('fs')
const path = require('path')

const LOCALES_DIR = path.join(process.cwd(), 'web-app', 'src', 'locales')

if (!fs.existsSync(LOCALES_DIR)) {
  console.error(`Could not find ${LOCALES_DIR}. Run this from the repo root.`)
  process.exit(1)
}

// Matches "Jan" as a whole word only (won't touch "January", "Janice", etc.)
const JAN_WORD = /\bJan\b/g

function walk(dir, files) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) walk(full, files)
    else if (entry.name.endsWith('.json')) files.push(full)
  }
  return files
}

function replaceInValues(obj) {
  if (typeof obj === 'string') {
    return obj.replace(JAN_WORD, 'MixGPT')
  }
  if (Array.isArray(obj)) {
    return obj.map(replaceInValues)
  }
  if (obj && typeof obj === 'object') {
    const result = {}
    for (const [key, value] of Object.entries(obj)) {
      result[key] = replaceInValues(value) // key itself untouched
    }
    return result
  }
  return obj
}

const files = walk(LOCALES_DIR, [])
let changedCount = 0

for (const file of files) {
  const original = fs.readFileSync(file, 'utf8')
  let json
  try {
    json = JSON.parse(original)
  } catch (e) {
    console.error(`Skipping (invalid JSON): ${file}`)
    continue
  }

  const updated = replaceInValues(json)
  const updatedStr = JSON.stringify(updated, null, 2) + '\n'

  if (updatedStr.replace(/\s/g, '') !== original.replace(/\s/g, '')) {
    fs.writeFileSync(file, updatedStr, 'utf8')
    changedCount++
    console.log(`Updated: ${path.relative(process.cwd(), file)}`)
  }
}

console.log(`\nDone. Updated ${changedCount} locale files.`)