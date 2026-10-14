/**
 * Review text from the dataset still contains HTML such as "<br />". We never inject it as HTML;
 * this turns it into plain text so React can render it safely.
 */
export function cleanReviewText(raw: string | null | undefined): string {
  if (!raw) return ''
  return raw
    .replace(/<br\s*\/?>/gi, ' ')
    // Only strip things shaped like HTML tags, so reviews that say "5 < 6 and 7 > 3" keep their text.
    .replace(/<\/?[a-zA-Z][^>]*>/g, '')
    .replace(/\s+/g, ' ')
    .trim()
}

export function truncate(text: string, max: number): string {
  if (text.length <= max) return text
  return `${text.slice(0, max).trimEnd()}…`
}

const escapeRegExp = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

/** Splits text around case-insensitive matches of `term` so the UI can wrap matches in <mark>. */
export function splitByTerm(text: string, term: string): Array<{ text: string; match: boolean }> {
  const trimmed = term.trim()
  if (!trimmed) return [{ text, match: false }]
  const parts = text.split(new RegExp(`(${escapeRegExp(trimmed)})`, 'gi'))
  return parts.filter((p) => p !== '').map((p) => ({ text: p, match: p.toLowerCase() === trimmed.toLowerCase() }))
}
