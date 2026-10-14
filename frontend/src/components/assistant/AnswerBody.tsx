import type { ReactNode } from 'react'

/**
 * Renders the assistant's answer text. Answers are Markdown-ish (bold, lists, [Review ID: ...] citations),
 * but they come from an LLM, so this builds React elements itself and never uses innerHTML.
 */

const INLINE = /(\*\*[^*\n]+\*\*|`[^`\n]+`|\[Review ID:[^\]]+\])/g

interface CitationProps {
  /** Ids of the reviews shown as evidence. A citation of any other id is marked, because the model may invent ids. */
  evidenceIds?: ReadonlySet<string>
  /** Called when a citation of a shown review is clicked. */
  onCite?: (reviewId: string) => void
}

function renderCitation(content: string, key: number, { evidenceIds, onCite }: CitationProps): ReactNode {
  const ids = content.replace('Review ID:', '').split(/[,;\s]+/).filter(Boolean)
  return ids.map((id, n) => {
    const short = id.slice(0, 8)
    const base = 'mx-0.5 inline-block rounded border px-1 py-px font-mono text-[0.8em]'
    if (!evidenceIds || !onCite) {
      return (
        <span key={`${key}-${n}`} className={`${base} border-blue/30 bg-blue/10 text-blue`}>
          {short}
        </span>
      )
    }
    if (!evidenceIds.has(id)) {
      return (
        <span key={`${key}-${n}`} className={`${base} border-dashed border-neu/50 bg-neu/10 text-neu`} title="This review id is not among the evidence shown below, so it cannot be checked.">
          {short}?
        </span>
      )
    }
    return (
      <button
        key={`${key}-${n}`}
        type="button"
        onClick={() => onCite(id)}
        className={`${base} cursor-pointer border-blue/30 bg-blue/10 text-blue hover:bg-blue/20 focus-visible:outline-2 focus-visible:outline-brand-2`}
        title="Show the cited review"
      >
        {short}
      </button>
    )
  })
}

function renderInline(text: string, citations: CitationProps = {}): ReactNode[] {
  return text.split(INLINE).map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
      return (
        <strong key={i} className="font-semibold text-ink">
          {part.slice(2, -2)}
        </strong>
      )
    }
    if (part.startsWith('`') && part.endsWith('`') && part.length > 2) {
      return (
        <code key={i} className="rounded bg-card-2 px-1 py-0.5 font-mono text-[0.85em] text-teal">
          {part.slice(1, -1)}
        </code>
      )
    }
    if (part.startsWith('[Review ID:')) {
      return renderCitation(part.slice(1, -1), i, citations)
    }
    return part
  })
}

const LIST_ITEM = /^\s*(?:[-*•]|\d+[.)])\s+(.*)$/

export function AnswerBody({ text, className = '', evidenceIds, onCite }: { text: string; className?: string } & CitationProps) {
  const citations: CitationProps = { evidenceIds, onCite }
  const lines = text.replace(/\r/g, '').split('\n')
  const blocks: ReactNode[] = []
  let paragraph: string[] = []
  let items: { ordered: boolean; text: string }[] = []

  const flushParagraph = () => {
    if (paragraph.length) blocks.push(<p key={blocks.length}>{renderInline(paragraph.join(' '), citations)}</p>)
    paragraph = []
  }
  const flushList = () => {
    if (!items.length) return
    const Tag = items[0]!.ordered ? 'ol' : 'ul'
    blocks.push(
      <Tag key={blocks.length} className={`space-y-1 pl-5 ${Tag === 'ol' ? 'list-decimal' : 'list-disc'} marker:text-muted`}>
        {items.map((item, i) => (
          <li key={i}>{renderInline(item.text, citations)}</li>
        ))}
      </Tag>,
    )
    items = []
  }

  for (const raw of lines) {
    const line = raw.replace(/^#{1,6}\s+/, '')
    const listMatch = LIST_ITEM.exec(raw)
    if (!line.trim()) {
      flushParagraph()
      flushList()
    } else if (listMatch) {
      flushParagraph()
      items.push({ ordered: /^\s*\d/.test(raw), text: listMatch[1] ?? '' })
    } else {
      flushList()
      paragraph.push(line.trim())
    }
  }
  flushParagraph()
  flushList()

  return <div className={`space-y-2 text-[13px] leading-relaxed text-ink-2 ${className}`}>{blocks}</div>
}
