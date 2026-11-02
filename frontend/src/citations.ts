export type Segment = { kind: 'text'; value: string } | { kind: 'cite'; n: number } | { kind: 'code'; value: string }

/**
 * Split an answer like "Use `Depends` [1] to share logic [2]." into text, inline-code and
 * citation segments so the UI can render citations as buttons without using innerHTML.
 */
export function parseAnswer(answer: string): Segment[] {
  const out: Segment[] = []
  const re = /\[(\d+)\]|`([^`]+)`/g
  let last = 0
  for (const m of answer.matchAll(re)) {
    const i = m.index ?? 0
    if (i > last) out.push({ kind: 'text', value: answer.slice(last, i) })
    if (m[1] !== undefined) out.push({ kind: 'cite', n: Number(m[1]) })
    else out.push({ kind: 'code', value: m[2] })
    last = i + m[0].length
  }
  if (last < answer.length) out.push({ kind: 'text', value: answer.slice(last) })
  return out
}

/** Citation numbers used in an answer, in first-use order, without duplicates. */
export function citedSources(answer: string): number[] {
  const seen: number[] = []
  for (const s of parseAnswer(answer)) if (s.kind === 'cite' && !seen.includes(s.n)) seen.push(s.n)
  return seen
}
