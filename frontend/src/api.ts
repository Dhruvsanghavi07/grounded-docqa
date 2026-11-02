export type Mode = 'hybrid' | 'bm25' | 'dense'

export interface Source {
  n: number
  doc: string
  heading: string
  text: string
  score: number
}

export interface Answer {
  question: string
  answer: string
  sources: Source[]
  grounded: boolean
  generator: string
  attempts: number
  fell_back: boolean
  violations: string[]
  retrieval_ms: number
  generation_ms: number
}

const BASE = import.meta.env.VITE_API_URL ?? '/api'

export async function ask(question: string, mode: Mode, k = 5, signal?: AbortSignal): Promise<Answer> {
  const res = await fetch(`${BASE}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, mode, k }),
    signal,
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    const detail = Array.isArray(body.detail) ? body.detail.map((d: { msg: string }) => d.msg).join('; ') : body.detail
    throw new Error(detail || `Request failed with status ${res.status}`)
  }
  return res.json()
}
