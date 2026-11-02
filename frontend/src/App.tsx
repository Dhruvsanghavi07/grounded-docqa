import { useRef, useState } from 'react'
import { ask, type Answer, type Mode } from './api'
import { citedSources, parseAnswer } from './citations'

const EXAMPLES = [
  'How do I return a 404 error when an item is not found?',
  'How do I hash passwords and create JWT access tokens?',
  'How do I accept several uploaded files in one request?',
]

export default function App() {
  const [question, setQuestion] = useState('')
  const [mode, setMode] = useState<Mode>('hybrid')
  const [answer, setAnswer] = useState<Answer | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [active, setActive] = useState<number | null>(null)
  const pending = useRef<AbortController | null>(null)

  async function submit(q = question) {
    if (q.trim().length < 3) return
    pending.current?.abort()
    const ctrl = new AbortController()
    pending.current = ctrl
    setLoading(true)
    setError(null)
    setActive(null)
    try {
      setAnswer(await ask(q.trim(), mode, 5, ctrl.signal))
    } catch (e) {
      if ((e as Error).name !== 'AbortError') setError((e as Error).message)
    } finally {
      if (pending.current === ctrl) setLoading(false)
    }
  }

  const cited = answer ? citedSources(answer.answer) : []

  return (
    <main className="wrap">
      <header>
        <h1>Grounded DocQA</h1>
        <p className="muted">Ask about the FastAPI docs. Every sentence in an answer is checked against the passage it cites.</p>
      </header>

      <form
        className="ask"
        onSubmit={(e) => {
          e.preventDefault()
          submit()
        }}
      >
        <label htmlFor="q" className="sr-only">Question</label>
        <input
          id="q"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="How do I declare an optional query parameter?"
          maxLength={500}
          autoFocus
        />
        <select aria-label="Retrieval mode" value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
          <option value="hybrid">Hybrid</option>
          <option value="bm25">BM25</option>
          <option value="dense">Dense</option>
        </select>
        <button type="submit" disabled={loading || question.trim().length < 3}>
          {loading ? 'Searching…' : 'Ask'}
        </button>
      </form>

      {!answer && !error && (
        <div className="examples">
          {EXAMPLES.map((q) => (
            <button key={q} className="chip" onClick={() => { setQuestion(q); submit(q) }}>{q}</button>
          ))}
        </div>
      )}

      {error && <p role="alert" className="error">{error}</p>}

      {answer && (
        <section className="result" aria-live="polite">
          <div className="badges">
            <span className={answer.grounded ? 'badge ok' : 'badge warn'}>
              {answer.grounded ? 'All claims cited' : 'Unverified'}
            </span>
            <span className="badge">{answer.generator}</span>
            {answer.fell_back && <span className="badge warn" title={answer.violations.join('\n')}>LLM answer rejected, showing quoted fallback</span>}
            <span className="badge mono">retrieval {answer.retrieval_ms.toFixed(0)} ms</span>
          </div>

          <p className="answer">
            {parseAnswer(answer.answer).map((s, i) =>
              s.kind === 'text' ? <span key={i}>{s.value}</span>
              : s.kind === 'code' ? <code key={i}>{s.value}</code>
              : (
                <button key={i} className={`cite${active === s.n ? ' on' : ''}`} onClick={() => setActive(s.n)}
                  aria-label={`Show source ${s.n}`}>{s.n}</button>
              ),
            )}
          </p>

          <h2>Sources</h2>
          <ol className="sources">
            {answer.sources.map((src) => (
              <li key={src.n} className={`${active === src.n ? 'on' : ''} ${cited.includes(src.n) ? 'cited' : ''}`}>
                <button className="srchead" onClick={() => setActive(active === src.n ? null : src.n)}>
                  <span className="num">{src.n}</span>
                  <span>{src.heading}</span>
                  <span className="muted mono">{src.doc}</span>
                </button>
                {active === src.n && <pre className="passage">{src.text}</pre>}
              </li>
            ))}
          </ol>
        </section>
      )}
    </main>
  )
}
