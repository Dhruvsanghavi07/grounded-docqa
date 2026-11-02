import { describe, expect, it } from 'vitest'
import { citedSources, parseAnswer } from './citations'

describe('parseAnswer', () => {
  it('splits text, inline code and citations', () => {
    expect(parseAnswer('Use `Depends` [1] here.')).toEqual([
      { kind: 'text', value: 'Use ' },
      { kind: 'code', value: 'Depends' },
      { kind: 'text', value: ' ' },
      { kind: 'cite', n: 1 },
      { kind: 'text', value: ' here.' },
    ])
  })

  it('handles adjacent citations and plain text', () => {
    expect(parseAnswer('a [2][3]')).toEqual([
      { kind: 'text', value: 'a ' },
      { kind: 'cite', n: 2 },
      { kind: 'cite', n: 3 },
    ])
    expect(parseAnswer('no citations')).toEqual([{ kind: 'text', value: 'no citations' }])
  })
})

describe('citedSources', () => {
  it('returns unique citations in first-use order', () => {
    expect(citedSources('x [2]. y [1]. z [2].')).toEqual([2, 1])
  })
})
