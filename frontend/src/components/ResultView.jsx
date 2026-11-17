import { useState } from 'react'
import { resultUrl } from '../services/api.js'

const VIEWS = [
  ['combined_outline.png', 'Outline'],
  ['outer_boundary.png', 'Outer boundary'],
  ['internal_design.png', 'Inner lines'],
  ['outline_preview.png', 'Photo vs outline'],
]

// Displays the real files the backend produced for this job.
export default function ResultView({ result }) {
  const available = VIEWS.filter(([name]) => result?.files?.[name])
  const [current, setCurrent] = useState(available[0]?.[0])
  if (!available.length) return null
  const active = available.find(([n]) => n === current) ? current : available[0][0]
  return (
    <section className="result">
      <div className="tabs" role="tablist">
        {available.map(([name, label]) => (
          <button
            key={name}
            type="button"
            role="tab"
            aria-selected={name === active}
            className={`tab${name === active ? ' on' : ''}`}
            onClick={() => setCurrent(name)}
          >
            {label}
          </button>
        ))}
      </div>
      <figure className="result-img">
        <img src={resultUrl(result.files[active])} alt={`Result: ${active}`} />
      </figure>
    </section>
  )
}
