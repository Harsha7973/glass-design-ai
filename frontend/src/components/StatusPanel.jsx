// Shows only real backend state: the step messages the pipeline actually reported.
// No fake progress bars or percentages.
const STAGE_LABEL = {
  uploaded: 'Uploaded',
  queued: 'Waiting in line',
  analyzing: 'Finding the design',
  segmenting: 'Isolating the design',
  extracting: 'Extracting lines',
  evaluating: 'Checking quality',
  done: 'Finished',
}

export default function StatusPanel({ phase, job, error }) {
  if (phase === 'uploading') {
    return (
      <div className="status busy" role="status">
        <span className="spinner" aria-hidden="true" />
        Uploading image...
      </div>
    )
  }
  if (phase === 'error') {
    return (
      <div className="status error" role="alert">
        {error}
      </div>
    )
  }
  if (!job) return null

  const running = phase === 'processing'
  const failed = job.status === 'failed'
  const review = job.status === 'needs_review'
  const cls = running ? 'busy' : failed ? 'error' : review ? 'warn' : 'ok'
  const events = (job.events || []).slice(-8)

  return (
    <div className={`status ${cls}`} role={failed ? 'alert' : 'status'}>
      <div className="status-head">
        {running && <span className="spinner" aria-hidden="true" />}
        <strong>
          {running
            ? `${STAGE_LABEL[job.stage] || 'Working'}...`
            : failed
              ? 'No usable outline was produced'
              : review
                ? 'Outline ready, please review it'
                : 'Outline ready'}
        </strong>
      </div>
      <p className="msg">{job.message}</p>

      {job.warnings?.length > 0 && (
        <ul className="warnings">
          {job.warnings.map((w) => (
            <li key={w}>{w}</li>
          ))}
        </ul>
      )}

      <ol className="events">
        {events.map((e, i) => (
          <li key={`${e.at}-${i}`}>{e.message}</li>
        ))}
      </ol>

      <dl>
        <dt>Job ID</dt>
        <dd>{job.job_id}</dd>
        <dt>Image</dt>
        <dd>
          {job.image_format} · {job.width} × {job.height}px
        </dd>
        {job.result && (
          <>
            <dt>Quality score</dt>
            <dd>
              {job.result.score.toFixed(2)} · {job.result.attempts} attempt(s)
            </dd>
          </>
        )}
      </dl>
      {job.result?.problems?.length > 0 && (
        <p className="note">Issues found: {job.result.problems.join('; ')}.</p>
      )}
      <p className="note">PNG and SVG downloads arrive in the next phase.</p>
    </div>
  )
}
