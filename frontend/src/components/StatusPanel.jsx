// Shows only real backend state. No fake progress bars or percentages.
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
  if (phase === 'uploaded' && job) {
    return (
      <div className="status ok" role="status">
        <strong>Upload accepted by the backend.</strong>
        <dl>
          <dt>Job ID</dt>
          <dd>{job.job_id}</dd>
          <dt>Status</dt>
          <dd>{job.status}</dd>
          <dt>Image</dt>
          <dd>
            {job.image_format} · {job.width} × {job.height}px
          </dd>
        </dl>
        <p className="note">The AI pipeline (outline extraction) is added in later phases.</p>
      </div>
    )
  }
  return null
}
