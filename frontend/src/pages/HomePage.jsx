import { useEffect, useRef, useState } from 'react'
import Header from '../components/Header.jsx'
import UploadZone from '../components/UploadZone.jsx'
import ImagePreview from '../components/ImagePreview.jsx'
import StatusPanel from '../components/StatusPanel.jsx'
import ResultView from '../components/ResultView.jsx'
import { ALLOWED_TYPES, FINISHED, MAX_UPLOAD_MB, checkHealth, getStatus, uploadImage } from '../services/api.js'

// phase: 'idle' | 'uploading' | 'processing' | 'done' | 'error'
export default function HomePage() {
  const [phase, setPhase] = useState('idle')
  const [previewUrl, setPreviewUrl] = useState(null)
  const [fileName, setFileName] = useState('')
  const [job, setJob] = useState(null)
  const [error, setError] = useState('')
  const [backendUp, setBackendUp] = useState(null) // null = checking
  const urlRef = useRef(null)
  const pollRef = useRef(null)

  // Initial check: state is set only after the async call resolves.
  useEffect(() => {
    let alive = true
    checkHealth().then((ok) => {
      if (alive) setBackendUp(ok)
    })
    return () => {
      alive = false
    }
  }, [])

  const refreshHealth = async () => {
    setBackendUp(null)
    setBackendUp(await checkHealth())
  }

  // Free the blob URL when the component unmounts.
  useEffect(
    () => () => {
      if (urlRef.current) URL.revokeObjectURL(urlRef.current)
      clearTimeout(pollRef.current)
    },
    [],
  )

  // Ask the backend for the real job state until it reports a final status.
  const poll = (jobId, failures = 0) => {
    pollRef.current = setTimeout(async () => {
      try {
        const current = await getStatus(jobId)
        setJob(current)
        if (FINISHED.includes(current.status)) {
          setPhase('done')
        } else {
          poll(jobId)
        }
      } catch (err) {
        if (failures >= 4) {
          setError(`Lost contact with the backend: ${err.message}`)
          setPhase('error')
        } else {
          poll(jobId, failures + 1)
        }
      }
    }, 1500)
  }

  const setPreview = (file) => {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current)
    urlRef.current = file ? URL.createObjectURL(file) : null
    setPreviewUrl(urlRef.current)
  }

  const handleFile = async (file) => {
    setError('')
    setJob(null)

    if (!ALLOWED_TYPES.includes(file.type)) {
      setPreview(null)
      setPhase('error')
      setError('Unsupported file type. Please choose a JPG, PNG or WEBP image.')
      return
    }
    if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
      setPreview(null)
      setPhase('error')
      setError(`File is too large. The maximum size is ${MAX_UPLOAD_MB} MB.`)
      return
    }

    setPreview(file)
    setFileName(file.name)
    setPhase('uploading')
    try {
      const created = await uploadImage(file)
      const current = await getStatus(created.job_id) // real status from the backend
      setJob(current)
      setBackendUp(true)
      if (FINISHED.includes(current.status)) {
        setPhase('done')
      } else {
        setPhase('processing')
        poll(created.job_id)
      }
    } catch (err) {
      setError(err.message)
      setPhase('error')
    }
  }

  const reset = () => {
    clearTimeout(pollRef.current)
    setPreview(null)
    setFileName('')
    setJob(null)
    setError('')
    setPhase('idle')
  }

  const busy = phase === 'uploading'

  return (
    <main className="page">
      <Header />

      {backendUp === false && (
        <div className="banner" role="alert">
          Backend not reachable. Start it with <code>uvicorn main:app --reload --port 8000</code>.
          <button type="button" className="link" onClick={refreshHealth}>
            Retry
          </button>
        </div>
      )}

      <section className="card">
        {!previewUrl && <UploadZone onFile={handleFile} disabled={busy || backendUp === false} />}

        {previewUrl && <ImagePreview src={previewUrl} name={fileName} />}

        <StatusPanel phase={phase} job={job} error={error} />

        {phase === 'done' && job?.result && job.status !== 'failed' && <ResultView result={job.result} />}

        {phase !== 'idle' && (
          <div className="actions">
            <button type="button" className="btn" onClick={reset} disabled={busy}>
              {phase === 'processing' ? 'Cancel view' : 'Start over'}
            </button>
          </div>
        )}
      </section>

      <footer className="footer">
        {backendUp === null ? 'Checking backend...' : backendUp ? 'Backend connected' : 'Backend offline'}
      </footer>
    </main>
  )
}
