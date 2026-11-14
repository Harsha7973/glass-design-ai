import { useEffect, useRef, useState } from 'react'
import Header from '../components/Header.jsx'
import UploadZone from '../components/UploadZone.jsx'
import ImagePreview from '../components/ImagePreview.jsx'
import StatusPanel from '../components/StatusPanel.jsx'
import { ALLOWED_TYPES, MAX_UPLOAD_MB, checkHealth, getStatus, uploadImage } from '../services/api.js'

// phase: 'idle' | 'uploading' | 'uploaded' | 'error'
export default function HomePage() {
  const [phase, setPhase] = useState('idle')
  const [previewUrl, setPreviewUrl] = useState(null)
  const [fileName, setFileName] = useState('')
  const [job, setJob] = useState(null)
  const [error, setError] = useState('')
  const [backendUp, setBackendUp] = useState(null) // null = checking
  const urlRef = useRef(null)

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
  useEffect(() => () => urlRef.current && URL.revokeObjectURL(urlRef.current), [])

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
      setPhase('uploaded')
      setBackendUp(true)
    } catch (err) {
      setError(err.message)
      setPhase('error')
    }
  }

  const reset = () => {
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

        {phase !== 'idle' && (
          <div className="actions">
            <button type="button" className="btn" onClick={reset} disabled={busy}>
              Reset
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
