import { useRef, useState } from 'react'

export default function UploadZone({ onFile, disabled }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)

  const handleFiles = (files) => {
    const file = files && files[0]
    if (file) onFile(file)
  }

  const onDrop = (event) => {
    event.preventDefault()
    setDragging(false)
    if (!disabled) handleFiles(event.dataTransfer.files)
  }

  const onDragOver = (event) => {
    event.preventDefault()
    if (!disabled) setDragging(true)
  }

  return (
    <div
      className={`dropzone${dragging ? ' dragging' : ''}${disabled ? ' disabled' : ''}`}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={() => setDragging(false)}
    >
      <svg viewBox="0 0 24 24" width="40" height="40" aria-hidden="true">
        <path
          d="M12 16V4m0 0L7 9m5-5 5 5M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      <p className="dz-title">Drag and drop your image here</p>
      <p className="dz-sub">JPG, PNG or WEBP, up to 15 MB</p>
      <button type="button" className="btn primary" disabled={disabled} onClick={() => inputRef.current?.click()}>
        Browse image
      </button>
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp"
        hidden
        onChange={(event) => {
          handleFiles(event.target.files)
          event.target.value = '' // allow re-selecting the same file
        }}
      />
    </div>
  )
}
