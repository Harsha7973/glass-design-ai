export default function ImagePreview({ src, name }) {
  return (
    <figure className="preview">
      <img src={src} alt={`Uploaded: ${name}`} />
      <figcaption>Original image · {name}</figcaption>
    </figure>
  )
}
