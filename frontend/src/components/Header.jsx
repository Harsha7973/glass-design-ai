export default function Header() {
  return (
    <header className="header">
      <div className="logo" aria-hidden="true">
        <svg viewBox="0 0 48 48" width="44" height="44">
          <path d="M24 4 44 24 24 44 4 24Z" fill="none" stroke="currentColor" strokeWidth="2.5" />
          <path d="M24 14 34 24 24 34 14 24Z" fill="none" stroke="currentColor" strokeWidth="2" />
          <path d="M4 24h40M24 4v40" stroke="currentColor" strokeWidth="1.2" opacity=".5" />
        </svg>
      </div>
      <h1>GLASS DESIGN AI</h1>
      <p className="tagline">
        Upload a photo of a decorative object. The AI finds the design and turns it into a
        clean black outline on white. No instructions needed.
      </p>
    </header>
  )
}
