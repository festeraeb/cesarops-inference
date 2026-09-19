# Task: NautiInferer Landing Page — "Code Faster. Save Lives."

Write a complete single-file HTML landing page for NautiInferer — a distributed inference platform where contributors donate GPU time that powers both AI coding tools AND search-and-rescue operations.

## The Story:

"Your idle GPU helps find missing people and lost shipwrecks."

When you're coding, your GPU runs your models. When you're sleeping, it scans satellite imagery for missing aircraft, processes bathymetric data for sunken ships, and analyzes magnetic anomalies that could be century-old wrecks.

## Sections:

### 1. Hero
- Headline: "Code Faster. Save Lives."
- Subhead: "Your GPU powers AI coding tools by day and search-and-rescue by night."
- CTA: "Join the Fleet" button
- Background: subtle animated grid/particles (dark theme)

### 2. How It Works (3 columns)
- **Contribute** — Download the node daemon. It runs silently, uses your idle GPU.
- **Code** — Access multi-model inference: route tasks to coders, reviewers, correctors. Design pipelines visually.
- **Save** — When idle, your GPU scans satellite imagery, processes bathymetry, detects anomalies. Every cycle helps.

### 3. Live Fleet Stats (auto-updating)
- GPUs Online: [number]
- Tiles Scanned Today: [number]
- Models Available: [number]
- SAR Missions Active: [number]
- Historical Wrecks Located: [number]
- Fetch from: `GET /cluster/nodes` and `GET /webhook/missions`

### 4. Recent Discoveries
- Cards showing recent wreck detections / SAR missions
- "SS Cedarville — Straits of Mackinac — Confirmed via magnetic anomaly"
- "Missing fishing vessel — Whitefish Point — Satellite temporal diff"
- (These can be placeholder/example data for now)

### 5. The Mission
- "CESAROPS is a search and rescue software suite that uses remote sensing to find missing people and lost ships."
- "Every GPU cycle donated helps bring closure to families and preserves maritime history."
- "We've detected wrecks in 500 feet of water from outer space."

### 6. Supported Hardware
- Show that ANY GPU works: Pascal, Turing, Ampere, even CPU-only
- "Got an old mining card? A retired gaming GPU? It can help."
- Table: P100, 1070, 2060, 3060, etc. with tok/s estimates

### 7. For Developers
- "NautiInferer is open source. Build pipelines, route models, contribute code."
- Link to GitHub
- API docs preview

### 8. Footer
- cesarops.com
- "Saving lives with abandoned GPUs since 2024"

## Styling:
- Dark theme: #0a0a1a background, white text, cyan (#00d4ff) accent
- Gradient hero: dark blue → deep purple
- Cards with subtle glow effects
- Responsive (mobile-friendly)
- Smooth scroll between sections
- Font: Inter or system-ui

## Technical:
- Single HTML file, all CSS/JS inline
- Vanilla JS for live stats (fetch API)
- CSS animations for hero particles
- No frameworks, no build step
- Under 400 lines

## Output:
Complete HTML file ready to serve.
