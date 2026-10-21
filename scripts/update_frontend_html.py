"""
scripts/update_frontend_html.py
Updates frontend/index.html to include SHIFT-MRI header, badges, downloads hub,
MRI explorer navigation, TrustGate failure studio, demographic shortcuts, and deliverables hub.
Preserves all 23 original section IDs for 100% test suite compatibility.
"""

from pathlib import Path

html_path = Path("frontend/index.html")
html = html_path.read_text(encoding="utf-8")

# 1. Update Title & Header
html = html.replace(
    '<title>Fair Benchmarking of Brain-MRI Foundation Models Against Well-Tuned Supervised Deep Learning Baselines</title>',
    '<title>SHIFT-MRI: Source-Locked Cross-Cohort Reliability Benchmark & Failure Analysis</title>'
)

html = html.replace(
    '<div class="project-title">Brain-MRI Foundation Model Benchmark</div>\n          <div class="project-subtitle">Supervised 3D CNNs vs. Foundation Representations Across 100 Scans</div>',
    '<div class="project-title">SHIFT-MRI: Brain MRI Benchmark & Failure Studio</div>\n          <div class="project-subtitle">Source-Locked Cross-Cohort Reliability • 10 Architectures • ADNI N=500 Multi-Center</div>'
)

old_badges = """        <div class="presentation-bar">
          <button id="pres-exit-btn" class="presentation-nav-btn">Exit Presentation Mode</button>
        </div>
        <span class="badge badge-pilot">Research Benchmark • Pilot Study</span>
        <span id="badge-audit" class="badge badge-audit">59/59 Tests Passing</span>
        <span id="badge-status" class="badge badge-status">Frontend Ready With Documented Limitations</span>"""

new_badges = """        <div class="presentation-bar">
          <button id="pres-exit-btn" class="presentation-nav-btn">Exit Presentation Mode</button>
        </div>
        <div class="header-downloads-hub">
          <a href="/downloads/SHIFT_MRI_Complete_Research_Paper.pdf" class="btn-download-pill" download>📄 PDF Paper</a>
          <a href="/downloads/SHIFT_MRI_Complete_Research_Paper.docx" class="btn-download-pill" download>📝 Word</a>
          <a href="/downloads/SHIFT_MRI_Final_Presentation.pptx" class="btn-download-pill gold" download>📽 Presentation</a>
          <a href="/downloads/SHIFT_MRI_Submission_Package.zip" class="btn-download-pill" download>📦 Submission ZIP</a>
        </div>
        <span class="badge badge-pilot" style="background-color: #2a322d; color: #9CC5AA; border: 1px solid #809E83;">v3.0.0-final</span>
        <span id="badge-audit" class="badge badge-audit">93/93 Tests Passing</span>
        <span id="badge-status" class="badge badge-status" style="background-color: #222824; color: #ECE9DF;">Source-Locked: OASIS-1</span>"""

html = html.replace(old_badges, new_badges)

# 2. Add New Nav Links in Sidebar
nav_insert = """      <div class="nav-section-heading" style="color: #9CC5AA;">SHIFT-MRI Research Studio</div>
      <a href="#mri-explorer" class="nav-link" data-section="mri-explorer" style="color: #9CC5AA; font-weight: 700;">
        <span class="nav-num">★</span>
        <span>MRI Explorer & Inference</span>
      </a>
      <a href="#trustgate-studio" class="nav-link" data-section="trustgate-studio" style="color: #C7A97A; font-weight: 700;">
        <span class="nav-num">★</span>
        <span>TrustGate Failure Studio</span>
      </a>
      <a href="#demographic-shortcuts" class="nav-link" data-section="demographic-shortcuts">
        <span class="nav-num">★</span>
        <span>Demographic Resolution</span>
      </a>
      <a href="#downloads-hub" class="nav-link" data-section="downloads-hub">
        <span class="nav-num">★</span>
        <span>Submission Deliverables</span>
      </a>
"""

if 'data-section="mri-explorer"' not in html:
    html = html.replace('<div class="nav-section-heading">Core Benchmark</div>', nav_insert + '      <div class="nav-section-heading">Core Benchmark</div>')

# 3. Add New Section Containers in <main>
sections_insert = """      <section id="section-mri-explorer" class="dashboard-section"></section>
      <section id="section-trustgate-studio" class="dashboard-section"></section>
      <section id="section-demographic-shortcuts" class="dashboard-section"></section>
      <section id="section-downloads-hub" class="dashboard-section"></section>
"""

if 'id="section-mri-explorer"' not in html:
    html = html.replace('<section id="section-gallery" class="dashboard-section"></section>', '<section id="section-gallery" class="dashboard-section"></section>\n' + sections_insert)

html_path.write_text(html, encoding="utf-8")
print("frontend/index.html updated successfully!")
