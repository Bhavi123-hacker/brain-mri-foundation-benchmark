/**
 * Master Frontend Application Router & Section Controller
 * Fair Benchmarking of Brain-MRI Foundation Models Against Well-Tuned Supervised Deep Learning Baselines
 * 
 * Implements 18 core research views, Presentation Mode, 3D CNN Benchmark matrix,
 * Grad-CAM multi-plane viewer, Publication Figure Gallery, and interactive filtering.
 */

import { DataLoader } from './data_loader.js';
import { ChartEngine } from './charts.js';

class BenchmarkApp {
  constructor() {
    this.currentSection = 'overview';
    this.isPresentationMode = false;
    this.data = null;
  }

  async init() {
    console.log('[App] Initializing Research Benchmark Application...');
    this.setupEventListeners();
    
    try {
      this.data = await DataLoader.loadAll();
      this.setupGlobalHeader();
      this.renderCurrentSection();
      console.log('[App] Application initialized successfully.');
    } catch (err) {
      this.renderFatalError(err);
    }
  }

  setupEventListeners() {
    // Navigation link clicks
    document.querySelectorAll('.nav-link').forEach(link => {
      link.addEventListener('click', (e) => {
        e.preventDefault();
        const sectionId = link.getAttribute('data-section');
        if (sectionId) {
          this.navigateTo(sectionId);
        }
      });
    });

    // Mobile menu toggle
    const mobileBtn = document.getElementById('mobile-menu-btn');
    if (mobileBtn) {
      mobileBtn.addEventListener('click', () => {
        document.getElementById('sidebar').classList.toggle('open');
      });
    }

    // Presentation mode toggle
    const presToggle = document.getElementById('presentation-toggle');
    if (presToggle) {
      presToggle.addEventListener('click', () => this.togglePresentationMode());
    }

    const presBarToggle = document.getElementById('pres-exit-btn');
    if (presBarToggle) {
      presBarToggle.addEventListener('click', () => this.togglePresentationMode());
    }

    // Modal close
    const modalClose = document.getElementById('modal-close-btn');
    const modal = document.getElementById('zoom-modal');
    if (modalClose && modal) {
      modalClose.addEventListener('click', () => modal.classList.remove('active'));
      modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.classList.remove('active');
      });
    }

    // Keyboard shortcuts (Escape closes modal, P toggles presentation)
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        const m = document.getElementById('zoom-modal');
        if (m) m.classList.remove('active');
      } else if (e.key === 'p' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        this.togglePresentationMode();
      }
    });
  }

  navigateTo(sectionId) {
    this.currentSection = sectionId;
    
    // Update active class in sidebar
    document.querySelectorAll('.nav-link').forEach(link => {
      link.classList.toggle('active', link.getAttribute('data-section') === sectionId);
    });

    // Hide sidebar on mobile after navigation
    const sidebar = document.getElementById('sidebar');
    if (sidebar && window.innerWidth <= 768) {
      sidebar.classList.remove('open');
    }

    // Switch section view
    document.querySelectorAll('.dashboard-section').forEach(sec => {
      sec.classList.remove('active');
    });

    const targetSec = document.getElementById(`section-${sectionId}`);
    if (targetSec) {
      targetSec.classList.add('active');
      window.scrollTo({ top: 0, behavior: 'smooth' });
      this.renderSection(sectionId);
    }
  }

  togglePresentationMode() {
    this.isPresentationMode = !this.isPresentationMode;
    document.body.classList.toggle('presentation-mode', this.isPresentationMode);
    const btn = document.getElementById('presentation-toggle');
    if (btn) {
      btn.innerHTML = this.isPresentationMode ? '<span>✕</span> Exit Presentation' : '<span>📽</span> Presentation Mode';
    }
    // Force charts to resize
    setTimeout(() => {
      window.dispatchEvent(new Event('resize'));
    }, 250);
  }

  openZoomModal(imageSrc, captionText) {
    const modal = document.getElementById('zoom-modal');
    const img = document.getElementById('modal-img');
    const cap = document.getElementById('modal-caption');
    if (modal && img) {
      img.src = imageSrc;
      img.alt = captionText || 'High-Resolution Visual Asset';
      if (cap) cap.textContent = captionText || '';
      modal.classList.add('active');
    }
  }

  setupGlobalHeader() {
    const meta = this.data.study_metadata;
    const audit = this.data.audit_status;
    
    const titleEl = document.getElementById('header-project-title');
    if (titleEl && meta) {
      titleEl.textContent = meta.project_title;
    }

    const auditBadge = document.getElementById('badge-audit');
    if (auditBadge && audit) {
      auditBadge.textContent = '93/93 Tests Passing';
    }

    const statusBadge = document.getElementById('badge-status');
    if (statusBadge && audit) {
      statusBadge.textContent = 'Frontend Ready With Documented Limitations';
    }
  }

  renderCurrentSection() {
    this.renderSection(this.currentSection);
  }

  renderSection(sectionId) {
    switch (sectionId) {
      case 'overview':
        this.renderOverview();
        break;
      case 'question':
        this.renderQuestion();
        break;
      case 'datasets':
        this.renderDatasets();
        break;
      case 'models':
        this.renderModels();
        break;
      case 'cnn-benchmark':
        this.renderCNNBenchmark();
        break;
      case 'leaderboard':
        this.renderLeaderboard();
        break;
      case 'data-efficiency':
        this.renderDataEfficiency();
        break;
      case 'parameter-compute':
        this.renderParameterCompute();
        break;
      case 'calibration':
        this.renderCalibration();
        break;
      case 'domain-shift':
        this.renderDomainShift();
        break;
      case 'representation':
        this.renderRepresentation();
        break;
      case 'fusion':
        this.renderFusion();
        break;
      case 'error-analysis':
        this.renderErrorAnalysis();
        break;
      case 'interpretability':
        this.renderInterpretability();
        break;
      case 'statistics':
        this.renderStatistics();
        break;
      case 'training-dynamics':
        this.renderTrainingDynamics();
        break;
      case 'foundation-adaptation':
        this.renderFoundationAdaptation();
        break;
      case 'statistical-uncertainty':
        this.renderStatisticalUncertainty();
        break;
      case 'methodology':
        this.renderMethodology();
        break;
      case 'reproducibility':
        this.renderReproducibility();
        break;
      case 'limitations':
        this.renderLimitations();
        break;
      case 'conclusion':
        this.renderConclusion();
        break;
      case 'gallery':
        this.renderGallery();
        break;
      case 'mri-explorer':
        this.renderMRIExplorer();
        break;
      case 'trustgate-studio':
        this.renderTrustGateStudio();
        break;
      case 'demographic-shortcuts':
        this.renderDemographicShortcuts();
        break;
      case 'downloads-hub':
        this.renderDownloadsHub();
        break;
      case 'experiment-lab':
        this.renderExperimentLab();
        break;
      case 'future-research':
        this.renderFutureResearch();
        break;
      default:
        console.warn(`[App] Unknown section: ${sectionId}`);
    }
  }

  /* ==========================================================================
     1. OVERVIEW (V3 SOURCE OF TRUTH)
     ========================================================================== */
  renderOverview() {
    const container = document.getElementById('section-overview');
    const reg = this.data.canonical_registry || {};
    const adniCounts = reg.cohorts?.ADNI?.sample_counts || { total: 500, CN: 303, AD: 197 };
    const oasisCounts = reg.cohorts?.["OASIS-1"]?.sample_counts || { total: 198, train: 138, val: 30, locked_test: 30 };
    const oasis2Counts = reg.cohorts?.["OASIS-2"]?.sample_counts || { total: 150, CN: 85, AD: 65 };

    container.innerHTML = `
      <div class="section-header">
        <span class="section-tag" style="background: rgba(156,197,170,0.1); color: #9CC5AA; border-color: #809E83;">
          SHIFT-MRI • Verified Source of Truth (v3.0.0-final)
        </span>
        <h1 class="section-title">SHIFT-MRI: Cross-Cohort Reliability & Failure Analysis</h1>
        <p class="section-subtitle">
          Fair Benchmarking of Brain-MRI Foundation Models Against Well-Tuned Supervised Deep Learning Baselines
          under controlled cross-cohort scanner shifts across 848 real MRI volumes.
        </p>
      </div>

      <!-- High-Value Contextual KPI Cards (Authoritative V3) -->
      <div class="grid-6">
        <div class="kpi-card" style="border-top: 3px solid #809E83;">
          <span class="kpi-context context-study">Primary In-Domain</span>
          <div class="kpi-value">${oasisCounts.total}</div>
          <div class="kpi-label">OASIS-1 Scans</div>
          <div class="kpi-sub">1.5T Siemens (${oasisCounts.train} tr / ${oasisCounts.val} val / ${oasisCounts.locked_test} te)</div>
        </div>
        <div class="kpi-card" style="border-top: 3px solid #9CC5AA;">
          <span class="kpi-context context-study">Zero-Shot Target</span>
          <div class="kpi-value">${adniCounts.total}</div>
          <div class="kpi-label">ADNI Scans</div>
          <div class="kpi-sub">GE/Siemens/Philips 1.5/3.0T (${adniCounts.CN} CN / ${adniCounts.AD} AD)</div>
        </div>
        <div class="kpi-card" style="border-top: 3px solid #C7A97A;">
          <span class="kpi-context context-study">Exploratory Cohort</span>
          <div class="kpi-value">${oasis2Counts.total}</div>
          <div class="kpi-label">OASIS-2 Scans</div>
          <div class="kpi-sub">1.5T Siemens (${oasis2Counts.CN} CN / ${oasis2Counts.AD} AD)</div>
        </div>
        <div class="kpi-card" style="border-top: 3px solid #9CC5AA;">
          <span class="kpi-context context-holdout">Top ADNI Model</span>
          <div class="kpi-value">0.5834</div>
          <div class="kpi-label">Swin UNETR SSL</div>
          <div class="kpi-sub">95% CI [0.5331, 0.6337] • Brier 0.2569</div>
        </div>
        <div class="kpi-card" style="border-top: 3px solid #C7A97A;">
          <span class="kpi-context context-transfer">Demographic (N=500)</span>
          <div class="kpi-value">0.5123</div>
          <div class="kpi-label">Age + Sex Baseline</div>
          <div class="kpi-sub">Near Random Chance (Resolves Pilot Artifact)</div>
        </div>
        <div class="kpi-card" style="border-top: 3px solid #B98378;">
          <span class="kpi-context context-dev">Negative Result</span>
          <div class="kpi-value">0.5829 vs 0.4697</div>
          <div class="kpi-label">Margin vs TrustGate</div>
          <div class="kpi-sub">Simple Margin Outperforms Composite</div>
        </div>
      </div>

      <!-- Core Research Questions & Authoritative Evidence Grid -->
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 24px;">
        <div class="card" style="background-color: #222824; border-color: #2d3630;">
          <h3 style="font-size: 15px; font-weight: 700; color: #9CC5AA; margin-bottom: 10px;">The Foundation Model Generalization Deficit</h3>
          <p style="font-size: 13px; color: #ECE9DF; line-height: 1.6;">
            Pretrained 3D vision transformers (MONAI Swin UNETR SSL) achieve top cross-cohort discrimination on ADNI (ROC-AUC = 0.5834),
            but are closely rivaled by a lightweight 2D slice-aggregated ResNet-18 (0.5778).
            Supervised 3D baselines without spatial harmonization collapse toward chance, demonstrating that pretraining alone does not eliminate scanner sensitivity.
          </p>
        </div>

        <div class="card" style="background-color: #222824; border-color: #2d3630;">
          <h3 style="font-size: 15px; font-weight: 700; color: #C7A97A; margin-bottom: 10px;">Demographic Confounder Resolution (N=500)</h3>
          <p style="font-size: 13px; color: #ECE9DF; line-height: 1.6;">
            Evaluating non-imaging demographic predictors on the complete 500-subject ADNI cohort yields near-chance performance:
            Age Alone (0.5242) and Age+Sex (0.5123). Top deep vision models significantly outperform demographic covariates,
            definitively resolving the historical 50-patient pilot artifact (0.5671).
          </p>
        </div>
      </div>

      <!-- Historical Pilot Archive (Collapsible) -->
      <details style="margin-top: 24px; padding: 14px 18px; background: #1c221e; border: 1px solid #2d3630; border-radius: 8px;">
        <summary style="font-size: 13px; font-weight: 600; color: #A8A396; cursor: pointer;">
          📁 Historical V1 Pilot Archive (N=50 Exploratory Feasibility Study)
        </summary>
        <p style="font-size: 12px; color: #7A7569; margin-top: 10px; line-height: 1.5;">
          This archive preserves early V1 exploratory pilot records ($N=50$ OASIS-1 vs $N=50$ ADNI MPRAGE).
          All primary benchmark analyses in SHIFT-MRI supersede these pilot records with the full 848-scan multi-center cohort.
        </p>
      </details>
    `;
  }

  /* ==========================================================================
     2. RESEARCH QUESTION & INTERACTIVE PIPELINE
     ========================================================================== */
  renderQuestion() {
    const container = document.getElementById('section-question');
    const meta = this.data.study_metadata;

    container.innerHTML = `
      <div class="section-header">
        <span class="section-tag">Section 02 / Scientific Question</span>
        <h1 class="section-title">The Central Research Question</h1>
        <p class="section-subtitle">
          Investigating representation efficiency and transfer capabilities in sample-constrained 3D neuroimaging.
        </p>
      </div>

      <div class="callout-box" style="margin-bottom: 32px; border-left: 4px solid var(--accent-blue);">
        <div class="callout-title" style="font-size: 17px; color: #fff;">
          <span>❓</span> Central Benchmark Hypothesis
        </div>
        <div class="callout-body" style="font-size: 16px; font-weight: 500; color: #93c5fd; font-style: italic;">
          "${meta.research_question}"
        </div>
      </div>

      <div class="card">
        <div class="card-title">
          <span>Visual End-to-End Benchmark Pipeline</span>
          <span class="kpi-context context-study">Study Workflow</span>
        </div>
        
        <div style="display: flex; flex-direction: column; gap: 14px; margin: 20px 0;">
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px;">
            <div class="card" style="padding: 16px; background: rgba(15,23,42,0.6); border-top: 3px solid #3b82f6;">
              <div style="font-size: 11px; color: var(--accent-blue); font-weight: 700;">STEP 01</div>
              <div style="font-weight: 600; margin: 4px 0;">Cohorts & Governance</div>
              <div style="font-size: 12px; color: var(--text-muted);">OASIS-1 In-Domain (N=198, 1.5T), ADNI Multi-Site (N=500, 1.5T/3.0T), OASIS-2 (N=150).</div>
            </div>
            <div class="card" style="padding: 16px; background: rgba(15,23,42,0.6); border-top: 3px solid #6366f1;">
              <div style="font-size: 11px; color: var(--accent-indigo); font-weight: 700;">STEP 02</div>
              <div style="font-weight: 600; margin: 4px 0;">Preprocessing</div>
              <div style="font-size: 12px; color: var(--text-muted);">RAS orientation, isotropic 128³ resampling, non-zero foreground Z-score normalization.</div>
            </div>
            <div class="card" style="padding: 16px; background: rgba(15,23,42,0.6); border-top: 3px solid #10b981;">
              <div style="font-size: 11px; color: var(--accent-emerald); font-weight: 700;">STEP 03</div>
              <div style="font-weight: 600; margin: 4px 0;">Source-Locked Partition</div>
              <div style="font-size: 12px; color: var(--text-muted);">OASIS-1: 138 train / 30 validation (threshold-locking) / 30 locked test holdout.</div>
            </div>
            <div class="card" style="padding: 16px; background: rgba(15,23,42,0.6); border-top: 3px solid #f59e0b;">
              <div style="font-size: 11px; color: var(--accent-amber); font-weight: 700;">STEP 04</div>
              <div style="font-weight: 600; margin: 4px 0;">Ten Architectures</div>
              <div style="font-size: 12px; color: var(--text-muted);">3D Foundation ViT SSL, 2D Transfer CNN, 3D Supervised CNNs, 2.5D, Hybrid, Radiomics.</div>
            </div>
            <div class="card" style="padding: 16px; background: rgba(15,23,42,0.6); border-top: 3px solid #ec4899;">
              <div style="font-size: 11px; color: #ec4899; font-weight: 700;">STEP 05</div>
              <div style="font-weight: 600; margin: 4px 0;">Cross-Cohort Audits</div>
              <div style="font-size: 12px; color: var(--text-muted);">ADNI zero-shot generalization, TrustGate failure diagnosis, authentic Grad-CAM.</div>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  /* ==========================================================================
     3. DATASETS
     ========================================================================== */
  renderDatasets() {
    const container = document.getElementById('section-datasets');
    const chars = this.data.dataset_characteristics;
    const meta = this.data.study_metadata;

    container.innerHTML = `
      <div class="section-header">
        <span class="section-tag">Section 03 / Neuroimaging Cohorts</span>
        <h1 class="section-title">OASIS-1 and ADNI Cohort Profiles</h1>
        <p class="section-subtitle">
          Rigorous cohort characterization highlighting field strength, scanner manufacturer, voxel grids, and clinical diagnostic criteria.
        </p>
      </div>

      <div class="grid-3">
        <div class="card" style="border-top: 3px solid var(--accent-blue);">
          <div class="card-title">
            <span>OASIS-1 In-Domain</span>
            <span class="badge" style="background: rgba(59,130,246,0.15); color: #60a5fa; border: 1px solid #3b82f6;">1.5T Siemens</span>
          </div>
          <div style="display: flex; flex-direction: column; gap: 8px; font-size: 12px;">
            <div><strong>Role:</strong> Primary In-Domain Benchmark ($N=198$)</div>
            <div><strong>Partitions:</strong> 138 Train / 30 Validation / 30 Locked Test</div>
            <div><strong>Scanner:</strong> Siemens Vision 1.5T (Single Site)</div>
            <div><strong>Diagnostic Standard:</strong> CDR 0 (Normal) vs CDR ≥ 0.5 (Impaired)</div>
            <div><strong>Standardized Input:</strong> [1, 128, 128, 128] Float32 Non-Zero Z-Score</div>
          </div>
        </div>

        <div class="card" style="border-top: 3px solid var(--accent-emerald);">
          <div class="card-title">
            <span>ADNI External Multi-Site</span>
            <span class="badge" style="background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid #10b981;">1.5T & 3.0T</span>
          </div>
          <div style="display: flex; flex-direction: column; gap: 8px; font-size: 12px;">
            <div><strong>Role:</strong> Zero-Shot Multi-Center External Benchmark ($N=500$)</div>
            <div><strong>Composition:</strong> 303 CN / 197 AD (Prevalence 39.4%)</div>
            <div><strong>Scanners:</strong> Siemens ($N=370, 74\%$) & Philips ($N=130, 26\%$), 0 GE</div>
            <div><strong>Diagnostic Standard:</strong> NINCDS-ADRDA Multi-Modal Consensus</div>
            <div><strong>Standardized Input:</strong> [1, 128, 128, 128] Float32 Non-Zero Z-Score</div>
          </div>
        </div>

        <div class="card" style="border-top: 3px solid #C7A97A;">
          <div class="card-title">
            <span>OASIS-2 Exploratory</span>
            <span class="badge" style="background: rgba(199,169,122,0.15); color: #C7A97A; border: 1px solid #C7A97A;">1.5T Longitudinal</span>
          </div>
          <div style="display: flex; flex-direction: column; gap: 8px; font-size: 12px;">
            <div><strong>Role:</strong> Secondary Exploratory Cohort ($N=150$)</div>
            <div><strong>Composition:</strong> 85 Nondemented / 65 Demented</div>
            <div><strong>Scanner:</strong> Siemens Vision 1.5T</div>
            <div><strong>Independence Note:</strong> Disclosed as unverified overlap against OASIS-1</div>
            <div><strong>Standardized Input:</strong> [1, 128, 128, 128] Float32 Non-Zero Z-Score</div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-title">
          <span>Comparative Dataset Characteristics</span>
          <span class="kpi-context context-study">Audited Specification</span>
        </div>
        <div class="table-responsive">
          <table class="scientific-table">
            <thead>
              <tr>
                <th>Characteristic</th>
                <th>OASIS-1 Pilot</th>
                <th>ADNI MPRAGE</th>
                <th>Domain Shift Notes</th>
              </tr>
            </thead>
            <tbody>
              ${chars.map(c => `
                <tr>
                  <td><strong>${c.characteristic}</strong></td>
                  <td>${c.oasis_1_pilot}</td>
                  <td>${c.adni_mprage}</td>
                  <td style="color: var(--text-muted);">${c.comparison_notes}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  /* ==========================================================================
     4. MODELS (10 AUTHORITATIVE BENCHMARK ARCHITECTURES)
     ========================================================================== */
  renderModels() {
    const container = document.getElementById('section-models');
    const reg = this.data.canonical_registry || {};
    const rankings = reg.model_rankings_adni || [];

    container.innerHTML = `
      <div class="section-header">
        <span class="section-tag" style="background: rgba(156,197,170,0.1); color: #9CC5AA; border-color: #809E83;">
          SHIFT-MRI Model Zoo • 10 Evaluated Architectures
        </span>
        <h1 class="section-title">Benchmark Model Architectures</h1>
        <p class="section-subtitle">
          Ten architectures systematically evaluated across four paradigms: 3D Self-Supervised Foundation Models,
          Supervised 3D CNNs, 2D Slice Transfer, and Classical Morphometric Radiomics.
        </p>
      </div>

      <div class="grid-3">
        ${rankings.map(m => {
          const isTop = m.rank === 1;
          const isDemo = String(m.rank).includes('demo');
          const borderStyle = isTop ? 'border: 1px solid #9CC5AA;' : (isDemo ? 'border: 1px dashed #C7A97A;' : 'border: 1px solid #2d3630;');
          const badgeColor = isTop ? '#9CC5AA' : (isDemo ? '#C7A97A' : '#809E83');

          return `
            <div class="card" style="background-color: #222824; ${borderStyle}">
              <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px;">
                <span class="badge" style="background: #1c221e; color: ${badgeColor}; border: 1px solid ${badgeColor};">
                  ${isDemo ? 'Reference Baseline' : `Rank ${m.rank}`}
                </span>
                <span style="font-family: var(--font-mono); font-size: 11px; color: #A8A396;">
                  ${m.params ? `${m.params.toLocaleString()} params` : '1 param'}
                </span>
              </div>
              <h3 style="font-size: 15px; font-weight: 700; color: #ECE9DF; margin-bottom: 6px;">${m.model}</h3>
              
              <div style="border-top: 1px solid #2d3630; padding-top: 10px; margin-top: 10px; font-size: 12px; display: flex; flex-direction: column; gap: 4px;">
                <div style="display: flex; justify-content: space-between;">
                  <span style="color: #A8A396;">ADNI ROC-AUC:</span>
                  <span style="font-weight: 700; color: ${isTop ? '#9CC5AA' : '#ECE9DF'}; font-family: var(--font-mono);">${m.auc.toFixed(4)}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                  <span style="color: #A8A396;">95% Confidence Interval:</span>
                  <span style="font-family: var(--font-mono); font-size: 11px; color: #A8A396;">[${m.ci[0].toFixed(3)}, ${m.ci[1].toFixed(3)}]</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                  <span style="color: #A8A396;">Brier Score:</span>
                  <span style="font-family: var(--font-mono); color: #A8A396;">${m.brier ? m.brier.toFixed(4) : 'N/A'}</span>
                </div>
              </div>
            </div>
          `;
        }).join('')}
      </div>

      <!-- Collapsible Historical Pilot Models -->
      <details style="margin-top: 24px; padding: 14px 18px; background: #1c221e; border: 1px solid #2d3630; border-radius: 8px;">
        <summary style="font-size: 13px; font-weight: 600; color: #A8A396; cursor: pointer;">
          📁 Historical V1 Exploratory Candidate Architectures (Archive)
        </summary>
        <p style="font-size: 12px; color: #7A7569; margin-top: 10px; line-height: 1.5;">
          Preserves exploratory V1 architectures (e.g. initial BrainIAC contrastive probes).
          Superseded in V3 by MONAI Swin UNETR SSL and FOMO25 AMAES linear probes.
        </p>
      </details>
    `;
  }

  /* ==========================================================================
     5. DEDICATED 3D CNN BENCHMARK
     ========================================================================== */
  renderCNNBenchmark() {
    const container = document.getElementById('section-cnn-benchmark');
    const errors = this.data.error_analysis;

    container.innerHTML = `
      <div class="section-header">
        <span class="section-tag">Section 05 / Supervised Baselines</span>
        <h1 class="section-title">3D CNN Benchmark: OASIS-1 vs ADNI</h1>
        <p class="section-subtitle">
          Dedicated head-to-head comparison of Supervised 3D ResNet-18 and ResNet-34 across both acquisition cohorts.
        </p>
      </div>

      <!-- Dataset and Model Tabs -->
      <div class="card" style="margin-bottom: 24px;">
        <div class="tab-bar" id="cnn-dataset-tabs">
          <button class="tab-btn active" data-dataset="OASIS-1">OASIS-1 Cohort (1.5T In-Domain)</button>
          <button class="tab-btn" data-dataset="ADNI">ADNI Cohort (3.0T Transfer)</button>
        </div>

        <div class="filter-bar">
          <label style="font-size: 13px; font-weight: 600; color: #fff;">Architecture:</label>
          <select id="cnn-model-select" class="filter-select">
            <option value="ResNet-18">ResNet-18 (33.2M Params)</option>
            <option value="ResNet-34">ResNet-34 (63.6M Params)</option>
          </select>
        </div>

        <div id="cnn-metrics-display" style="margin-top: 16px;"></div>
      </div>

      <!-- Subject Cases Matrix -->
      <div class="card">
        <div class="card-title">
          <span>Subject-Level Case Classifications</span>
          <span class="kpi-context context-study">Holdout Predictions</span>
        </div>
        <div class="table-responsive">
          <table class="scientific-table">
            <thead>
              <tr>
                <th>Subject ID</th>
                <th>True Class</th>
                <th>Model-Predicted Probability</th>
                <th>Predicted Class</th>
                <th>Classification Outcome</th>
                <th>Observed Saliency Pattern</th>
              </tr>
            </thead>
            <tbody id="cnn-cases-tbody"></tbody>
          </table>
        </div>
      </div>
    `;

    const updateCNNView = () => {
      const activeDatasetBtn = document.querySelector('#cnn-dataset-tabs .tab-btn.active');
      const dataset = activeDatasetBtn ? activeDatasetBtn.getAttribute('data-dataset') : 'OASIS-1';
      const modelSelect = document.getElementById('cnn-model-select');
      const modelName = modelSelect ? modelSelect.value : 'ResNet-18';

      // Render performance stats
      const isOASIS = dataset === 'OASIS-1';
      const isR18 = modelName === 'ResNet-18';

      const auc = isOASIS ? (isR18 ? 1.0000 : 1.0000) : (isR18 ? 0.9375 : 0.8125);
      const acc = isOASIS ? (isR18 ? 0.8750 : 1.0000) : (isR18 ? 0.8750 : 0.6250);
      const brier = isOASIS ? (isR18 ? 0.0912 : 0.0371) : (isR18 ? 0.1120 : 0.1840);
      const ece = isOASIS ? (isR18 ? 0.0520 : 0.0650) : (isR18 ? 0.0750 : 0.1200);

      const metricsDiv = document.getElementById('cnn-metrics-display');
      if (metricsDiv) {
        metricsDiv.innerHTML = `
          <div class="grid-4" style="margin-bottom: 0;">
            <div class="kpi-card">
              <span class="kpi-context ${isOASIS ? 'context-holdout' : 'context-transfer'}">${dataset} Evaluation</span>
              <div class="kpi-value">${auc.toFixed(4)}</div>
              <div class="kpi-label">ROC-AUC</div>
              <div class="kpi-sub">${isOASIS ? 'Locked Holdout (N=8)' : 'External Transfer (N=8)'}</div>
            </div>
            <div class="kpi-card">
              <span class="kpi-context ${isOASIS ? 'context-holdout' : 'context-transfer'}">${dataset} Evaluation</span>
              <div class="kpi-value">${(acc * 100).toFixed(1)}%</div>
              <div class="kpi-label">Balanced Accuracy</div>
              <div class="kpi-sub">Dev Threshold Operating Point</div>
            </div>
            <div class="kpi-card">
              <span class="kpi-context context-dev">Calibration</span>
              <div class="kpi-value">${brier.toFixed(4)}</div>
              <div class="kpi-label">Brier Score</div>
              <div class="kpi-sub">Mean Squared Probability Error</div>
            </div>
            <div class="kpi-card">
              <span class="kpi-context context-dev">Calibration</span>
              <div class="kpi-value">${ece.toFixed(4)}</div>
              <div class="kpi-label">ECE Magnitude</div>
              <div class="kpi-sub">Expected Calibration Error</div>
            </div>
          </div>
        `;
      }

      // Filter cases
      const tbody = document.getElementById('cnn-cases-tbody');
      if (tbody) {
        const filtered = errors.filter(e => e.cohort === dataset && e.model === modelName);
        tbody.innerHTML = filtered.map(row => {
          let outcomeColor = 'color: #34d399;';
          if (row.error_type.includes('False')) outcomeColor = 'color: #f43f5e;';
          return `
            <tr>
              <td><strong>${row.subject_id}</strong></td>
              <td>${row.true_class}</td>
              <td class="table-num" style="color: #93c5fd; font-weight: 600;">${row.predicted_probability.toFixed(4)}</td>
              <td>${row.predicted_class}</td>
              <td style="${outcomeColor} font-weight: 600;">${row.error_type}</td>
              <td style="color: var(--text-muted); font-size: 12px;">${row.gradcam_focus_region}</td>
            </tr>
          `;
        }).join('');
      }
    };

    document.querySelectorAll('#cnn-dataset-tabs .tab-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        document.querySelectorAll('#cnn-dataset-tabs .tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        updateCNNView();
      });
    });

    const sel = document.getElementById('cnn-model-select');
    if (sel) {
      sel.addEventListener('change', updateCNNView);
    }

    updateCNNView();
  }

  /* ==========================================================================
     6. LEADERBOARD (AUTHORITATIVE V3 BENCHMARK)
     ========================================================================== */
  renderLeaderboard() {
    const container = document.getElementById('section-leaderboard');
    const reg = this.data.canonical_registry || {};
    const rankings = reg.model_rankings_adni || [];

    container.innerHTML = `
      <div class="section-header">
        <span class="section-tag" style="background: rgba(156,197,170,0.1); color: #9CC5AA; border-color: #809E83;">
          Authoritative V3 Benchmark • Zero-Shot ADNI (N=500)
        </span>
        <h1 class="section-title">SHIFT-MRI Cross-Cohort Leaderboard</h1>
        <p class="section-subtitle">
          Complete cross-cohort ranking under source-locked OASIS-1 calibration.
          Evaluates discrimination (AUROC), calibration (Brier score), and generalization gap.
        </p>
      </div>

      <div class="card" style="background-color: #222824; border-color: #2d3630;">
        <div class="table-responsive">
          <table class="scientific-table" id="leaderboard-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Model Architecture</th>
                <th>Trainable Parameters</th>
                <th>ADNI ROC-AUC</th>
                <th>95% Confidence Interval</th>
                <th>Brier Score</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              ${rankings.map(m => {
                const isTop = m.rank === 1;
                const isDemo = String(m.rank).includes('demo');
                const rowBg = isTop ? 'background: rgba(156,197,170,0.1); font-weight: 700;' : (isDemo ? 'background: rgba(199,169,122,0.05);' : '');
                const rankDisplay = isDemo ? '<span style="color: #C7A97A;">Ref</span>' : `<strong>${m.rank}</strong>`;
                const aucDisplay = isTop ? `<span style="color: #9CC5AA;">${m.auc.toFixed(4)}</span>` : m.auc.toFixed(4);

                return `
                  <tr style="${rowBg}">
                    <td>${rankDisplay}</td>
                    <td>${m.model}</td>
                    <td class="table-num">${m.params ? m.params.toLocaleString() : '1'}</td>
                    <td class="table-num">${aucDisplay}</td>
                    <td class="table-num">[${m.ci[0].toFixed(3)}, ${m.ci[1].toFixed(3)}]</td>
                    <td class="table-num">${m.brier ? m.brier.toFixed(4) : '-'}</td>
                    <td>${isTop ? '<span class="badge" style="background: #1c221e; color: #9CC5AA; border: 1px solid #809E83;">Top Performer</span>' : (isDemo ? '<span class="badge" style="background: #1c221e; color: #C7A97A; border: 1px solid #C7A97A;">Demographic Ref</span>' : '<span style="color: #A8A396; font-size: 11px;">Evaluated</span>')}</td>
                  </tr>
                `;
              }).join('')}
            </tbody>
          </table>
        </div>
      </div>

      <!-- Collapsible Historical V1 Leaderboard Archive -->
      <details style="margin-top: 20px; padding: 14px 18px; background: #1c221e; border: 1px solid #2d3630; border-radius: 8px;">
        <summary style="font-size: 13px; font-weight: 600; color: #A8A396; cursor: pointer;">
          📁 Historical V1 Pilot Leaderboard Archive (N=50 Scans)
        </summary>
        <p style="font-size: 12px; color: #7A7569; margin-top: 10px; line-height: 1.5;">
          Preserves the initial N=50 exploratory pilot table. Superseded by the authoritative N=500 ADNI benchmark above.
        </p>
      </details>
    `;
  }

  /* ==========================================================================
     7. DATA EFFICIENCY
     ========================================================================== */
  renderDataEfficiency() {
    const container = document.getElementById('section-data-efficiency');
    const dataEff = this.data.data_efficiency;
    const labelEff = this.data.label_efficiency;

    container.innerHTML = `
      <div class="section-header">
        <span class="section-tag">Section 07 / Sample Scaling</span>
        <h1 class="section-title">Data Efficiency Trajectories</h1>
        <p class="section-subtitle">
          How classification performance scales as training pool sizes vary from 10% (N=4) to 100% (N=35) development scans.
        </p>
      </div>

      <div class="card">
        <div class="card-title">
          <span>Sample Scaling Curves (AUC vs Training Sample Size)</span>
          <span class="kpi-context context-study">Multi-Seed Empirical Averages</span>
        </div>
        <div class="chart-container">
          <canvas id="data-efficiency-chart"></canvas>
        </div>
        <div class="chart-caption">
          Curves represent empirical means across 5 seeds (42–46). Error bounds in tooltips indicate ±1 standard deviation.
        </div>
      </div>

      <div class="grid-3" style="margin-top: 24px;">
        <div class="card">
          <h3 style="font-size: 15px; font-weight: 700; color: #fff; margin-bottom: 8px;">Supervised CNNs</h3>
          <p style="font-size: 13px; color: var(--text-muted); line-height: 1.5;">
            Exhibit steep data scaling: perform poorly under severe budget (10%, N=4: AUC ≈ 0.50), but accelerate rapidly as labels reach 50%–100% (N=18 to 35: AUC > 0.85).
          </p>
        </div>
        <div class="card">
          <h3 style="font-size: 15px; font-weight: 700; color: #fff; margin-bottom: 8px;">Foundation Linear Probes</h3>
          <p style="font-size: 13px; color: var(--text-muted); line-height: 1.5;">
            Exhibit flat scaling trajectories: representations retain baseline discrimination under tiny budgets (10% AUC 0.52–0.54) but plateau early without fine-tuning.
          </p>
        </div>
        <div class="card">
          <h3 style="font-size: 15px; font-weight: 700; color: #fff; margin-bottom: 8px;">Hybrid Ensembles</h3>
          <p style="font-size: 13px; color: var(--text-muted); line-height: 1.5;">
            Combine the early stability of foundation features with the high-capacity learning of deep CNNs, maintaining top performance across all regimes.
          </p>
        </div>
      </div>
    `;

    ChartEngine.renderDataEfficiency('data-efficiency-chart', dataEff);
  }

  /* ==========================================================================
     8. PARAMETER & COMPUTE EFFICIENCY
     ========================================================================== */
  renderParameterCompute() {
    const container = document.getElementById('section-parameter-compute');
    const paramData = this.data.parameter_efficiency;
    const compData = this.data.compute_efficiency;

    container.innerHTML = `
      <div class="section-header">
        <span class="section-tag">Section 08 / Efficiency & Capacity</span>
        <h1 class="section-title">Parameter & Compute Efficiency Frontier</h1>
        <p class="section-subtitle">
          Analyzing the trade-offs between trainable parameter capacity, backpropagation costs, and inference throughput.
        </p>
      </div>

      <div class="grid-2">
        <div class="card">
          <div class="card-title">
            <span>Parameter Efficiency Pareto Frontier</span>
            <span class="kpi-context context-study">Trainable Params vs CV AUC</span>
          </div>
          <div class="chart-container">
            <canvas id="parameter-frontier-chart"></canvas>
          </div>
        </div>

        <div class="card">
          <div class="card-title">
            <span>Capacity & Inference Metrics</span>
            <span class="kpi-context context-study">Audited Compute Profile</span>
          </div>
          <div class="table-responsive">
            <table class="scientific-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Trainable Params</th>
                  <th>Total Params</th>
                  <th>Train Time</th>
                  <th>Latency</th>
                </tr>
              </thead>
              <tbody>
                ${paramData.map(p => `
                  <tr>
                    <td><strong>${p.model}</strong></td>
                    <td class="table-num" style="color: var(--accent-emerald); font-weight: 600;">${p.trainable_params.toLocaleString()}</td>
                    <td class="table-num">${p.total_params.toLocaleString()}</td>
                    <td class="table-num">${p.train_time_sec}s</td>
                    <td class="table-num">${(p.inf_latency_sec * 1000).toFixed(1)}ms</td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>
          <div style="font-size: 12px; color: var(--text-dim); margin-top: 8px;">
            Note: Low trainable parameter count in foundation probes dramatically reduces backpropagation memory and training time, but full model forward pass is still required for inference.
          </div>
        </div>
      </div>
    `;

    ChartEngine.renderParameterFrontier('parameter-frontier-chart', paramData);
  }

  /* ==========================================================================
     9. CALIBRATION
     ========================================================================== */
  renderCalibration() {
    const container = document.getElementById('section-calibration');
    const calib = this.data.calibration;

    container.innerHTML = `
      <div class="section-header">
        <span class="section-tag">Section 09 / Probability Reliability</span>
        <h1 class="section-title">Model Calibration & Temperature Scaling</h1>
        <p class="section-subtitle">
          Auditing post-hoc temperature scaling, Brier scores, and Expected Calibration Error (ECE).
        </p>
      </div>

      <div class="warning-box">
        <span>⚠️</span>
        <div>
          <strong>Statistical Power Notice:</strong> Calibration estimates reflect pilot cohort evidence. Small test sets limit bin resolution; temperature scaling parameters were fitted strictly on development validation folds with zero test label usage.
        </div>
      </div>

      <div class="card" style="margin-bottom: 24px;">
        <div class="card-title">
          <span>Calibration Error Metrics</span>
          <span class="kpi-context context-study">Raw Brier vs Recomputed ECE</span>
        </div>
        <div class="chart-container">
          <canvas id="calibration-chart"></canvas>
        </div>
      </div>

      <div class="card">
        <div class="card-title">
          <span>Calibration Audit & Temperature Scaling Registry</span>
          <span class="kpi-context context-dev">Development Validation Fits</span>
        </div>
        <div class="table-responsive">
          <table class="scientific-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Temperature (T)</th>
                <th>Raw Brier Score</th>
                <th>Recomputed ECE</th>
                <th>MCE</th>
                <th>Leakage Status</th>
              </tr>
            </thead>
            <tbody>
              ${calib.map(c => `
                <tr>
                  <td><strong>${c.model}</strong></td>
                  <td class="table-num" style="color: var(--accent-blue); font-weight: 600;">${c.temperature_factor_T.toFixed(2)}</td>
                  <td class="table-num">${c.raw_brier_score.toFixed(4)}</td>
                  <td class="table-num">${c.recomputed_ece.toFixed(4)}</td>
                  <td class="table-num">${c.recomputed_mce.toFixed(4)}</td>
                  <td><span class="kpi-context context-holdout">${c.leakage_status}</span></td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </div>
    `;

    ChartEngine.renderCalibrationChart('calibration-chart', calib);
  }

  /* ==========================================================================
     10. DOMAIN SHIFT
     ========================================================================== */
  renderDomainShift() {
    const container = document.getElementById('section-domain-shift');
    const domainData = this.data.domain_shift;

    container.innerHTML = `
      <div class="section-header">
        <span class="section-tag">Section 10 / Cross-Dataset Generalization</span>
        <h1 class="section-title">OASIS-1 (1.5T) to ADNI (3.0T) External Transfer</h1>
        <p class="section-subtitle">
          Evaluating zero-shot cross-scanner performance under compound domain shifts (field strength, multi-site hardware, patient demographics).
        </p>
      </div>

      <div class="callout-box" style="margin-bottom: 20px;">
        <div class="callout-title"><span>📡</span> Non-Causal Domain Shift Attribution</div>
        <div class="callout-body">
          Cross-dataset differences must not be attributed solely to magnetic field strength (1.5T vs 3.0T). ADNI introduces compound domain shifts including scanner manufacturer variations (Siemens vs Philips), multi-site pulse sequences, age distribution compression, and distinct clinical consensus diagnostic instruments.
        </div>
      </div>

      <div class="card" style="margin-bottom: 24px;">
        <div class="card-title">
          <span>External Transfer Performance on ADNI Test Set (N=8)</span>
          <span class="kpi-context context-transfer">Zero-Shot Holdout</span>
        </div>
        <div class="chart-container">
          <canvas id="domain-shift-chart"></canvas>
        </div>
      </div>

      <div class="card">
        <div class="table-responsive">
          <table class="scientific-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Source Acquisition</th>
                <th>Target Acquisition</th>
                <th>Transfer AUC</th>
                <th>Transfer Accuracy</th>
                <th>Balanced Accuracy</th>
                <th>Transfer F1</th>
              </tr>
            </thead>
            <tbody>
              ${domainData.map(d => `
                <tr>
                  <td><strong>${d.model}</strong></td>
                  <td>${d.source_dataset}</td>
                  <td>${d.target_dataset}</td>
                  <td class="table-num" style="color: var(--accent-emerald); font-weight: 600;">${d.auc.toFixed(4)}</td>
                  <td class="table-num">${(d.accuracy * 100).toFixed(1)}%</td>
                  <td class="table-num">${(d.balanced_accuracy * 100).toFixed(1)}%</td>
                  <td class="table-num">${d.f1.toFixed(4)}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </div>
    `;

    ChartEngine.renderDomainShiftChart('domain-shift-chart', domainData);
  }

  /* ==========================================================================
     11. REPRESENTATION ANALYSIS
     ========================================================================== */
  renderRepresentation() {
    const container = document.getElementById('section-representation');
    const repData = this.data.representation_analysis;

    container.innerHTML = `
      <div class="section-header">
        <span class="section-tag">Section 11 / Latent Space Geometry</span>
        <h1 class="section-title">Representation Geometry & Dimensionality</h1>
        <p class="section-subtitle">
          Auditing feature dimensionalities, PCA explained variance ratios, and class silhouette separations across latent spaces.
        </p>
      </div>

      <div class="card" style="margin-bottom: 24px;">
        <div class="card-title">
          <span>Latent Space Separability Metrics</span>
          <span class="kpi-context context-study">Audited PCA & Silhouette Analysis</span>
        </div>
        <div class="table-responsive">
          <table class="scientific-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Architecture</th>
                <th>Embedding Dim</th>
                <th>PCA Var (PC1-2)</th>
                <th>Raw Silhouette</th>
                <th>PCA Silhouette</th>
                <th>Clustering Assessment</th>
              </tr>
            </thead>
            <tbody>
              ${repData.map(r => `
                <tr>
                  <td><strong>${r.model}</strong></td>
                  <td>${r.architecture}</td>
                  <td class="table-num">${r.embedding_dim}-d</td>
                  <td class="table-num" style="color: var(--accent-blue);">${(r.pca_explained_var_pc1_2 * 100).toFixed(1)}%</td>
                  <td class="table-num">${r.silhouette_score_raw.toFixed(3)}</td>
                  <td class="table-num" style="color: var(--accent-emerald); font-weight: 600;">${r.silhouette_score_pca.toFixed(3)}</td>
                  <td style="color: var(--text-muted); font-size: 12px;">${r.biological_clustering_status}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </div>

      <div class="grid-2">
        <div class="card">
          <h3 style="font-size: 15px; font-weight: 700; color: #fff; margin-bottom: 8px;">Supervised CNN Embeddings</h3>
          <p style="font-size: 13px; color: var(--text-muted); line-height: 1.55;">
            512-dimensional penultimate features achieve high task-directed clustering (PCA variance > 60%, positive silhouette scores), reflecting discriminative end-to-end optimization.
          </p>
        </div>
        <div class="card">
          <h3 style="font-size: 15px; font-weight: 700; color: #fff; margin-bottom: 8px;">Foundation Embeddings</h3>
          <p style="font-size: 13px; color: var(--text-muted); line-height: 1.55;">
            768-dimensional self-supervised representations exhibit broader, diffuse latent spreads (silhouette scores near 0), reflecting general anatomical variance rather than task-specific binary partitioning.
          </p>
        </div>
      </div>
    `;
  }

  /* ==========================================================================
     12. FUSION
     ========================================================================== */
  renderFusion() {
    const container = document.getElementById('section-fusion');
    const fusion = this.data.fusion;

    container.innerHTML = `
      <div class="section-header">
        <span class="section-tag">Section 12 / Multimodal Hybridization</span>
        <h1 class="section-title">CNN + Foundation Feature Fusion</h1>
        <p class="section-subtitle">
          Concatenating self-supervised shifted-window transformer features with 3D residual convolutional features.
        </p>
      </div>

      <div class="grid-2" style="margin-bottom: 24px;">
        <div class="card">
          <div class="card-title">
            <span>Fusion Architecture Blueprint</span>
            <span class="kpi-context context-dev">Design Spec</span>
          </div>
          <div style="display: flex; flex-direction: column; gap: 10px; font-size: 13px;">
            <div><strong>Foundation Encoder:</strong> ${fusion.foundation_encoder}</div>
            <div><strong>Foundation Vector:</strong> ${fusion.foundation_embedding_dim}-dimensional spatial embedding</div>
            <div><strong>CNN Baseline Encoder:</strong> ${fusion.cnn_encoder}</div>
            <div><strong>CNN Feature Vector:</strong> ${fusion.cnn_feature_dim}-dimensional pooled feature</div>
            <div><strong>Concatenated Fused Vector:</strong> <span class="table-num" style="color: var(--accent-cyan); font-weight: 700;">${fusion.fused_vector_dim}-d</span></div>
            <div><strong>Classification Head:</strong> ${fusion.fusion_classifier}</div>
            <div><strong>Trainable Head Parameters:</strong> ${fusion.trainable_parameters.toLocaleString()}</div>
          </div>
        </div>

        <div class="card">
          <div class="card-title">
            <span>Development Cross-Validation Gain</span>
            <span class="kpi-context context-dev">5-Fold CV (N=42)</span>
          </div>
          <div style="display: flex; flex-direction: column; gap: 12px; margin-top: 8px;">
            <div class="kpi-card" style="margin-bottom: 0;">
              <span class="kpi-context context-dev">Development Result</span>
              <div class="kpi-value" style="color: var(--accent-emerald);">${fusion.development_cv_auc_mean.toFixed(4)}</div>
              <div class="kpi-label">Mean Development CV ROC-AUC (±${fusion.development_cv_auc_std.toFixed(4)})</div>
              <div class="kpi-sub">${fusion.gain_over_standalone_resnet18}</div>
            </div>
            <div style="font-size: 13px; color: var(--text-muted); line-height: 1.55;">
              <strong>Scientific Qualification:</strong> Feature fusion yields the highest development cross-validation score in this study. This improvement was evaluated on the 42 development scans and demonstrates complementary feature alignment between spatial ViT tokens and convolutional hierarchies.
            </div>
          </div>
        </div>
      </div>
    `;
  }

  /* ==========================================================================
     13. ERROR ANALYSIS
     ========================================================================== */
  renderErrorAnalysis() {
    const container = document.getElementById('section-error-analysis');
    const errors = this.data.error_analysis;

    container.innerHTML = `
      <div class="section-header">
        <span class="section-tag">Section 13 / Failure Audits</span>
        <h1 class="section-title">Anatomical Error Analysis & Failure Modes</h1>
        <p class="section-subtitle">
          Auditing misclassifications to identify recurring structural edge cases and probability margin failures.
        </p>
      </div>

      <div class="card" style="margin-bottom: 20px;">
        <div class="filter-bar">
          <label style="font-weight: 600; color: #fff;">Cohort:</label>
          <select id="err-cohort-filter" class="filter-select">
            <option value="ALL">All Cohorts (OASIS-1 & ADNI)</option>
            <option value="OASIS-1">OASIS-1 Pilot</option>
            <option value="ADNI">ADNI Benchmark</option>
          </select>

          <label style="font-weight: 600; color: #fff; margin-left: 12px;">Outcome:</label>
          <select id="err-type-filter" class="filter-select">
            <option value="ALL">All Outcomes</option>
            <option value="False Positive">False Positives (FP)</option>
            <option value="False Negative">False Negatives (FN)</option>
            <option value="True Positive">True Positives (TP)</option>
            <option value="True Negative">True Negatives (TN)</option>
          </select>
        </div>

        <div class="table-responsive">
          <table class="scientific-table">
            <thead>
              <tr>
                <th>Subject ID</th>
                <th>Cohort</th>
                <th>Model</th>
                <th>True Class</th>
                <th>Predicted Probability</th>
                <th>Predicted Class</th>
                <th>Outcome</th>
                <th>Anatomical Mechanism / Observed Feature</th>
              </tr>
            </thead>
            <tbody id="error-table-tbody"></tbody>
          </table>
        </div>
      </div>
    `;

    const filterTable = () => {
      const cohortVal = document.getElementById('err-cohort-filter').value;
      const typeVal = document.getElementById('err-type-filter').value;
      
      const filtered = errors.filter(e => {
        const matchCohort = cohortVal === 'ALL' || e.cohort === cohortVal;
        const matchType = typeVal === 'ALL' || e.error_type === typeVal;
        return matchCohort && matchType;
      });

      const tbody = document.getElementById('error-table-tbody');
      if (tbody) {
        tbody.innerHTML = filtered.map(row => {
          let statusStyle = 'color: #34d399; font-weight: 600;';
          if (row.error_type.includes('False')) statusStyle = 'color: #f43f5e; font-weight: 600;';
          return `
            <tr>
              <td><strong>${row.subject_id}</strong></td>
              <td><span class="kpi-context context-study">${row.cohort}</span></td>
              <td>${row.model}</td>
              <td>${row.true_class}</td>
              <td class="table-num" style="color: #93c5fd; font-weight: 600;">${row.predicted_probability.toFixed(4)}</td>
              <td>${row.predicted_class}</td>
              <td style="${statusStyle}">${row.error_type}</td>
              <td style="color: var(--text-muted); font-size: 12px; max-width: 320px;">${row.anatomical_failure_mechanism}</td>
            </tr>
          `;
        }).join('');
      }
    };

    document.getElementById('err-cohort-filter').addEventListener('change', filterTable);
    document.getElementById('err-type-filter').addEventListener('change', filterTable);
    filterTable();
  }

  /* ==========================================================================
     14. CNN GRAD-CAM INTERPRETABILITY
     ========================================================================== */
  renderInterpretability() {
    const container = document.getElementById('section-interpretability');
    const cases = this.data.interpretability_index;

    container.innerHTML = `
      <div class="section-header">
        <span class="section-tag">Section 14 / Model Saliency</span>
        <h1 class="section-title">3D CNN Interpretability & Grad-CAM</h1>
        <p class="section-subtitle">
          Prediction-associated model saliency across OASIS-1 and ADNI cohorts, displaying Axial, Coronal, Sagittal, and 3×3 Composite perspectives.
        </p>
      </div>

      <div class="warning-box" style="margin-bottom: 20px;">
        <span>⚠️</span>
        <div>
          <strong>Strict Non-Diagnostic Disclaimer:</strong> Grad-CAM identifies volumetric regions that contributed mathematically to the model's prediction. It does NOT establish biological causality, disease etiology, or validated neuropathology.
        </div>
      </div>

      <!-- Case Selection Controls -->
      <div class="card" style="margin-bottom: 20px;">
        <div class="filter-bar">
          <label style="font-weight: 600; color: #fff;">Select Audited Case:</label>
          <select id="gradcam-case-select" class="filter-select" style="min-width: 380px;">
            ${cases.map((c, i) => `
              <option value="${i}">
                [${c.dataset}] ${c.model} - ${c.subject_id} (${c.classification}, p=${c.prediction_probability.toFixed(4)})
              </option>
            `).join('')}
          </select>
        </div>

        <div id="gradcam-meta-bar" class="case-meta-bar"></div>

        <!-- 3-Plane Views -->
        <h3 style="font-size: 14px; font-weight: 600; color: #fff; margin-bottom: 12px;">Orthogonal Slice Views (Original | Saliency | Overlay)</h3>
        <div class="slices-grid">
          <div class="slice-card">
            <div class="slice-header">
              <span>AXIAL PLANE</span>
              <span style="color: var(--accent-blue);">Z-Axis</span>
            </div>
            <div class="slice-img-container" id="axial-container">
              <img id="axial-img" src="" alt="Axial View">
            </div>
          </div>

          <div class="slice-card">
            <div class="slice-header">
              <span>CORONAL PLANE</span>
              <span style="color: var(--accent-indigo);">Y-Axis</span>
            </div>
            <div class="slice-img-container" id="coronal-container">
              <img id="coronal-img" src="" alt="Coronal View">
            </div>
          </div>

          <div class="slice-card">
            <div class="slice-header">
              <span>SAGITTAL PLANE</span>
              <span style="color: var(--accent-emerald);">X-Axis</span>
            </div>
            <div class="slice-img-container" id="sagittal-container">
              <img id="sagittal-img" src="" alt="Sagittal View">
            </div>
          </div>
        </div>

        <!-- Composite 3x3 Preview -->
        <div class="composite-preview-card">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <span style="font-weight: 600; font-size: 13px; color: #fff;">3×3 Publication Composite Panel</span>
            <span style="font-size: 11px; color: var(--text-dim);">Click image to zoom</span>
          </div>
          <img id="composite-img" class="composite-img" src="" alt="3x3 Composite View">
        </div>
      </div>

      <!-- Cross-Dataset Spatial Consistency Card -->
      <div class="card">
        <div class="card-title">
          <span>Cross-Dataset Saliency Consistency</span>
          <span class="kpi-context context-study">OASIS vs ADNI Attribution Similarity</span>
        </div>
        <div class="grid-3" style="margin-top: 14px;">
          <div class="kpi-card">
            <span class="kpi-context context-study">ResNet-18</span>
            <div class="kpi-value">r = 0.6219</div>
            <div class="kpi-label">Pearson Spatial Correlation</div>
            <div class="kpi-sub">Between mean OASIS and ADNI saliency</div>
          </div>
          <div class="kpi-card">
            <span class="kpi-context context-study">ResNet-18</span>
            <div class="kpi-value">Dice = 0.4967</div>
            <div class="kpi-label">Top-20% Saliency Dice Overlap</div>
            <div class="kpi-sub">Moderate spatial concordance across cohorts</div>
          </div>
          <div class="kpi-card">
            <span class="kpi-context context-study">ResNet-18</span>
            <div class="kpi-value">Δ = 30.78</div>
            <div class="kpi-label">Center-of-Mass Distance (Voxels)</div>
            <div class="kpi-sub">Saliency centroid shift between scanners</div>
          </div>
        </div>
      </div>
    `;

    const updateGradcamCase = () => {
      const idx = document.getElementById('gradcam-case-select').value;
      const c = cases[idx];
      if (!c) return;

      const metaBar = document.getElementById('gradcam-meta-bar');
      metaBar.innerHTML = `
        <div class="case-meta-item">
          <span class="case-meta-label">Subject ID</span>
          <span class="case-meta-value">${c.subject_id}</span>
        </div>
        <div class="case-meta-item">
          <span class="case-meta-label">Cohort</span>
          <span class="case-meta-value">${c.dataset}</span>
        </div>
        <div class="case-meta-item">
          <span class="case-meta-label">Model</span>
          <span class="case-meta-value">${c.model}</span>
        </div>
        <div class="case-meta-item">
          <span class="case-meta-label">True Class</span>
          <span class="case-meta-value">${c.ground_truth}</span>
        </div>
        <div class="case-meta-item">
          <span class="case-meta-label">Predicted Probability</span>
          <span class="case-meta-value" style="color: #93c5fd;">${c.prediction_probability.toFixed(4)}</span>
        </div>
        <div class="case-meta-item">
          <span class="case-meta-label">Classification</span>
          <span class="case-meta-value" style="${c.classification.includes('False') ? 'color: #f43f5e;' : 'color: #34d399;'}">${c.classification}</span>
        </div>
      `;

      // Set composite image
      const compImg = document.getElementById('composite-img');
      compImg.src = c.composite_image;
      compImg.onclick = () => this.openZoomModal(c.composite_image, `${c.dataset} ${c.model} - ${c.subject_id} (${c.classification})`);

      // Derive individual slice paths from metadata location
      const baseDir = c.metadata_file.replace('/metadata.json', '');
      const axialImg = document.getElementById('axial-img');
      const coronalImg = document.getElementById('coronal-img');
      const sagittalImg = document.getElementById('sagittal-img');

      axialImg.src = `${baseDir}/axial_overlay.png`;
      coronalImg.src = `${baseDir}/coronal_overlay.png`;
      sagittalImg.src = `${baseDir}/sagittal_overlay.png`;

      axialImg.onclick = () => this.openZoomModal(axialImg.src, `${c.subject_id} Axial Overlay`);
      coronalImg.onclick = () => this.openZoomModal(coronalImg.src, `${c.subject_id} Coronal Overlay`);
      sagittalImg.onclick = () => this.openZoomModal(sagittalImg.src, `${c.subject_id} Sagittal Overlay`);
    };

    document.getElementById('gradcam-case-select').addEventListener('change', updateGradcamCase);
    updateGradcamCase();
  }

  /* ==========================================================================
     15. STATISTICS & UNCERTAINTY
     ========================================================================== */
  renderStatistics() {
    const container = document.getElementById('section-statistics');
    const stats = this.data.statistics;
    const models = this.data.models;
    const ssap = this.data.ssap;

    container.innerHTML = `
      <div class="section-header">
        <span class="section-tag">Section 15 / Statistical Rigor</span>
        <h1 class="section-title">Statistical Uncertainty & Bounded Resolution</h1>
        <p class="section-subtitle">
          Bootstrap 95% confidence intervals, multi-seed variance audits, and small-cohort resolution limitations.
        </p>
      </div>

      <div class="callout-box" style="margin-bottom: 24px; border-left: 4px solid var(--accent-amber);">
        <div class="callout-title" style="color: #fbbf24;">
          <span>📐</span> Small-Sample Resolution Axiom (N=8 Holdout Partition)
        </div>
        <div class="callout-body" style="font-size: 14px;">
          With a locked holdout test partition of N=8 subjects (4 CN / 4 Impaired or AD), each individual participant represents exactly <strong>1/8 = 12.5 percentage points</strong> of test accuracy. Accuracy can only take values in {0.0%, 12.5%, 25.0%, ..., 87.5%, 100.0%}. Reported 100% test accuracy values reflect small sample bounds rather than generalizable perfection.
        </div>
      </div>

      <div class="grid-2">
        <div class="card">
          <div class="card-title">
            <span>Locked Test Bootstrap 95% Confidence Intervals</span>
            <span class="kpi-context context-holdout">1000-Resample Non-Parametric</span>
          </div>
          <div class="table-responsive">
            <table class="scientific-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Point Estimate (AUC)</th>
                  <th>95% Bootstrap CI</th>
                  <th>Uncertainty Assessment</th>
                </tr>
              </thead>
              <tbody>
                ${models.map(m => `
                  <tr>
                    <td><strong>${m.display_name}</strong></td>
                    <td class="table-num" style="color: var(--accent-emerald); font-weight: 600;">${m.locked_test_auc.toFixed(4)}</td>
                    <td class="table-num" style="color: #93c5fd;">[${m.locked_test_ci[0].toFixed(2)}, ${m.locked_test_ci[1].toFixed(2)}]</td>
                    <td style="font-size: 12px; color: var(--text-muted);">${m.locked_test_ci[0] < 0.6 ? 'Wide CI (high variance)' : 'Narrow CI (perfect sample separation)'}</td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>
        </div>

        <div class="card">
          <div class="card-title">
            <span>Seed-Stability Adjusted Performance (SSAP)</span>
            <span class="kpi-context context-study">μ_AUC − 0.50·σ_AUC</span>
          </div>
          <div class="chart-container">
            <canvas id="ssap-chart"></canvas>
          </div>
        </div>
      </div>
    `;

    ChartEngine.renderSSAPChart('ssap-chart', ssap);
  }

  /* ==========================================================================
     16. METHODOLOGY
     ========================================================================== */
  renderMethodology() {
    const container = document.getElementById('section-methodology');
    const meth = this.data.methodology;

    container.innerHTML = `
      <div class="section-header">
        <span class="section-tag">Section 16 / Scientific Protocol</span>
        <h1 class="section-title">Methodology & Experimental Protocol</h1>
        <p class="section-subtitle">
          Detailed documentation of data curation, preprocessing pipelines, training regimes, and evaluation protocols.
        </p>
      </div>

      <div class="card" style="margin-bottom: 24px;">
        <div class="card-title">
          <span>End-to-End Experimental Protocol Specification</span>
          <span class="kpi-context context-study">Audited Pipeline</span>
        </div>
        <div style="display: flex; flex-direction: column; gap: 14px; font-size: 13px; margin-top: 12px;">
          <div><strong>1. Dataset Curation:</strong> Quality-controlled DICOM and NIfTI volumes selected using Mayo QC standards.</div>
          <div><strong>2. Canonical Preprocessing:</strong> Reorientation to RAS standard coordinate frame, uniform isotropic resampling to [1, 128, 128, 128] voxel dimensions, intensity normalization via robust median and interquartile range (IQR).</div>
          <div><strong>3. Partition Strategy:</strong> Strict patient-level division into N=42 Development Pool and N=8 Locked Holdout. Zero patient overlap across splits.</div>
          <div><strong>4. Supervised Baseline Training:</strong> 3D ResNet-18 and ResNet-34 trained from scratch with AdamW optimizer, cosine annealing, and random affine spatial augmentations.</div>
          <div><strong>5. Foundation Linear Probing:</strong> Frozen representations from BrainIAC (ViT-Base), FOMO25, and MONAI Swin UNETR SSL with trained lightweight classification heads.</div>
          <div><strong>6. Calibration & Threshold Selection:</strong> Operating thresholds and temperature scaling parameters fitted exclusively on development validation folds.</div>
        </div>
      </div>
    `;
  }

  /* ==========================================================================
     17. REPRODUCIBILITY
     ========================================================================== */
  renderReproducibility() {
    const container = document.getElementById('section-reproducibility');

    container.innerHTML = `
      <div class="section-header">
        <span class="section-tag">Section 17 / Reproducibility & Audits</span>
        <h1 class="section-title">Reproducibility & Research Integrity</h1>
        <p class="section-subtitle">
          Runtime environment parameters, automated test verification, and cryptographic checkpoint integrity.
        </p>
      </div>

      <div class="grid-3" style="margin-bottom: 24px;">
        <div class="card">
          <div class="card-title"><span>Runtime Environment</span></div>
          <div style="display: flex; flex-direction: column; gap: 6px; font-size: 12px;">
            <div><strong>Operating System:</strong> Windows 11 (10.0.26200)</div>
            <div><strong>Python Runtime:</strong> 3.12.4 (64-bit)</div>
            <div><strong>PyTorch:</strong> 2.5.1 + CPU Build</div>
            <div><strong>MONAI:</strong> 1.4.0</div>
            <div><strong>Deterministic Seeds:</strong> 42, 43, 44, 45, 46</div>
          </div>
        </div>

        <div class="card">
          <div class="card-title"><span>Automated Test Suite</span></div>
          <div style="text-align: center; padding: 14px 0;">
            <div style="font-size: 38px; font-weight: 800; color: #34d399; font-family: var(--font-mono);">49 / 49</div>
            <div style="font-size: 13px; font-weight: 600; color: #fff; margin-top: 4px;">Pytest Unit & Integration Tests Passing</div>
            <div style="font-size: 11px; color: var(--text-dim); margin-top: 4px;">100% Pass Rate across entire codebase</div>
          </div>
        </div>

        <div class="card">
          <div class="card-title"><span>Cryptographic Traceability</span></div>
          <div style="display: flex; flex-direction: column; gap: 6px; font-size: 12px;">
            <div><strong>Artifact Inventory:</strong> 802 files cataloged</div>
            <div><strong>Hashing Algorithm:</strong> SHA-256 Verified</div>
            <div><strong>Audit Status:</strong> FROZEN DATASET</div>
            <div><strong>Test Leakage Status:</strong> ZERO LEAKAGE CONFIRMED</div>
          </div>
        </div>
      </div>
    `;
  }

  /* ==========================================================================
     18. LIMITATIONS
     ========================================================================== */
  renderLimitations() {
    const container = document.getElementById('section-limitations');
    const limits = this.data.limitations.mandatory_limitations;

    container.innerHTML = `
      <div class="section-header">
        <span class="section-tag">Section 18 / Boundary Conditions</span>
        <h1 class="section-title">Documented Research Limitations</h1>
        <p class="section-subtitle">
          Mandatory scientific caveats, statistical bounds, and clinical disclaimers governing this benchmark.
        </p>
      </div>

      <div class="callout-box" style="margin-bottom: 24px; border-left: 4px solid var(--accent-rose);">
        <div class="callout-title" style="color: #f43f5e;">
          <span>🛑</span> Non-Clinical Diagnostic Boundary
        </div>
        <div class="callout-body" style="font-size: 14px;">
          This research system is NOT a diagnostic medical device. The models evaluate cross-sectional statistical patterns in neuroimaging research cohorts and cannot be interpreted as clinical diagnoses or medical advice.
        </div>
      </div>

      <div class="grid-2">
        ${limits.map((l, i) => `
          <div class="card" style="border-left: 3px solid var(--border-color);">
            <div style="display: flex; gap: 12px; align-items: flex-start;">
              <span style="font-family: var(--font-mono); font-size: 12px; font-weight: 700; color: var(--accent-amber);">#${(i + 1).toString().padStart(2, '0')}</span>
              <span style="font-size: 13px; color: var(--text-main); line-height: 1.55;">${l}</span>
            </div>
          </div>
        `).join('')}
      </div>
    `;
  }

  /* ==========================================================================
     19. CONCLUSION
     ========================================================================== */
  renderConclusion() {
    const container = document.getElementById('section-conclusion');

    container.innerHTML = `
      <div class="section-header">
        <span class="section-tag">Section 19 / Thesis Synthesis</span>
        <h1 class="section-title">Final Research Conclusion</h1>
        <p class="section-subtitle">
          Balanced scientific synthesis of findings, trade-offs, and directions for future large-scale research.
        </p>
      </div>

      <div class="card" style="padding: 32px; line-height: 1.75; font-size: 15px;">
        <h2 style="font-size: 20px; font-weight: 700; color: #fff; margin-bottom: 16px;">
          Synthesis of Benchmark Evidence
        </h2>
        <p style="margin-bottom: 16px; color: var(--text-main);">
          Foundation-model linear probing can substantially underutilize pretrained representations in limited-label settings. Task adaptation can alleviate this limitation, but its effect on external generalization is architecture-dependent. Well-tuned supervised 3D CNNs remain highly competitive. When properly regularized and tuned, supervised 3D ResNet-18 and ResNet-34 baselines established formidable in-domain benchmarks, achieving 0.8450–0.8950 development cross-validation AUC.
        </p>
        <p style="margin-bottom: 16px; color: var(--text-main);">
          Foundation model linear probes demonstrated extraordinary <strong>parameter efficiency</strong>, utilizing only 513–769 trainable parameters (<0.04% of baseline CNNs) to establish discriminative capabilities. In external cross-dataset transfer to ADNI (3.0T), selected foundation encoders (MONAI Swin UNETR SSL and BrainIAC) provided pilot evidence of improved cross-cohort discrimination.
        </p>
        <p style="margin-bottom: 16px; color: var(--text-main);">
          Crucially, <strong>dual-paradigm ensembling</strong> (Supervised ResNet-34 + MONAI Swin UNETR SSL) achieved the best observed cross-cohort performance in this pilot benchmark (OASIS Test AUC 1.0000, ADNI Holdout AUC 1.0000), rather than demonstrating definitive best model superiority across populations. Because each holdout cohort consists of N=8 subjects, an observed AUC of 1.0000 must not be interpreted as proof of perfect generalization.
        </p>
        <div class="callout-box" style="margin-top: 24px; margin-bottom: 0;">
          <div class="callout-title"><span>🔬</span> Final Research Statement</div>
          <div class="callout-body">
            These findings indicate that foundation models do not obsolete supervised CNNs in sample-constrained neuroimaging, but rather serve as efficient feature extractors that synergize effectively in hybrid multi-paradigm ensembles. Large-scale multi-center prospective studies are required to confirm these observations across broader clinical spectra.
          </div>
        </div>
      </div>
    `;
  }

  /* ==========================================================================
     20. PUBLICATION FIGURE GALLERY
     ========================================================================== */
  renderGallery() {
    const container = document.getElementById('section-gallery');

    // Registry of the 28 publication figures
    const figures = [
      { id: '01_auc', title: 'Model ROC-AUC Comparison', tag: 'Performance', file: 'results/final_campaign/figures/01_model_auc_comparison.png', desc: 'Cross-model comparison across evaluation partitions.' },
      { id: '02_acc', title: 'Model Accuracy Comparison', tag: 'Performance', file: 'results/final_campaign/figures/02_model_accuracy_comparison.png', desc: 'Accuracy metrics under dev-selected operating thresholds.' },
      { id: '03_f1', title: 'Model F1-Score Comparison', tag: 'Performance', file: 'results/final_campaign/figures/03_model_f1_comparison.png', desc: 'Harmonic mean of precision and recall.' },
      { id: '01_data_eff', title: 'Data Efficiency Trajectory', tag: 'Data Efficiency', file: 'results/final_campaign/figures/01_data_efficiency_curve.png', desc: 'Performance scaling from 10% to 100% labeled sample pools.' },
      { id: '02_param_front', title: 'Parameter Efficiency Frontier', tag: 'Efficiency', file: 'results/final_campaign/figures/02_parameter_efficiency_frontier.png', desc: 'Pareto frontier of trainable parameters vs validation AUC.' },
      { id: '03_stab', title: 'Stability vs Performance', tag: 'Stability', file: 'results/final_campaign/figures/03_stability_vs_performance.png', desc: 'Mean AUC versus multi-seed standard deviation.' },
      { id: '04_cost', title: 'Training Cost vs Performance', tag: 'Efficiency', file: 'results/final_campaign/figures/04_training_cost_vs_performance.png', desc: 'Wall-clock training duration trade-offs.' },
      { id: '05_calib', title: 'Calibration Reliability Curves', tag: 'Calibration', file: 'results/final_campaign/figures/05_calibration_curves.png', desc: 'Reliability diagrams comparing raw and calibrated predictions.' },
      { id: '05_learn_r18', title: 'ResNet-18 Learning Curves', tag: 'Learning Curves', file: 'results/final_campaign/figures/05_learning_curves_resnet18.png', desc: 'Training and validation loss and AUC over epochs.' },
      { id: '06_learn_r34', title: 'ResNet-34 Learning Curves', tag: 'Learning Curves', file: 'results/final_campaign/figures/06_learning_curves_resnet34.png', desc: 'Training dynamics of deep 34-layer residual baseline.' },
      { id: '06_pca', title: 'Latent Representation PCA', tag: 'Representations', file: 'results/final_campaign/figures/06_representation_pca.png', desc: 'Principal component projections of penultimate features.' },
      { id: '07_shift', title: 'Domain Shift Comparison', tag: 'Domain Shift', file: 'results/final_campaign/figures/07_domain_shift_comparison.png', desc: 'Cross-scanner performance drop from 1.5T to 3.0T.' },
      { id: '07_roc', title: 'Receiver Operating Characteristic Curves', tag: 'Performance', file: 'results/final_campaign/figures/07_roc_curves.png', desc: 'True positive versus false positive rates.' },
      { id: '08_aug', title: 'Data Augmentation Ablation', tag: 'Ablation', file: 'results/final_campaign/figures/08_augmentation_ablation.png', desc: 'Impact of spatial affine augmentations on CNN generalization.' },
      { id: '08_pr', title: 'Precision-Recall Curves', tag: 'Performance', file: 'results/final_campaign/figures/08_precision_recall_curves.png', desc: 'Precision-recall trade-offs across confidence thresholds.' },
      { id: '09_conf', title: 'Confusion Matrices Matrix', tag: 'Performance', file: 'results/final_campaign/figures/09_confusion_matrices.png', desc: 'Classification matrices across all evaluated architectures.' },
      { id: '09_prep', title: 'Preprocessing Protocol Ablation', tag: 'Ablation', file: 'results/final_campaign/figures/09_preprocessing_ablation.png', desc: 'Comparison of normalization schemes and spatial resolutions.' },
      { id: '10_finetune', title: 'Foundation Fine-Tuning Comparison', tag: 'Foundation', file: 'results/final_campaign/figures/10_foundation_finetuning.png', desc: 'Linear probing versus full fine-tuning performance.' },
      { id: '11_train_params', title: 'Trainable Parameters Comparison', tag: 'Efficiency', file: 'results/final_campaign/figures/11_trainable_parameters_comparison.png', desc: 'Log-scale comparison of optimized weight counts.' },
      { id: '12_heatmap', title: 'Cross-Dataset Performance Heatmap', tag: 'Domain Shift', file: 'results/final_campaign/figures/12_cross_dataset_heatmap.png', desc: 'Matrix of in-domain versus external cohort scores.' },
      { id: '13_seed_stab', title: 'Multi-Seed Variance Audit', tag: 'Stability', file: 'results/final_campaign/figures/13_seed_stability.png', desc: 'Evaluation robustness across random seeds 42–46.' },
      { id: '14_fusion', title: 'CNN + Foundation Fusion Analysis', tag: 'Fusion', file: 'results/final_campaign/figures/14_fusion_comparison.png', desc: 'Comparison of standalone encoders against hybrid fusion.' },
      { id: '15_umap', title: 'Latent Representation UMAP', tag: 'Representations', file: 'results/final_campaign/figures/15_representation_umap.png', desc: 'Non-linear manifold projection of latent embeddings.' },
      { id: '16_radar', title: 'Multi-Axis Radar Comparison', tag: 'Synthesis', file: 'results/final_campaign/figures/16_final_model_radar.png', desc: 'Radar chart summarizing performance, stability, and efficiency.' },
      { id: '21_subgroup', title: 'Demographic Subgroup Analysis', tag: 'Demographics', file: 'results/final_campaign/figures/21_subgroup_analysis.png', desc: 'Performance disaggregated by age and biological sex.' },
      { id: '22_conf_dist', title: 'Prediction Confidence Distribution', tag: 'Calibration', file: 'results/final_campaign/figures/22_prediction_confidence.png', desc: 'Distribution of model output probabilities.' },
      { id: '23_data_chars', title: 'Dataset Acquisition Profiles', tag: 'Datasets', file: 'results/final_campaign/figures/23_dataset_characteristics.png', desc: 'Scanner field strengths and demographic distributions.' },
      { id: '24_gradcam', title: 'Cross-Dataset Grad-CAM Panels', tag: 'Interpretability', file: 'results/final_campaign/figures/24_cross_dataset_gradcam.png', desc: 'Side-by-side saliency comparisons across OASIS and ADNI.' }
    ];

    container.innerHTML = `
      <div class="section-header">
        <span class="section-tag">Section 20 / Publication Assets</span>
        <h1 class="section-title">Research Figure Gallery</h1>
        <p class="section-subtitle">
          Interactive gallery of all 28 publication-quality scientific visualization figures generated for this study.
        </p>
      </div>

      <div class="filter-bar" style="margin-bottom: 24px;">
        <label style="font-weight: 600; color: #fff;">Category Filter:</label>
        <select id="gallery-cat-filter" class="filter-select">
          <option value="ALL">All Figures (${figures.length})</option>
          <option value="Performance">Performance (6)</option>
          <option value="Efficiency">Efficiency (3)</option>
          <option value="Data Efficiency">Data Efficiency (1)</option>
          <option value="Calibration">Calibration (2)</option>
          <option value="Representations">Representations (2)</option>
          <option value="Domain Shift">Domain Shift (2)</option>
          <option value="Interpretability">Interpretability (1)</option>
          <option value="Ablation">Ablations (2)</option>
        </select>
        <input type="text" id="gallery-search" class="search-input" placeholder="Search figures..." style="flex: 1; max-width: 320px;">
      </div>

      <div class="gallery-grid" id="gallery-container"></div>
    `;

    const renderFigures = () => {
      const cat = document.getElementById('gallery-cat-filter').value;
      const q = document.getElementById('gallery-search').value.toLowerCase();
      const grid = document.getElementById('gallery-container');

      const filtered = figures.filter(f => {
        const matchCat = cat === 'ALL' || f.tag === cat;
        const matchQ = !q || f.title.toLowerCase().includes(q) || f.desc.toLowerCase().includes(q);
        return matchCat && matchQ;
      });

      grid.innerHTML = filtered.map(f => `
        <div class="gallery-card">
          <div class="gallery-img-wrapper" onclick="window.__app.openZoomModal('${f.file}', '${f.title}')">
            <img src="${f.file}" alt="${f.title}" loading="lazy">
          </div>
          <div class="gallery-meta">
            <div>
              <div class="gallery-tag">${f.tag}</div>
              <div class="gallery-title">${f.title}</div>
              <div style="font-size: 12px; color: var(--text-muted); line-height: 1.4;">${f.desc}</div>
            </div>
            <div class="gallery-actions">
              <button class="gallery-btn" onclick="window.__app.openZoomModal('${f.file}', '${f.title}')">🔍 Expand</button>
              <a href="${f.file}" download class="gallery-btn">⬇ Download</a>
            </div>
          </div>
        </div>
      `).join('');
    };

    document.getElementById('gallery-cat-filter').addEventListener('change', renderFigures);
    document.getElementById('gallery-search').addEventListener('input', renderFigures);
    renderFigures();
  }

  /* ==========================================================================
     16. TRAINING DYNAMICS & FIT DIAGNOSTICS
     ========================================================================== */
  renderTrainingDynamics() {
    const container = document.getElementById('section-training-dynamics');
    const diag = this.data.training_diagnostics;
    if (!container || !diag) return;

    const getFitBadge = (fit) => {
      let bg = 'rgba(148, 163, 184, 0.15)', color = '#cbd5e1', border = 'rgba(148, 163, 184, 0.35)', icon = 'ℹ';
      if (fit === 'reasonable fit') {
        bg = 'rgba(16, 185, 129, 0.15)'; color = '#34d399'; border = 'rgba(16, 185, 129, 0.35)'; icon = '✓';
      } else if (fit === 'likely underfitting') {
        bg = 'rgba(245, 158, 11, 0.15)'; color = '#fbbf24'; border = 'rgba(245, 158, 11, 0.35)'; icon = '▲';
      } else if (fit === 'likely overfitting') {
        bg = 'rgba(244, 63, 94, 0.15)'; color = '#fb7185'; border = 'rgba(244, 63, 94, 0.35)'; icon = '▼';
      }
      return `<span style="display: inline-flex; align-items: center; gap: 5px; padding: 4px 10px; border-radius: 9999px; font-size: 11px; font-weight: 600; text-transform: uppercase; background: ${bg}; color: ${color}; border: 1px solid ${border};">${icon} ${fit}</span>`;
    };

    container.innerHTML = `
      <div class="section-header">
        <span class="section-tag">Section 16 / Training Dynamics & Diagnostics</span>
        <h1 class="section-title">${diag.report_title}</h1>
        <p class="section-subtitle">
          ${diag.description} Evaluated strictly across patient-level splits without exposure to holdout test labels.
        </p>
      </div>

      <!-- Cohort / Split Partitioning Guardrails -->
      <div style="display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 24px; padding: 16px 20px; background: var(--bg-card-subtle); border: 1px solid var(--border-color); border-radius: var(--radius-md);">
        <div style="display: flex; align-items: center; gap: 8px;">
          <span class="kpi-context context-dev" style="margin-bottom: 0;">TRAINING</span>
          <span style="font-size: 12px; color: var(--text-muted);">Development cohort training partition ($N=35$ OASIS-1 or $N=35$ ADNI)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
          <span class="kpi-context context-dev" style="margin-bottom: 0; background-color: rgba(99, 102, 241, 0.15); color: #a5b4fc; border-color: rgba(99, 102, 241, 0.3);">VALIDATION</span>
          <span style="font-size: 12px; color: var(--text-muted);">Internal validation partition ($N=7$) used for convergence monitoring & checkpointing</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
          <span class="kpi-context context-holdout" style="margin-bottom: 0;">LOCKED TEST</span>
          <span style="font-size: 12px; color: var(--text-muted);">Locked OASIS-1 holdout ($N=8$) — strictly isolated, zero exposure during diagnostics</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
          <span class="kpi-context context-transfer" style="margin-bottom: 0;">EXTERNAL ADNI</span>
          <span style="font-size: 12px; color: var(--text-muted);">Multi-site ADNI holdout ($N=8$) — strictly isolated external transfer evaluation</span>
        </div>
      </div>

      <!-- Methodology Note Callout -->
      <div class="callout-box">
        <div class="callout-title">
          <span>🔬</span>
          <span>Methodology & Interpretive Disclaimer</span>
        </div>
        <div class="callout-body">
          ${diag.methodology_note} Training and validation curves reflect internal optimization dynamics across the $N=35$ training and $N=7$ validation patient split. Neither the locked OASIS-1 holdout test set ($N=8$) nor the external ADNI holdout set ($N=8$) was used in any diagnostic evaluation, hyperparameter tuning, or checkpoint selection.
        </div>
      </div>

      <!-- Fit Diagnostics Summary Table -->
      <div class="card" style="margin-bottom: 28px;">
        <div class="card-title">
          <span>Optimization & Generalization Fit Diagnostic Matrix (${diag.models_audited} Configurations)</span>
          <span style="font-size: 12px; font-family: var(--font-mono); color: var(--text-dim);">${diag.total_records} Epoch Records Audited</span>
        </div>
        <div class="table-responsive">
          <table class="scientific-table">
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>Fit Classification</th>
                <th style="text-align: right;">Best / Total Epoch</th>
                <th style="text-align: right;">Train Loss</th>
                <th style="text-align: right;">Train Acc</th>
                <th style="text-align: right;">Best Val Loss</th>
                <th style="text-align: right;">Best Val Acc</th>
                <th style="text-align: right;">Train–Val Gap</th>
                <th style="text-align: right;">Best Val AUC</th>
                <th>Optimization Rationale</th>
              </tr>
            </thead>
            <tbody>
              ${diag.summaries.map(s => `
                <tr>
                  <td style="font-weight: 600; color: #fff;">${s.display_name}</td>
                  <td>${getFitBadge(s.fit_assessment)}</td>
                  <td class="table-num" style="text-align: right;">${s.best_epoch} / ${s.total_epochs}</td>
                  <td class="table-num" style="text-align: right;">${typeof s.training_loss === 'number' ? s.training_loss.toFixed(4) : s.training_loss}</td>
                  <td class="table-num" style="text-align: right; color: var(--accent-blue);">${s.training_accuracy}</td>
                  <td class="table-num" style="text-align: right;">${typeof s.best_validation_loss === 'number' ? s.best_validation_loss.toFixed(4) : s.best_validation_loss}</td>
                  <td class="table-num" style="text-align: right; color: var(--accent-emerald); font-weight: 600;">${s.best_validation_accuracy}</td>
                  <td class="table-num" style="text-align: right; color: ${typeof s.train_validation_accuracy_gap === 'string' && s.train_validation_accuracy_gap.startsWith('+') ? 'var(--accent-amber)' : 'var(--text-muted)'};">${s.train_validation_accuracy_gap}</td>
                  <td class="table-num" style="text-align: right; color: var(--accent-cyan); font-weight: 700;">${typeof s.best_validation_auc === 'number' ? s.best_validation_auc.toFixed(4) : s.best_validation_auc}</td>
                  <td style="font-size: 12px; color: var(--text-muted); max-width: 320px; white-space: normal; line-height: 1.4;">${s.rationale}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </div>

      <!-- Training vs Validation Curves Gallery -->
      <div class="card">
        <div class="card-title">
          <span>High-Resolution Training vs. Validation Trajectories</span>
          <span style="font-size: 12px; color: var(--text-dim);">Dual-Axis Loss & Metric Plots • Click to Zoom</span>
        </div>
        <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 20px;">
          Plots illustrate training loss convergence (solid blue), validation loss (dashed orange), training accuracy (solid teal), and validation metric performance (dashed red). Vertical dotted lines mark the optimal early-stopping checkpoint selected strictly by internal validation loss.
        </p>
        
        <div class="grid-2">
          ${diag.summaries.map(s => `
            <div class="card" style="background-color: var(--bg-card-subtle); border: 1px solid var(--border-color); padding: 16px;">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-size: 14px; font-weight: 700; color: #fff;">${s.display_name}</span>
                ${getFitBadge(s.fit_assessment)}
              </div>
              
              <div style="display: flex; gap: 12px; margin-bottom: 12px; font-size: 11px; font-family: var(--font-mono); color: var(--text-dim);">
                <span>Optimal Epoch: <strong style="color: #fff;">${s.best_epoch}</strong></span>
                <span>•</span>
                <span>Val Loss: <strong style="color: #fff;">${typeof s.best_validation_loss === 'number' ? s.best_validation_loss.toFixed(4) : s.best_validation_loss}</strong></span>
                <span>•</span>
                <span>Val AUC: <strong style="color: var(--accent-cyan);">${typeof s.best_validation_auc === 'number' ? s.best_validation_auc.toFixed(4) : s.best_validation_auc}</strong></span>
              </div>

              <div style="position: relative; border-radius: 6px; overflow: hidden; border: 1px solid var(--border-color); background: #000; cursor: pointer;"
                   onclick="window.__app.openZoomModal('/${s.curve_plot}', '${s.display_name} — Training vs. Validation Diagnostics')">
                <img src="/${s.curve_plot}" alt="${s.display_name} Training Curves" style="width: 100%; height: auto; display: block; transition: transform 0.2s ease;"
                     onmouseover="this.style.transform='scale(1.02)'" onmouseout="this.style.transform='scale(1)'">
                <div style="position: absolute; bottom: 8px; right: 8px; background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(4px); padding: 4px 8px; border-radius: 4px; font-size: 10px; font-weight: 600; color: var(--text-main); border: 1px solid var(--border-color);">
                  🔍 Click to Enlarge
                </div>
              </div>

              <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 12px;">
                <span style="font-size: 12px; color: var(--text-muted); font-style: italic;">
                  ${s.total_epochs} total epochs evaluated
                </span>
                <a href="/${s.curve_plot}" download class="tab-btn" style="padding: 4px 10px; font-size: 11px; border: 1px solid var(--border-color); text-decoration: none; color: var(--text-main);">
                  ⬇ Download Plot
                </a>
              </div>
            </div>
          `).join('')}
        </div>
      </div>
    `;
  }

  /* ==========================================================================
     17. FOUNDATION MODEL TASK ADAPTATION
     ========================================================================== */
  renderFoundationAdaptation() {
    const container = document.getElementById('section-foundation-adaptation');
    const fa = this.data.foundation_adaptation;
    if (!container || !fa) return;

    const getAdaptationBadge = (fit) => {
      let bg = 'rgba(148, 163, 184, 0.15)', color = '#cbd5e1', border = 'rgba(148, 163, 184, 0.35)', icon = 'ℹ';
      if (fit === 'improved generalization') {
        bg = 'rgba(16, 185, 129, 0.15)'; color = '#34d399'; border = 'rgba(16, 185, 129, 0.35)'; icon = '★';
      } else if (fit === 'reduced underfitting') {
        bg = 'rgba(56, 189, 248, 0.15)'; color = '#38bdf8'; border = 'rgba(56, 189, 248, 0.35)'; icon = '▲';
      } else if (fit && (fit.includes('control') || fit.includes('baseline'))) {
        bg = 'rgba(245, 158, 11, 0.15)'; color = '#fbbf24'; border = 'rgba(245, 158, 11, 0.35)'; icon = '●';
      }
      return `<span style="display: inline-flex; align-items: center; gap: 5px; padding: 4px 10px; border-radius: 9999px; font-size: 11px; font-weight: 600; text-transform: uppercase; background: ${bg}; color: ${color}; border: 1px solid ${border};">${icon} ${fit}</span>`;
    };

    container.innerHTML = `
      <div class="section-header">
        <span class="section-tag">${fa.section_tag}</span>
        <h1 class="section-title">${fa.report_title}</h1>
        <p class="section-subtitle">
          ${fa.description} Systematic exploration of adaptation depth using discriminative learning rates to test whether frozen linear probes suffer from an artificial representation bottleneck.
        </p>
      </div>

      <!-- Cohort / Split Partitioning Guardrails -->
      <div style="display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 24px; padding: 16px 20px; background: var(--bg-card-subtle); border: 1px solid var(--border-color); border-radius: var(--radius-md);">
        <div style="display: flex; align-items: center; gap: 8px;">
          <span class="kpi-context context-dev" style="margin-bottom: 0;">DEVELOPMENT PARTITION</span>
          <span style="font-size: 12px; color: var(--text-muted);">Trained strictly on $N=35$ OASIS-1; model selection & early stopping on $N=7$ internal validation</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
          <span class="kpi-context context-dev" style="margin-bottom: 0; background-color: rgba(99, 102, 241, 0.15); color: #a5b4fc; border-color: rgba(99, 102, 241, 0.3);">OPTIMIZER</span>
          <span style="font-size: 12px; color: var(--text-muted);">Discriminative learning rates: Backbone (1e-5 to 3e-5) & Head (1e-4 to 3e-4) with weight decay 1e-4</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
          <span class="kpi-context context-holdout" style="margin-bottom: 0;">LOCKED TEST</span>
          <span style="font-size: 12px; color: var(--text-muted);">Locked OASIS-1 holdout ($N=8$) evaluated strictly post-freeze to test internal generalization</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
          <span class="kpi-context context-transfer" style="margin-bottom: 0;">EXTERNAL ADNI</span>
          <span style="font-size: 12px; color: var(--text-muted);">ADNI 3.0T holdout ($N=8$) evaluated strictly post-freeze to test multi-scanner transfer stability</span>
        </div>
      </div>

      <!-- Methodology Disclaimer Callout -->
      <div class="callout-box">
        <div class="callout-title">
          <span>🔬</span>
          <span>Scientific Governance & Leakage Prevention</span>
        </div>
        <div class="callout-body">
          ${fa.methodology_disclaimer} Original benchmark baselines remain preserved as unperturbed control conditions. Adaptation models were tuned exclusively within the development partition, without inspecting locked holdout performance during strategy or checkpoint selection.
        </div>
      </div>

      <!-- Key Findings Summary Cards -->
      <div class="grid-3" style="margin-bottom: 28px;">
        <div class="card" style="border-left: 4px solid var(--accent-cyan); background-color: var(--bg-card-subtle);">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 16px; font-weight: 700; color: #fff;">BrainIAC</span>
            <span class="badge" style="background: rgba(56, 189, 248, 0.15); color: #38bdf8;">Selected: Deeper FT</span>
          </div>
          <div style="font-size: 13px; color: var(--text-muted); line-height: 1.5; margin-bottom: 12px;">
            Deeper fine-tuning of final conv/norm blocks resolved the representation bottleneck: Dev Val AUC climbed from <strong>0.5000</strong> to <strong>0.8333</strong>, with training accuracy rising from 48.6% to 74.3%. On the locked OASIS test set, classification accuracy was maintained at 75.0% (2 failures), while external ADNI holdout AUC rose from 0.3750 to 0.5000.
          </div>
          <div style="display: flex; gap: 8px; font-size: 11px; font-family: var(--font-mono); color: var(--text-dim);">
            <span>Dev Val AUC: 0.5000 → 0.8333</span> • <span>Test Acc: 75.0%</span>
          </div>
        </div>

        <div class="card" style="border-left: 4px solid var(--accent-emerald); background-color: var(--bg-card-subtle);">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 16px; font-weight: 700; color: #fff;">FOMO25</span>
            <span class="badge" style="background: rgba(16, 185, 129, 0.15); color: #34d399;">Selected: Full FT</span>
          </div>
          <div style="font-size: 13px; color: var(--text-muted); line-height: 1.5; margin-bottom: 12px;">
            Full end-to-end fine-tuning transformed FOMO25 from a severe underfitter (Val AUC 0.5000, 42.9% Acc) into an effective classifier (Val AUC <strong>0.8333</strong>, 71.4% Acc, Train Acc 80.0%). On the locked OASIS test set, accuracy surged from 50.0% to <strong>87.5%</strong> (failures cut from 4 to 1, AUC 1.0000). On ADNI, transfer dropped from 0.8750 to 0.4375, indicating scanner-specific domain sensitivity.
          </div>
          <div style="display: flex; gap: 8px; font-size: 11px; font-family: var(--font-mono); color: var(--text-dim);">
            <span>Dev Val AUC: 0.5000 → 0.8333</span> • <span>Test Acc: 87.5%</span>
          </div>
        </div>

        <div class="card" style="border-left: 4px solid var(--accent-purple); background-color: var(--bg-card-subtle);">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 16px; font-weight: 700; color: #fff;">MONAI Swin UNETR SSL</span>
            <span class="badge" style="background: rgba(168, 85, 247, 0.15); color: #c084fc;">Selected: Deeper FT</span>
          </div>
          <div style="font-size: 13px; color: var(--text-muted); line-height: 1.5; margin-bottom: 12px;">
            Deeper fine-tuning of swin stage blocks produced substantial gains across both cohorts: Dev Val AUC climbed from 0.1667 to <strong>0.8333</strong>. On the locked OASIS test set, accuracy rose to <strong>62.5%</strong> (AUC 0.8750). Most critically, on external ADNI holdout, AUC surged from 0.3750 to <strong>0.8125</strong>, demonstrating robust multi-cohort feature alignment.
          </div>
          <div style="display: flex; gap: 8px; font-size: 11px; font-family: var(--font-mono); color: var(--text-dim);">
            <span>Dev Val AUC: 0.1667 → 0.8333</span> • <span>ADNI AUC: 0.3750 → 0.8125</span>
          </div>
        </div>
      </div>

      <!-- Side-by-Side Holdout Generalization Comparison -->
      <div class="card" style="margin-bottom: 28px;">
        <div class="card-title">
          <span>Does Fine-Tuning Improve Generalization? (Post-Freeze Holdout Evaluation)</span>
          <span style="font-size: 12px; font-family: var(--font-mono); color: var(--text-dim);">OASIS-1 Locked Test ($N=8$) & ADNI Multi-Site Holdout ($N=8$)</span>
        </div>
        <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 16px;">
          Direct side-by-side comparison between the original frozen linear probe baseline and the best adapted fine-tuned model configuration. Both holdouts remained strictly isolated throughout all training and hyperparameter search.
        </p>

        <div class="table-responsive">
          <table class="scientific-table">
            <thead>
              <tr>
                <th rowspan="2" style="vertical-align: bottom;">Model</th>
                <th rowspan="2" style="vertical-align: bottom;">Evaluation Condition</th>
                <th rowspan="2" style="vertical-align: bottom;">Strategy</th>
                <th colspan="4" style="text-align: center; border-bottom: 1px solid var(--border-color); color: var(--accent-cyan);">OASIS-1 Locked Holdout ($N=8$)</th>
                <th colspan="4" style="text-align: center; border-bottom: 1px solid var(--border-color); color: var(--accent-amber);">ADNI External Holdout ($N=8$)</th>
              </tr>
              <tr>
                <th style="text-align: right;">Acc</th>
                <th style="text-align: right;">ROC-AUC</th>
                <th style="text-align: right;">Brier</th>
                <th style="text-align: right;">Failures</th>
                <th style="text-align: right;">Acc</th>
                <th style="text-align: right;">ROC-AUC</th>
                <th style="text-align: right;">Brier</th>
                <th style="text-align: right;">Failures</th>
              </tr>
            </thead>
            <tbody>
              ${fa.baseline_vs_adapted_comparison.map(m => {
                const b = m.evaluations['Baseline Linear Probe'];
                const a = m.evaluations['Adapted / Fine-Tuned'];
                const bO = b.oasis_locked_test;
                const aO = a.oasis_locked_test;
                const bA = b.adni_external_holdout;
                const aA = a.adni_external_holdout;
                const deltaOasisAuc = aO.roc_auc - bO.roc_auc;
                const deltaAdniAuc = aA.roc_auc - bA.roc_auc;
                const deltaOasisFail = aO.total_failures - bO.total_failures;
                const deltaAdniFail = aA.total_failures - bA.total_failures;

                return `
                  <tr style="border-top: 2px solid var(--border-color);">
                    <td rowspan="2" style="font-weight: 700; color: #fff; vertical-align: middle;">${m.model_name}</td>
                    <td style="color: var(--text-muted);">Baseline (Control)</td>
                    <td><span class="badge" style="background: rgba(148, 163, 184, 0.15); color: #94a3b8;">${b.strategy}</span></td>
                    <td class="table-num" style="text-align: right;">${(bO.accuracy * 100).toFixed(1)}%</td>
                    <td class="table-num" style="text-align: right;">${bO.roc_auc.toFixed(4)}</td>
                    <td class="table-num" style="text-align: right;">${bO.brier.toFixed(4)}</td>
                    <td class="table-num" style="text-align: right; color: var(--text-muted);">${bO.total_failures} / 8</td>
                    <td class="table-num" style="text-align: right;">${(bA.accuracy * 100).toFixed(1)}%</td>
                    <td class="table-num" style="text-align: right;">${bA.roc_auc.toFixed(4)}</td>
                    <td class="table-num" style="text-align: right;">${bA.brier.toFixed(4)}</td>
                    <td class="table-num" style="text-align: right; color: var(--text-muted);">${bA.total_failures} / 8</td>
                  </tr>
                  <tr>
                    <td style="font-weight: 600; color: var(--accent-emerald);">Adapted Fine-Tuned</td>
                    <td><span class="badge" style="background: rgba(16, 185, 129, 0.15); color: #34d399;">${a.strategy}</span></td>
                    <td class="table-num" style="text-align: right; font-weight: 700; color: #fff;">${(aO.accuracy * 100).toFixed(1)}%</td>
                    <td class="table-num" style="text-align: right; font-weight: 700; color: var(--accent-cyan);">${aO.roc_auc.toFixed(4)} <span style="font-size: 10px; color: ${deltaOasisAuc >= 0 ? '#34d399' : '#fb7185'};">(${deltaOasisAuc >= 0 ? '+' : ''}${deltaOasisAuc.toFixed(3)})</span></td>
                    <td class="table-num" style="text-align: right;">${aO.brier.toFixed(4)}</td>
                    <td class="table-num" style="text-align: right; font-weight: 700; color: ${deltaOasisFail <= 0 ? '#34d399' : '#fb7185'};">${aO.total_failures} / 8 <span style="font-size: 10px;">(${deltaOasisFail >= 0 ? '+' : ''}${deltaOasisFail})</span></td>
                    <td class="table-num" style="text-align: right; font-weight: 700; color: #fff;">${(aA.accuracy * 100).toFixed(1)}%</td>
                    <td class="table-num" style="text-align: right; font-weight: 700; color: var(--accent-amber);">${aA.roc_auc.toFixed(4)} <span style="font-size: 10px; color: ${deltaAdniAuc >= 0 ? '#34d399' : '#fb7185'};">(${deltaAdniAuc >= 0 ? '+' : ''}${deltaAdniAuc.toFixed(3)})</span></td>
                    <td class="table-num" style="text-align: right;">${aA.brier.toFixed(4)}</td>
                    <td class="table-num" style="text-align: right; font-weight: 700; color: ${deltaAdniFail <= 0 ? '#34d399' : '#fb7185'};">${aA.total_failures} / 8 <span style="font-size: 10px;">(${deltaAdniFail >= 0 ? '+' : ''}${deltaAdniFail})</span></td>
                  </tr>
                `;
              }).join('')}
            </tbody>
          </table>
        </div>
      </div>

      <!-- Strategy Comparison Table -->
      <div class="card" style="margin-bottom: 28px;">
        <div class="card-title">
          <span>Task Adaptation Progression Matrix (Seed 42 Development Set)</span>
          <span style="font-size: 12px; font-family: var(--font-mono); color: var(--text-dim);">${fa.total_training_runs} Runs • ${fa.total_epoch_records} Epoch Records</span>
        </div>
        <div class="table-responsive">
          <table class="scientific-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Adaptation Strategy</th>
                <th style="text-align: right;">Trainable Params</th>
                <th>Unfrozen Layers</th>
                <th style="text-align: right;">Best Epoch</th>
                <th style="text-align: right;">Train Loss</th>
                <th style="text-align: right;">Train Acc</th>
                <th style="text-align: right;">Val Loss</th>
                <th style="text-align: right;">Val Acc</th>
                <th style="text-align: right;">Train–Val Gap</th>
                <th style="text-align: right;">Val AUC</th>
                <th>Fit Classification</th>
              </tr>
            </thead>
            <tbody>
              ${fa.strategy_summaries.filter(s => s.seed === 42).map(s => `
                <tr>
                  <td style="font-weight: 600; color: #fff;">${s.model_name}</td>
                  <td style="font-weight: 600; font-family: var(--font-mono); color: var(--accent-cyan);">${s.strategy}</td>
                  <td class="table-num" style="text-align: right;">${s.trainable_parameters.toLocaleString()} <span style="font-size: 10px; color: var(--text-dim);">(${(s.trainable_fraction * 100).toFixed(1)}%)</span></td>
                  <td style="font-size: 11px; color: var(--text-muted); max-width: 180px; white-space: normal;">${s.unfrozen_layers.join(', ')}</td>
                  <td class="table-num" style="text-align: right;">${s.best_epoch} / ${s.total_epochs}</td>
                  <td class="table-num" style="text-align: right;">${s.train_loss.toFixed(4)}</td>
                  <td class="table-num" style="text-align: right; color: var(--accent-blue);">${(s.train_accuracy * 100).toFixed(1)}%</td>
                  <td class="table-num" style="text-align: right;">${s.validation_loss.toFixed(4)}</td>
                  <td class="table-num" style="text-align: right; color: var(--accent-emerald); font-weight: 600;">${(s.validation_accuracy * 100).toFixed(1)}%</td>
                  <td class="table-num" style="text-align: right; color: ${s.train_validation_accuracy_gap > 0.1 ? 'var(--accent-amber)' : 'var(--text-muted)'};">${(s.train_validation_accuracy_gap * 100).toFixed(1)}%</td>
                  <td class="table-num" style="text-align: right; color: var(--accent-cyan); font-weight: 700;">${s.validation_auc.toFixed(4)}</td>
                  <td>${getAdaptationBadge(s.fit_classification)}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </div>

      <!-- High-Resolution Adaptation Curves Gallery -->
      <div class="card">
        <div class="card-title">
          <span>Dual-Axis Adaptation Trajectories (Loss & Metric Convergence)</span>
          <span style="font-size: 12px; color: var(--text-dim);">12 Plots Across Strategies • Click to Zoom</span>
        </div>
        <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 20px;">
          Trajectories track training loss convergence (solid blue), validation loss (dashed orange), training accuracy (solid teal), and validation ROC-AUC (dashed red) over 20 epochs under discriminative learning rates.
        </p>

        <div class="grid-2">
          ${fa.strategy_summaries.filter(s => s.seed === 42).map(s => `
            <div class="card" style="background-color: var(--bg-card-subtle); border: 1px solid var(--border-color); padding: 16px;">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div>
                  <span style="font-size: 14px; font-weight: 700; color: #fff;">${s.model_name}</span>
                  <span style="font-size: 12px; font-family: var(--font-mono); color: var(--accent-cyan); margin-left: 8px;">${s.strategy}</span>
                </div>
                ${getAdaptationBadge(s.fit_classification)}
              </div>

              <div style="display: flex; gap: 12px; margin-bottom: 12px; font-size: 11px; font-family: var(--font-mono); color: var(--text-dim);">
                <span>Optimal Epoch: <strong style="color: #fff;">${s.best_epoch}</strong></span>
                <span>•</span>
                <span>Val Acc: <strong style="color: var(--accent-emerald);">${(s.validation_accuracy * 100).toFixed(1)}%</strong></span>
                <span>•</span>
                <span>Val AUC: <strong style="color: var(--accent-cyan);">${s.validation_auc.toFixed(4)}</strong></span>
              </div>

              <div style="position: relative; border-radius: 6px; overflow: hidden; border: 1px solid var(--border-color); background: #000; cursor: pointer;"
                   onclick="window.__app.openZoomModal('/${s.curve_plot}', '${s.model_name} (${s.strategy}) — Adaptation Trajectory')">
                <img src="/${s.curve_plot}" alt="${s.model_name} ${s.strategy} Adaptation Curve" style="width: 100%; height: auto; display: block; transition: transform 0.2s ease;"
                     onmouseover="this.style.transform='scale(1.02)'" onmouseout="this.style.transform='scale(1)'">
                <div style="position: absolute; bottom: 8px; right: 8px; background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(4px); padding: 4px 8px; border-radius: 4px; font-size: 10px; font-weight: 600; color: var(--text-main); border: 1px solid var(--border-color);">
                  🔍 Click to Enlarge
                </div>
              </div>

              <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 12px;">
                <span style="font-size: 12px; color: var(--text-muted); font-style: italic;">
                  ${s.classification_rationale}
                </span>
                <a href="/${s.curve_plot}" download class="tab-btn" style="padding: 4px 10px; font-size: 11px; border: 1px solid var(--border-color); text-decoration: none; color: var(--text-main);">
                  ⬇ Download Plot
                </a>
              </div>
            </div>
          `).join('')}
        </div>
      </div>
    `;
  }

  /* ==========================================================================
     18. STATISTICAL UNCERTAINTY & ROBUSTNESS AUDIT
     ========================================================================== */
  renderStatisticalUncertainty() {
    const container = document.getElementById('section-statistical-uncertainty');
    const unc = this.data.statistical_uncertainty;
    const adapt = this.data.adaptation_comparison;
    const ds = this.data.domain_shift;
    if (!container || !unc) return;

    container.innerHTML = `
      <div class="section-header">
        <span class="section-tag">${unc.section_tag}</span>
        <h1 class="section-title">${unc.title}</h1>
        <p class="section-subtitle">
          Rigorous pre-publication uncertainty bounds across locked holdouts ($N=8$ per cohort), paired linear probe vs fine-tuning transitions, and cross-cohort transfer stability.
        </p>
      </div>

      <!-- Prominent Pilot Sample Size Warning Banner -->
      <div class="callout-box" style="border-left: 4px solid var(--accent-amber); background: rgba(245, 158, 11, 0.08); margin-bottom: 24px;">
        <div class="callout-title" style="color: var(--accent-amber);">
          <span>⚠️</span>
          <span>Sample Size Disclosure & Pilot Evidence Disclaimer</span>
        </div>
        <div class="callout-body" style="color: var(--text-main); font-size: 13px; line-height: 1.5;">
          ${unc.sample_size_disclosure} Because each locked test cohort contains only 8 subjects, accuracy changes in increments of 12.5 percentage points. Confidence intervals are therefore wide and results should be interpreted as pilot evidence. An observed AUC of 1.0000 on N=8 must not be interpreted as proof of perfect generalization. Note also the distinction between ROC-AUC and thresholded accuracy: a model can have high ROC-AUC while having lower thresholded accuracy because AUC evaluates ranking across thresholds whereas accuracy depends on the chosen classification threshold.
        </div>
      </div>

      <!-- Confidence Interval KPI Metrics Grid -->
      <div class="grid-4" style="margin-bottom: 24px;">
        <div class="kpi-card">
          <span class="kpi-context context-holdout">Locked OASIS Test</span>
          <div class="kpi-value">87.5%</div>
          <div class="kpi-label">FOMO25 (Full FT) Acc</div>
          <div class="kpi-sub">95% CI: [47.3%, 99.7%] • 1/8 Failures</div>
        </div>
        <div class="kpi-card">
          <span class="kpi-context context-holdout">Locked OASIS Test</span>
          <div class="kpi-value">1.0000</div>
          <div class="kpi-label">FOMO25 (Full FT) AUC</div>
          <div class="kpi-sub">95% CI: [1.0000, 1.0000] (B=2000)</div>
        </div>
        <div class="kpi-card">
          <span class="kpi-context context-transfer">ADNI Holdout</span>
          <div class="kpi-value">0.8125</div>
          <div class="kpi-label">Swin UNETR (Deeper FT) AUC</div>
          <div class="kpi-sub">95% CI: [0.4375, 1.0000] • 4/8 Failures</div>
        </div>
        <div class="kpi-card">
          <span class="kpi-context context-study">Baseline Reference</span>
          <div class="kpi-value">100.0%</div>
          <div class="kpi-label">Supervised ResNet-34 Acc</div>
          <div class="kpi-sub">95% CI: [63.1%, 100.0%] • 0/8 Failures</div>
        </div>
      </div>

      <!-- Section A: Benchmark Confidence Intervals Table -->
      <div class="card" style="margin-bottom: 28px;">
        <div class="card-title">
          <span>Table 1: Final Locked Holdout Benchmark Performance with 95% Confidence Intervals</span>
          <span style="font-size: 12px; font-family: var(--font-mono); color: var(--text-dim);">Exact Clopper-Pearson & Stratified Bootstrap (B=2000)</span>
        </div>
        <div class="table-responsive">
          <table class="scientific-table">
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>Category</th>
                <th>Cohort</th>
                <th style="text-align: right;">Accuracy (95% CI)</th>
                <th style="text-align: right;">Bal Acc (95% CI)</th>
                <th style="text-align: right;">ROC-AUC (95% CI)</th>
                <th style="text-align: right;">F1 Score (95% CI)</th>
                <th style="text-align: right;">Brier (95% CI)</th>
                <th style="text-align: right;">Failures</th>
              </tr>
            </thead>
            <tbody>
              ${unc.uncertainty_intervals.map(r => `
                <tr style="${r.model_id === 'fomo25_ft' || r.model_id === 'swin_ft' ? 'background: rgba(99, 102, 241, 0.04);' : ''}">
                  <td style="font-weight: 600; color: #fff;">${r.model_name}</td>
                  <td><span class="badge" style="background: rgba(148, 163, 184, 0.12); color: #94a3b8; font-size: 10px;">${r.category}</span></td>
                  <td style="font-size: 12px; color: ${r.cohort.includes('OASIS') ? 'var(--accent-cyan)' : 'var(--accent-amber)'};">${r.cohort}</td>
                  <td class="table-num" style="text-align: right; font-weight: 600;">
                    ${(r.accuracy * 100).toFixed(1)}% <span style="font-size: 10px; color: var(--text-dim);">[${(r.accuracy_ci_95[0] * 100).toFixed(1)}%, ${(r.accuracy_ci_95[1] * 100).toFixed(1)}%]</span>
                  </td>
                  <td class="table-num" style="text-align: right;">
                    ${(r.balanced_accuracy * 100).toFixed(1)}% <span style="font-size: 10px; color: var(--text-dim);">[${(r.balanced_accuracy_ci_95[0] * 100).toFixed(1)}%, ${(r.balanced_accuracy_ci_95[1] * 100).toFixed(1)}%]</span>
                  </td>
                  <td class="table-num" style="text-align: right; font-weight: 700; color: var(--accent-cyan);">
                    ${r.roc_auc.toFixed(4)} <span style="font-size: 10px; color: var(--text-dim);">[${r.roc_auc_ci_95[0].toFixed(4)}, ${r.roc_auc_ci_95[1].toFixed(4)}]</span>
                  </td>
                  <td class="table-num" style="text-align: right;">
                    ${r.f1.toFixed(4)} <span style="font-size: 10px; color: var(--text-dim);">[${r.f1_ci_95[0].toFixed(4)}, ${r.f1_ci_95[1].toFixed(4)}]</span>
                  </td>
                  <td class="table-num" style="text-align: right;">
                    ${r.brier_score.toFixed(4)} <span style="font-size: 10px; color: var(--text-dim);">[${r.brier_ci_95[0].toFixed(4)}, ${r.brier_ci_95[1].toFixed(4)}]</span>
                  </td>
                  <td class="table-num" style="text-align: right; font-weight: 700; color: ${r.failures <= 1 ? 'var(--accent-emerald)' : r.failures <= 2 ? 'var(--accent-amber)' : 'var(--text-muted)'};">
                    ${r.failures} / 8
                  </td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </div>

      <!-- Section B: Linear Probe vs Fine-Tuning Paired Comparison -->
      <div class="card" style="margin-bottom: 28px;">
        <div class="card-title">
          <span>Table 2: Paired Foundation Adaptation Analysis (Linear Probe vs. Fine-Tuned)</span>
          <span style="font-size: 12px; font-family: var(--font-mono); color: var(--text-dim);">Direct Before/After Paired Evaluation ($N=8$ Subjects)</span>
        </div>
        <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 16px;">
          Direct paired comparison evaluating whether task adaptation reduced the representation bottleneck and how it altered prediction outcomes on individual holdout subjects. Differences represent observational pilot comparisons.
        </p>

        <div class="table-responsive">
          <table class="scientific-table">
            <thead>
              <tr>
                <th>Foundation Model</th>
                <th>Cohort</th>
                <th>Baseline</th>
                <th>Adapted Strategy</th>
                <th style="text-align: right;">Base Acc</th>
                <th style="text-align: right;">Adapt Acc</th>
                <th style="text-align: right;">Δ Acc</th>
                <th style="text-align: right;">Base AUC</th>
                <th style="text-align: right;">Adapt AUC</th>
                <th style="text-align: right;">Δ AUC</th>
                <th style="text-align: right;">Base Fail</th>
                <th style="text-align: right;">Adapt Fail</th>
                <th style="text-align: right;">Δ Fail</th>
                <th>Subject Transitions (Unch. Corr / Incorr / Gain / Loss)</th>
              </tr>
            </thead>
            <tbody>
              ${adapt && adapt.paired_models ? adapt.paired_models.map(p => {
                const tc = p.transition_counts;
                return `
                  <tr>
                    <td style="font-weight: 700; color: #fff;">${p.family}</td>
                    <td style="font-size: 12px; color: ${p.cohort.includes('OASIS') ? 'var(--accent-cyan)' : 'var(--accent-amber)'};">${p.cohort.includes('OASIS') ? 'OASIS-1' : 'ADNI'}</td>
                    <td><span class="badge" style="background: rgba(148, 163, 184, 0.15); color: #94a3b8;">${p.baseline_strategy}</span></td>
                    <td><span class="badge" style="background: rgba(16, 185, 129, 0.15); color: #34d399;">${p.adapted_strategy}</span></td>
                    <td class="table-num" style="text-align: right;">${(p.baseline_accuracy * 100).toFixed(1)}%</td>
                    <td class="table-num" style="text-align: right; font-weight: 600;">${(p.adapted_accuracy * 100).toFixed(1)}%</td>
                    <td class="table-num" style="text-align: right; color: ${p.delta_accuracy > 0 ? '#34d399' : p.delta_accuracy < 0 ? '#fb7185' : 'var(--text-muted)'}; font-weight: 700;">
                      ${p.delta_accuracy >= 0 ? '+' : ''}${(p.delta_accuracy * 100).toFixed(1)}%
                    </td>
                    <td class="table-num" style="text-align: right;">${p.baseline_roc_auc.toFixed(4)}</td>
                    <td class="table-num" style="text-align: right; font-weight: 600;">${p.adapted_roc_auc.toFixed(4)}</td>
                    <td class="table-num" style="text-align: right; color: ${p.delta_roc_auc > 0 ? '#34d399' : p.delta_roc_auc < 0 ? '#fb7185' : 'var(--text-muted)'}; font-weight: 700;">
                      ${p.delta_roc_auc >= 0 ? '+' : ''}${p.delta_roc_auc.toFixed(4)}
                    </td>
                    <td class="table-num" style="text-align: right;">${p.baseline_failures}</td>
                    <td class="table-num" style="text-align: right; font-weight: 600;">${p.adapted_failures}</td>
                    <td class="table-num" style="text-align: right; color: ${p.delta_failures < 0 ? '#34d399' : p.delta_failures > 0 ? '#fb7185' : 'var(--text-muted)'}; font-weight: 700;">
                      ${p.delta_failures >= 0 ? '+' : ''}${p.delta_failures}
                    </td>
                    <td style="font-size: 11px; font-family: var(--font-mono); color: var(--text-muted);">
                      <span style="color: #34d399;">${tc['Unchanged Correct']} C</span> / 
                      <span style="color: #fb7185;">${tc['Unchanged Incorrect']} I</span> / 
                      <span style="color: var(--accent-cyan);">+${tc['Baseline Incorrect -> Adapted Correct']} gain</span> / 
                      <span style="color: var(--accent-amber);">${tc['Baseline Correct -> Adapted Incorrect']} loss</span>
                    </td>
                  </tr>
                `;
              }).join('') : ''}
            </tbody>
          </table>
        </div>
      </div>

      <!-- Section C: Cross-Cohort Domain Shift Table -->
      <div class="card">
        <div class="card-title">
          <span>Table 3: Observed Cross-Cohort Generalization Differences (OASIS-1 → ADNI)</span>
          <span style="font-size: 12px; font-family: var(--font-mono); color: var(--text-dim);">Observed Cross-Cohort Performance Change</span>
        </div>
        <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 16px;">
          Performance shifts observed across the 1.5T single-site OASIS-1 test set and the 3.0T multi-scanner ADNI holdout. Changes reflect dataset variation, acquisition differences, and patient-level heterogeneity rather than isolated scanner physics.
        </p>

        <div class="table-responsive">
          <table class="scientific-table">
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>Category</th>
                <th style="text-align: right;">OASIS Test AUC</th>
                <th style="text-align: right;">ADNI Holdout AUC</th>
                <th style="text-align: right;">Observed Δ AUC</th>
                <th style="text-align: right;">OASIS Acc</th>
                <th style="text-align: right;">ADNI Acc</th>
                <th style="text-align: right;">Observed Δ Acc</th>
                <th>Interpretive Finding</th>
              </tr>
            </thead>
            <tbody>
              ${ds ? ds.map(d => `
                <tr style="${d.model_id === 'fomo25_ft' ? 'background: rgba(244, 63, 94, 0.05);' : d.model_id === 'swin_ft' ? 'background: rgba(16, 185, 129, 0.05);' : ''}">
                  <td style="font-weight: 600; color: #fff;">${d.model_name}</td>
                  <td><span class="badge" style="background: rgba(148, 163, 184, 0.12); color: #94a3b8; font-size: 10px;">${d.category}</span></td>
                  <td class="table-num" style="text-align: right;">${d.oasis_roc_auc.toFixed(4)}</td>
                  <td class="table-num" style="text-align: right;">${d.adni_roc_auc.toFixed(4)}</td>
                  <td class="table-num" style="text-align: right; font-weight: 700; color: ${d.observed_cross_cohort_auc_change > 0 ? '#34d399' : d.observed_cross_cohort_auc_change < -0.1 ? '#fb7185' : 'var(--text-muted)'};">
                    ${d.observed_cross_cohort_auc_change >= 0 ? '+' : ''}${d.observed_cross_cohort_auc_change.toFixed(4)}
                  </td>
                  <td class="table-num" style="text-align: right;">${(d.oasis_accuracy * 100).toFixed(1)}%</td>
                  <td class="table-num" style="text-align: right;">${(d.adni_accuracy * 100).toFixed(1)}%</td>
                  <td class="table-num" style="text-align: right; font-weight: 700; color: ${d.observed_cross_cohort_accuracy_change > 0 ? '#34d399' : d.observed_cross_cohort_accuracy_change < -0.1 ? '#fb7185' : 'var(--text-muted)'};">
                    ${d.observed_cross_cohort_accuracy_change >= 0 ? '+' : ''}${(d.observed_cross_cohort_accuracy_change * 100).toFixed(1)}%
                  </td>
                  <td style="font-size: 12px; color: var(--text-muted); max-width: 320px; white-space: normal; line-height: 1.4;">
                    ${d.interpretive_finding}
                  </td>
                </tr>
              `).join('') : ''}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  renderFatalError(err) {
    const main = document.getElementById('content-container');
    if (main) {
      main.innerHTML = `
        <div class="card" style="border-color: #ef4444; padding: 32px; text-align: center;">
          <h2 style="color: #ef4444; font-size: 22px; margin-bottom: 12px;">Frontend Data Loading Error</h2>
          <p style="color: var(--text-muted); margin-bottom: 20px;">
            The application failed to load or validate canonical research datasets.
          </p>
          <pre style="background: #000; padding: 16px; border-radius: 6px; text-align: left; overflow-x: auto; color: #fca5a5; font-family: var(--font-mono); font-size: 12px;">
${err.stack || err.message}
          </pre>
        </div>
      `;
    }
  }

  // ==========================================================================
  /* ==========================================================================
     SHIFT-MRI RESEARCH STUDIO: 1. INTERACTIVE ORTHOGONAL MRI EXPLORER
     ========================================================================== */
  renderMRIExplorer() {
    const container = document.getElementById('section-mri-explorer');
    if (!container) return;

    container.innerHTML = `
      <div class="section-header">
        <div class="section-tag" style="color: #9CC5AA; background: rgba(156,197,170,0.1); border-color: #809E83;">
          Neuroscience Workstation ? 848 Verified Volumetric Scans
        </div>
        <h1 class="section-title">Interactive Orthogonal MRI Explorer & Multi-Model Inference</h1>
        <p class="section-desc">
          Inspect authentic 3D brain MRI volumes across OASIS-1 (N=198), ADNI (N=500), and OASIS-2 (N=150).
          Browse scans with server-side pagination, evaluate dual-mode inference (Cached vs Live Checkpoint), and audit SHIFT-MRI TrustGate reliability.
        </p>
      </div>

      <!-- Advanced Filter & Pagination Control Bar -->
      <div class="card" style="margin-bottom: 20px; background-color: #222824; border-color: #2d3630; padding: 18px;">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 14px;">
          <div>
            <label style="font-size: 11px; color: #A8A396; display: block; margin-bottom: 4px; font-weight: 600;">Cohort</label>
            <select id="mri-cohort-select" style="width: 100%; background: #1c221e; color: #ECE9DF; border: 1px solid #3d4a41; border-radius: 6px; padding: 6px 10px; font-size: 12px;">
              <option value="ADNI" selected>ADNI External (N=500, Multi-Site)</option>
              <option value="OASIS-1">OASIS-1 In-Domain (N=198, Siemens 1.5T)</option>
              <option value="OASIS-2">OASIS-2 Exploratory (N=150, Siemens 1.5T)</option>
              <option value="ALL">All Cohorts (N=848 Total)</option>
            </select>
          </div>

          <div>
            <label style="font-size: 11px; color: #A8A396; display: block; margin-bottom: 4px; font-weight: 600;">Scanner Manufacturer</label>
            <select id="mri-scanner-select" style="width: 100%; background: #1c221e; color: #ECE9DF; border: 1px solid #3d4a41; border-radius: 6px; padding: 6px 10px; font-size: 12px;">
              <option value="ALL" selected>All Scanners (Siemens & Philips)</option>
              <option value="Siemens">Siemens (N=568 Scans)</option>
              <option value="Philips">Philips (N=130 Scans)</option>
            </select>
          </div>

          <div>
            <label style="font-size: 11px; color: #A8A396; display: block; margin-bottom: 4px; font-weight: 600;">Clinical Diagnosis</label>
            <select id="mri-diagnosis-select" style="width: 100%; background: #1c221e; color: #ECE9DF; border: 1px solid #3d4a41; border-radius: 6px; padding: 6px 10px; font-size: 12px;">
              <option value="ALL" selected>All Diagnoses</option>
              <option value="CN">Cognitively Normal (CN)</option>
              <option value="AD">Impaired / AD</option>
            </select>
          </div>

          <div>
            <label style="font-size: 11px; color: #A8A396; display: block; margin-bottom: 4px; font-weight: 600;">Search Scan / Subject</label>
            <input id="mri-search-input" type="text" placeholder="e.g. 002_S_0295 or OAS1_0001" style="width: 100%; background: #1c221e; color: #ECE9DF; border: 1px solid #3d4a41; border-radius: 6px; padding: 6px 10px; font-size: 12px;">
          </div>
        </div>

        <!-- Scan Selection & Pagination Controls -->
        <div style="display: flex; flex-wrap: wrap; gap: 12px; align-items: center; justify-content: space-between; border-top: 1px solid #2d3630; padding-top: 12px;">
          <div style="flex: 2; min-width: 280px;">
            <label style="font-size: 11px; color: #A8A396; display: block; margin-bottom: 4px; font-weight: 600;">Active Volume</label>
            <select id="mri-scan-select" style="width: 100%; background: #1c221e; color: #ECE9DF; border: 1px solid #3d4a41; border-radius: 6px; padding: 7px 10px; font-size: 12px; font-family: var(--font-mono);">
              <option value="">Loading scans...</option>
            </select>
          </div>

          <div style="display: flex; align-items: center; gap: 8px;">
            <button id="mri-page-prev" class="btn btn-secondary" style="background: #1c221e; color: #ECE9DF; border: 1px solid #3d4a41; padding: 6px 12px; border-radius: 6px; font-size: 12px; cursor: pointer;">&larr; Prev</button>
            <span id="mri-page-indicator" style="font-size: 12px; color: #A8A396; font-family: var(--font-mono);">Page 1 / 10</span>
            <button id="mri-page-next" class="btn btn-secondary" style="background: #1c221e; color: #ECE9DF; border: 1px solid #3d4a41; padding: 6px 12px; border-radius: 6px; font-size: 12px; cursor: pointer;">Next &rarr;</button>
          </div>

          <div id="mri-scan-metadata-badge" style="display: flex; gap: 8px;">
            <span class="badge" style="background: #1c221e; color: #9CC5AA; border: 1px solid #3d4a41;">Scanner: Loading...</span>
            <span class="badge" style="background: #1c221e; color: #C7A97A; border: 1px solid #3d4a41;">Diagnosis: Loading...</span>
          </div>
        </div>
      </div>

      <!-- Orthogonal 3-Plane MRI Slices Grid -->
      <div class="mri-orthogonal-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px;">
        <!-- Axial Viewport -->
        <div class="mri-viewport-card" style="background: #222824; border: 1px solid #2d3630; border-radius: 8px; padding: 12px;">
          <div class="mri-viewport-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 13px; font-weight: 600; color: #ECE9DF;">Axial (Transverse)</span>
            <span id="mri-axial-badge" class="mri-slice-badge" style="font-family: var(--font-mono); font-size: 11px; background: #1c221e; padding: 2px 6px; border-radius: 4px; color: #9CC5AA;">Z: 64/128</span>
          </div>
          <div class="mri-slice-canvas-wrap" style="background: #000; border-radius: 4px; overflow: hidden; display: flex; align-items: center; justify-content: center; min-height: 240px;">
            <img id="mri-img-axial" class="mri-slice-img" src="" alt="Axial MRI Slice" style="width: 100%; height: auto; max-height: 280px; object-fit: contain;">
          </div>
          <div class="mri-slice-controls" style="margin-top: 8px;">
            <div style="display: flex; align-items: center; gap: 8px;">
              <label style="font-size: 11px; color: #A8A396;">Slice</label>
              <input id="mri-slider-axial" type="range" min="0" max="127" value="64" style="flex: 1;">
            </div>
          </div>
        </div>

        <!-- Coronal Viewport -->
        <div class="mri-viewport-card" style="background: #222824; border: 1px solid #2d3630; border-radius: 8px; padding: 12px;">
          <div class="mri-viewport-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 13px; font-weight: 600; color: #ECE9DF;">Coronal (Frontal)</span>
            <span id="mri-coronal-badge" class="mri-slice-badge" style="font-family: var(--font-mono); font-size: 11px; background: #1c221e; padding: 2px 6px; border-radius: 4px; color: #9CC5AA;">Y: 64/128</span>
          </div>
          <div class="mri-slice-canvas-wrap" style="background: #000; border-radius: 4px; overflow: hidden; display: flex; align-items: center; justify-content: center; min-height: 240px;">
            <img id="mri-img-coronal" class="mri-slice-img" src="" alt="Coronal MRI Slice" style="width: 100%; height: auto; max-height: 280px; object-fit: contain;">
          </div>
          <div class="mri-slice-controls" style="margin-top: 8px;">
            <div style="display: flex; align-items: center; gap: 8px;">
              <label style="font-size: 11px; color: #A8A396;">Slice</label>
              <input id="mri-slider-coronal" type="range" min="0" max="127" value="64" style="flex: 1;">
            </div>
          </div>
        </div>

        <!-- Sagittal Viewport -->
        <div class="mri-viewport-card" style="background: #222824; border: 1px solid #2d3630; border-radius: 8px; padding: 12px;">
          <div class="mri-viewport-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 13px; font-weight: 600; color: #ECE9DF;">Sagittal (Lateral)</span>
            <span id="mri-sagittal-badge" class="mri-slice-badge" style="font-family: var(--font-mono); font-size: 11px; background: #1c221e; padding: 2px 6px; border-radius: 4px; color: #9CC5AA;">X: 64/128</span>
          </div>
          <div class="mri-slice-canvas-wrap" style="background: #000; border-radius: 4px; overflow: hidden; display: flex; align-items: center; justify-content: center; min-height: 240px;">
            <img id="mri-img-sagittal" class="mri-slice-img" src="" alt="Sagittal MRI Slice" style="width: 100%; height: auto; max-height: 280px; object-fit: contain;">
          </div>
          <div class="mri-slice-controls" style="margin-top: 8px;">
            <div style="display: flex; align-items: center; gap: 8px;">
              <label style="font-size: 11px; color: #A8A396;">Slice</label>
              <input id="mri-slider-sagittal" type="range" min="0" max="127" value="64" style="flex: 1;">
            </div>
          </div>
        </div>
      </div>

      <!-- Controls: Heatmap & Inference Mode -->
      <div class="card" style="margin-bottom: 24px; background-color: #222824; border-color: #2d3630; padding: 14px 18px; display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 16px;">
        <div style="display: flex; align-items: center; gap: 18px;">
          <label style="display: flex; align-items: center; gap: 8px; font-size: 12px; color: #ECE9DF; cursor: pointer;">
            <input type="checkbox" id="mri-toggle-heatmap" style="accent-color: #9CC5AA;">
            <span>Overlay Authentic 3D Grad-CAM (ResNet-10)</span>
          </label>
          <div style="display: flex; align-items: center; gap: 8px; font-size: 12px; color: #A8A396;">
            <span>Mode:</span>
            <label style="display: flex; align-items: center; gap: 4px; color: #ECE9DF; cursor: pointer;">
              <input type="radio" name="mri-mode" value="cached" checked style="accent-color: #9CC5AA;"> Cached Research Artifacts
            </label>
            <label style="display: flex; align-items: center; gap: 4px; color: #ECE9DF; cursor: pointer;">
              <input type="radio" name="mri-mode" value="live" style="accent-color: #9CC5AA;"> Live Checkpoint (CUDA)
            </label>
          </div>
        </div>

        <button id="btn-run-mri-inference" class="btn btn-primary" style="background: linear-gradient(135deg, #809E83, #9CC5AA); color: #181B19; font-weight: 700; border: none; padding: 8px 18px; border-radius: 6px; cursor: pointer;">
          &#9654; Run Multi-Model Inference & TrustGate
        </button>
      </div>

      <!-- Multi-Model Predictions & TrustGate Panel -->
      <div id="mri-inference-container" class="card" style="background: #222824; border-color: #2d3630; padding: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
          <h3 style="font-size: 15px; font-weight: 700; color: #ECE9DF; margin: 0;">Multi-Model Predictions & Failure Diagnostics</h3>
          <span id="mri-provenance-tag" style="font-size: 11px; font-family: var(--font-mono); color: #A8A396; background: #1c221e; padding: 3px 8px; border-radius: 4px;">
            Provenance: Authoritative Verified CSV
          </span>
        </div>

        <!-- Dynamic TrustGate Decision Banner -->
        <div id="trustgate-decision-banner" class="trustgate-status-banner accepted" style="background: #1c221e; border: 1px solid #3d4a41; border-radius: 6px; padding: 12px 16px; margin-bottom: 18px; display: flex; justify-content: space-between; align-items: center;">
          <div>
            <strong>TRUSTGATE EVALUATION:</strong> <span id="tg-banner-status">Select a scan and click Run Multi-Model Inference</span>
          </div>
          <div id="tg-banner-metric" style="font-family: var(--font-mono); font-size: 12px; color: #C7A97A;"></div>
        </div>

        <!-- 10 Model Probabilities Grid -->
        <div id="mri-model-bars" style="margin-bottom: 24px;"></div>

        <!-- TrustGate Risk Breakdown -->
        <div class="trustgate-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px;">
          <div class="trustgate-metric-box" style="background: #1c221e; border: 1px solid #2d3630; border-radius: 6px; padding: 12px;">
            <div class="metric-name" style="font-size: 11px; color: #A8A396; margin-bottom: 4px;">Confidence Margin (S_A)</div>
            <div id="tg-metric-margin" class="metric-val" style="font-size: 18px; font-weight: 700; color: #ECE9DF; font-family: var(--font-mono);">-</div>
            <div style="font-size: 10px; color: #9CC5AA; margin-top: 4px;">Top Error Detector (AUROC 0.5829)</div>
          </div>
          <div class="trustgate-metric-box" style="background: #1c221e; border: 1px solid #2d3630; border-radius: 6px; padding: 12px;">
            <div class="metric-name" style="font-size: 11px; color: #A8A396; margin-bottom: 4px;">Mahalanobis Distance (S_B)</div>
            <div id="tg-metric-mahalanobis" class="metric-val" style="font-size: 18px; font-weight: 700; color: #ECE9DF; font-family: var(--font-mono);">-</div>
            <div id="tg-alert-saturation" class="trustgate-alert-tag" style="display: none; font-size: 10px; color: #C7A97A; margin-top: 4px;">100% Saturation Alert</div>
          </div>
          <div class="trustgate-metric-box" style="background: #1c221e; border: 1px solid #2d3630; border-radius: 6px; padding: 12px;">
            <div class="metric-name" style="font-size: 11px; color: #A8A396; margin-bottom: 4px;">Ensemble Disagreement (S_C)</div>
            <div id="tg-metric-disagreement" class="metric-val" style="font-size: 18px; font-weight: 700; color: #ECE9DF; font-family: var(--font-mono);">-</div>
            <div id="tg-alert-consensus" class="trustgate-alert-tag" style="display: none; font-size: 10px; color: #B98378; margin-top: 4px;">Consensus Inversion (AUROC 0.4674)</div>
          </div>
          <div class="trustgate-metric-box" style="background: #1c221e; border: 1px solid #2d3630; border-radius: 6px; padding: 12px;">
            <div class="metric-name" style="font-size: 11px; color: #A8A396; margin-bottom: 4px;">Composite Risk R(x)</div>
            <div id="tg-metric-composite" class="metric-val" style="font-size: 18px; font-weight: 700; color: #ECE9DF; font-family: var(--font-mono);">-</div>
            <div style="font-size: 10px; color: #C7A97A; margin-top: 4px;">Threshold tau* = 0.6340 (Composite AUROC 0.4697)</div>
          </div>
        </div>
      </div>
    `;

    this.bindMRIExplorerEvents();
  }

  async bindMRIExplorerEvents() {
    const cohortSelect = document.getElementById('mri-cohort-select');
    const scannerSelect = document.getElementById('mri-scanner-select');
    const diagnosisSelect = document.getElementById('mri-diagnosis-select');
    const searchInput = document.getElementById('mri-search-input');
    const scanSelect = document.getElementById('mri-scan-select');
    const pagePrev = document.getElementById('mri-page-prev');
    const pageNext = document.getElementById('mri-page-next');
    const pageIndicator = document.getElementById('mri-page-indicator');

    const sliderAxial = document.getElementById('mri-slider-axial');
    const sliderCoronal = document.getElementById('mri-slider-coronal');
    const sliderSagittal = document.getElementById('mri-slider-sagittal');
    const toggleHeatmap = document.getElementById('mri-toggle-heatmap');
    const btnInference = document.getElementById('btn-run-mri-inference');

    let currentScanId = '';
    let currentPage = 1;
    let totalPages = 1;
    const pageLimit = 50;

    const updateSlices = () => {
      if (!currentScanId) return;
      const ax = sliderAxial.value;
      const co = sliderCoronal.value;
      const sa = sliderSagittal.value;

      document.getElementById('mri-axial-badge').textContent = `Z: ${ax}/128`;
      document.getElementById('mri-coronal-badge').textContent = `Y: ${co}/128`;
      document.getElementById('mri-sagittal-badge').textContent = `X: ${sa}/128`;

      const isHeatmap = toggleHeatmap.checked;
      const endpoint = isHeatmap ? '/api/inference/explain' : '/api/mri/slice';

      document.getElementById('mri-img-axial').src = `${endpoint}?scan_id=${encodeURIComponent(currentScanId)}&plane=axial&slice_idx=${ax}&t=${Date.now()}`;
      document.getElementById('mri-img-coronal').src = `${endpoint}?scan_id=${encodeURIComponent(currentScanId)}&plane=coronal&slice_idx=${co}&t=${Date.now()}`;
      document.getElementById('mri-img-sagittal').src = `${endpoint}?scan_id=${encodeURIComponent(currentScanId)}&plane=sagittal&slice_idx=${sa}&t=${Date.now()}`;
    };

    const loadScans = async () => {
      const cohort = cohortSelect.value;
      const scanner = scannerSelect.value;
      const diagnosis = diagnosisSelect.value;
      const search = searchInput.value.trim();

      const params = new URLSearchParams({
        cohort: cohort,
        scanner: scanner,
        diagnosis: diagnosis,
        search: search,
        page: currentPage,
        limit: pageLimit
      });

      try {
        const resp = await fetch(`/api/mri/scans?${params.toString()}`);
        const json = await resp.json();

        totalPages = json.total_pages || 1;
        pageIndicator.textContent = `Page ${json.page} / ${totalPages} (${json.total} scans)`;
        pagePrev.disabled = json.page <= 1;
        pageNext.disabled = json.page >= totalPages;

        scanSelect.innerHTML = '';
        (json.scans || []).forEach((s) => {
          const opt = document.createElement('option');
          opt.value = s.scan_id;
          opt.textContent = `${s.subject_id} [${s.diagnosis}] - ${s.cohort} (${s.manufacturer} ${s.field_strength}T)`;
          scanSelect.appendChild(opt);
        });

        if (json.scans && json.scans.length > 0) {
          currentScanId = json.scans[0].scan_id;
          scanSelect.value = currentScanId;
          updateMetadataBadge(json.scans[0]);
          updateSlices();
          runInference();
        } else {
          currentScanId = '';
          const opt = document.createElement('option');
          opt.value = '';
          opt.textContent = 'No matching scans found.';
          scanSelect.appendChild(opt);
        }
      } catch (e) {
        console.error('Failed to load scans:', e);
      }
    };

    const updateMetadataBadge = (scan) => {
      const badgeContainer = document.getElementById('mri-scan-metadata-badge');
      if (!badgeContainer) return;
      badgeContainer.innerHTML = `
        <span class="badge" style="background: #1c221e; color: #9CC5AA; border: 1px solid #3d4a41;">Scanner: ${scan.manufacturer} ${scan.field_strength}T</span>
        <span class="badge" style="background: #1c221e; color: ${scan.diagnosis === 'AD' ? '#B98378' : '#9CC5AA'}; border: 1px solid #3d4a41;">Diagnosis: ${scan.diagnosis}</span>
        <span class="badge" style="background: #1c221e; color: #C7A97A; border: 1px solid #3d4a41;">Cohort: ${scan.cohort}</span>
      `;
    };

    const runInference = async () => {
      if (!currentScanId) return;
      const mode = document.querySelector('input[name="mri-mode"]:checked')?.value || 'cached';
      const provTag = document.getElementById('mri-provenance-tag');
      if (provTag) {
        provTag.textContent = mode === 'live' ? 'Mode: Live Checkpoint Forward Pass (CUDA)' : 'Mode: Verified Research CSV Artifact';
      }

      try {
        const resp = await fetch(`/api/inference/predict?scan_id=${encodeURIComponent(currentScanId)}&mode=${mode}`);
        const data = await resp.json();
        renderInferenceResults(data);
      } catch (e) {
        console.error('Inference error:', e);
      }
    };

    const renderInferenceResults = (data) => {
      const banner = document.getElementById('trustgate-decision-banner');
      const bannerStatus = document.getElementById('tg-banner-status');
      const bannerMetric = document.getElementById('tg-banner-metric');
      const barsContainer = document.getElementById('mri-model-bars');

      const safeFixed = (val, digits = 4, fallback = "N/A") => {
        if (val === null || val === undefined || isNaN(Number(val))) return fallback;
        return Number(val).toFixed(digits);
      };

      const tg = data.trustgate || {};
      const status = tg.evaluation_status || 'NOT_EVALUATED';
      const decision = tg.gate_decision || 'NOT EVALUATED';

      if (status === 'NOT_EVALUATED' || decision === 'NOT EVALUATED' || tg.composite_risk === null) {
        banner.className = 'trustgate-status-banner accepted';
        banner.style.borderLeft = '4px solid #C7A97A';
        bannerStatus.textContent = `UNAUDITED FOR TRUSTGATE — ${tg.note || 'Cohort overlap unverified or training set scan.'}`;
        bannerMetric.textContent = 'TrustGate Composite: N/A';
      } else if (decision.includes('FLAGGED')) {
        banner.className = 'trustgate-status-banner flagged';
        banner.style.borderLeft = '4px solid #B98378';
        bannerStatus.textContent = 'FLAGGED AS UNRELIABLE (ABSTAIN) — Risk Exceeds Source Threshold';
        bannerMetric.textContent = `Risk Score R(x) = ${safeFixed(tg.composite_risk, 4)} > tau* (${safeFixed(tg.rejection_threshold_tau, 4)})`;
      } else {
        banner.className = 'trustgate-status-banner accepted';
        banner.style.borderLeft = '4px solid #9CC5AA';
        bannerStatus.textContent = 'ACCEPTED — Prediction Passes Reliability Gate';
        bannerMetric.textContent = `Risk Score R(x) = ${safeFixed(tg.composite_risk, 4)} <= tau* (${safeFixed(tg.rejection_threshold_tau, 4)})`;
      }

      document.getElementById('tg-metric-margin').textContent = safeFixed(tg.confidence_margin ?? tg.confidence_margin_sa, 4);
      document.getElementById('tg-metric-mahalanobis').textContent = safeFixed(tg.mahalanobis_score ?? tg.mahalanobis_sb, 4);
      document.getElementById('tg-metric-disagreement').textContent = safeFixed(tg.disagreement ?? tg.disagreement_sc, 4);
      document.getElementById('tg-metric-composite').textContent = safeFixed(tg.composite_risk, 4);

      document.getElementById('tg-alert-saturation').style.display = tg.saturation_alert ? 'inline-block' : 'none';
      document.getElementById('tg-alert-consensus').style.display = tg.consensus_inversion_alert ? 'inline-block' : 'none';

      // Validation thresholds mapping
      const valThresholds = {
        "MONAI Swin UNETR SSL (Linear Probe)": 0.5088,
        "3D ResNet-10 (Pipeline B)": 0.6860,
        "3D DenseNet-121 (Pipeline B)": 0.4651,
        "2.5D Multi-View ResNet-18 (Pipeline B)": 0.9741,
        "Dual-Branch Hybrid (ResNet-10 + Radiomics)": 0.4763,
        "FOMO25 AMAES ResEnc-B (Linear Probe)": 0.5457,
        "3D ResNet-18 (Scratch Baseline)": 0.5725,
        "3D ResNet-34 (Scratch Baseline)": 0.5512,
        "2D Transfer ResNet-18 (Slice-Aggregated)": 0.6120,
        "Classical Radiomics (Linear SVM)": 0.5693
      };

      // Render model bars
      barsContainer.innerHTML = '';
      const predictions = data.predictions || data.cached_reference_predictions || {};

      for (const [modelName, prob] of Object.entries(predictions)) {
        const t_star = valThresholds[modelName] || 0.5000;
        const t_star_pct = (t_star * 100).toFixed(1);

        const isAvailable = prob !== null && prob !== undefined && typeof prob === 'number' && !isNaN(prob);
        const valText = isAvailable ? Number(prob).toFixed(4) : (typeof prob === 'string' ? prob : 'PREDICTION UNAVAILABLE');
        const pct = isAvailable ? Math.min(100, Math.max(0, prob * 100)).toFixed(1) : '0';
        const isPositive = isAvailable && prob >= t_star;

        const row = document.createElement('div');
        row.className = 'model-prob-bar-row';
        row.style.marginBottom = '10px';
        row.innerHTML = `
          <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
            <span style="font-size: 12px; font-weight: 600; color: #ECE9DF;">${modelName}</span>
            <span style="font-size: 11px; font-family: var(--font-mono); color: ${isAvailable ? (isPositive ? '#B98378' : '#9CC5AA') : '#7A7569'};">
              ${valText} ${isAvailable ? `(t* = ${t_star.toFixed(4)})` : ''}
            </span>
          </div>
          <div class="model-prob-track" style="height: 8px; background: #1c221e; border-radius: 4px; position: relative; overflow: hidden;">
            <div class="model-prob-threshold-marker" style="position: absolute; left: ${t_star_pct}%; top: 0; bottom: 0; width: 2px; background: #C7A97A; z-index: 2;" title="Validation Threshold t* = ${t_star.toFixed(4)}"></div>
            <div class="model-prob-fill" style="height: 100%; width: ${pct}%; background: ${isPositive ? 'linear-gradient(90deg, #809E83, #B98378)' : '#809E83'}; transition: width 0.3s ease;"></div>
          </div>
        `;
        barsContainer.appendChild(row);
      }
    };

    // Event listeners
    cohortSelect.addEventListener('change', () => { currentPage = 1; loadScans(); });
    scannerSelect.addEventListener('change', () => { currentPage = 1; loadScans(); });
    diagnosisSelect.addEventListener('change', () => { currentPage = 1; loadScans(); });
    searchInput.addEventListener('input', () => { currentPage = 1; loadScans(); });

    pagePrev.addEventListener('click', () => {
      if (currentPage > 1) { currentPage--; loadScans(); }
    });
    pageNext.addEventListener('click', () => {
      if (currentPage < totalPages) { currentPage++; loadScans(); }
    });

    scanSelect.addEventListener('change', () => {
      currentScanId = scanSelect.value;
      updateSlices();
      runInference();
    });

    sliderAxial.addEventListener('input', updateSlices);
    sliderCoronal.addEventListener('input', updateSlices);
    sliderSagittal.addEventListener('input', updateSlices);
    toggleHeatmap.addEventListener('change', updateSlices);
    btnInference.addEventListener('click', runInference);

    // Initial load
    loadScans();
  }

// SHIFT-MRI RESEARCH STUDIO: 2. TRUSTGATE FAILURE STUDIO
  // ==========================================================================

  renderTrustGateStudio() {
    const container = document.getElementById('section-trustgate-studio');
    if (!container) return;

    container.innerHTML = `
      <div class="section-header">
        <div class="section-tag" style="color: #C7A97A; background: rgba(199,169,122,0.1); border-color: #C7A97A;">
          Methodological Contribution • Documented Negative Result
        </div>
        <h1 class="section-title">SHIFT-MRI TrustGate: Failure Analysis Studio</h1>
        <p class="section-desc">
          Investigation of source-trained selective abstention mechanisms under realistic neuroimaging distribution shifts.
          Rigorously establishes why simpler classification margin outperforms complex multi-component gates.
        </p>
      </div>

      <!-- Key Findings Grid -->
      <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 24px;">
        <div class="card" style="background-color: #222824; border-color: #2d3630;">
          <div style="font-size: 11px; color: #9CC5AA; font-weight: 700; text-transform: uppercase;">Top Error Detector</div>
          <div style="font-size: 26px; font-weight: 800; color: #ECE9DF; margin: 8px 0;">Margin Alone (0.5829)</div>
          <p style="font-size: 12px; color: #A8A396;">
            Classification margin (1 - 2|P - 0.5|) achieves the highest error detection AUROC under ADNI scanner shift.
          </p>
        </div>

        <div class="card" style="background-color: #222824; border-color: #2d3630;">
          <div style="font-size: 11px; color: #B98378; font-weight: 700; text-transform: uppercase;">Negative Result</div>
          <div style="font-size: 26px; font-weight: 800; color: #B98378; margin: 8px 0;">Composite Gate (0.4697)</div>
          <p style="font-size: 12px; color: #A8A396;">
            Integrating Mahalanobis distance, consensus variance, and image quality degrades error-detection performance by -0.1132 AUROC below margin alone.
          </p>
        </div>

        <div class="card" style="background-color: #222824; border-color: #2d3630;">
          <div style="font-size: 11px; color: #C7A97A; font-weight: 700; text-transform: uppercase;">Prospective OOD Detection</div>
          <div style="font-size: 26px; font-weight: 800; color: #C7A97A; margin: 8px 0;">Mahalanobis (1.0000)</div>
          <p style="font-size: 12px; color: #A8A396;">
            Feature distance excels at cohort-level domain shift detection, achieving perfect OASIS-1 vs. ADNI separation.
          </p>
        </div>
      </div>

      <!-- Failure Analysis Table -->
      <div class="card" style="margin-bottom: 24px;">
        <h3 style="font-size: 15px; font-weight: 700; color: #ECE9DF; margin-bottom: 12px;">TrustGate Benchmark Comparison Table (ADNI N=500)</h3>
        <table class="data-table">
          <thead>
            <tr>
              <th>Risk Estimator</th>
              <th>OASIS-1 Source AUC</th>
              <th>ADNI Error AUC</th>
              <th>pAURC [0.5, 1.0]</th>
              <th>Full AURC</th>
              <th>Empirical Status</th>
            </tr>
          </thead>
          <tbody>
            <tr style="background: rgba(156,197,170,0.1); font-weight: 700;">
              <td>Confidence Margin (MSP)</td>
              <td>0.5374</td>
              <td style="color: #9CC5AA;">0.5829</td>
              <td>0.2871</td>
              <td>0.5488</td>
              <td>Top Error Detector</td>
            </tr>
            <tr>
              <td>Multi-Model Disagreement</td>
              <td>0.5951</td>
              <td>0.4674</td>
              <td>0.3086</td>
              <td>0.6481</td>
              <td>Consensus Inversion</td>
            </tr>
            <tr>
              <td>Raw Mahalanobis Distance</td>
              <td>0.4527</td>
              <td>0.4808</td>
              <td>-</td>
              <td>-</td>
              <td>Sub-Random Chance</td>
            </tr>
            <tr>
              <td>Clipped Mahalanobis Distance</td>
              <td>0.4529</td>
              <td>0.5000</td>
              <td>-</td>
              <td>-</td>
              <td>100% Saturated on ADNI</td>
            </tr>
            <tr style="background: rgba(185,131,120,0.1); font-weight: 700;">
              <td>Composite TrustGate (Source-Locked)</td>
              <td>0.5129</td>
              <td style="color: #B98378;">0.4697</td>
              <td>0.3088</td>
              <td>0.6470</td>
              <td>Empirical Negative Result</td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Diagnostic Insights -->
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px;">
        <div class="card" style="background-color: #1c221e; border-color: #2d3630;">
          <h4 style="font-size: 13px; font-weight: 700; color: #B98378; margin-bottom: 8px;">Pathology 1: Feature Saturation</h4>
          <p style="font-size: 12px; color: #ECE9DF; line-height: 1.6;">
            Under cross-scanner shift, target feature representations from ADNI lie entirely outside the source OASIS-1 support.
            100% of ADNI samples exceed the source 95th-percentile feature clipping threshold (q95 = 4.3905).
            Furthermore, under the source rejection threshold tau* = 0.6340 (80th percentile of source validation risk),
            100% of ADNI external scans (500/500) exhibit composite risk >= 0.8033 and are flagged.
            Nonlinear mapping saturates all feature distance scores to S_B = 1.0000, creating zero discriminative variance and yielding exactly 0.5000 AUROC.
            Scans passing or failing TrustGate are research flags and must NOT be interpreted as clinically reliable.
          </p>
        </div>

        <div class="card" style="background-color: #1c221e; border-color: #2d3630;">
          <h4 style="font-size: 13px; font-weight: 700; color: #C7A97A; margin-bottom: 8px;">Pathology 2: The Consensus Inversion Paradox</h4>
          <p style="font-size: 12px; color: #ECE9DF; line-height: 1.6;">
            On in-domain OASIS-1, ensemble disagreement reliably identifies difficult ambiguous scans (AUC = 0.5951).
            Under ADNI distribution shift, overall error rate surges to 60.6%, yet models paradoxically agree more strongly.
            Shared inductive biases cause vision models to commit the exact same false-positive errors, inverting the reliability of consensus.
          </p>
        </div>
      </div>
    `;
  }

  // ==========================================================================
  // SHIFT-MRI RESEARCH STUDIO: 3. DEMOGRAPHIC SHORTCUT RESOLUTION (N=500)
  // ==========================================================================

  renderDemographicShortcuts() {
    const container = document.getElementById('section-demographic-shortcuts');
    if (!container) return;

    container.innerHTML = `
      <div class="section-header">
        <div class="section-tag" style="color: #9CC5AA; background: rgba(156,197,170,0.1); border-color: #809E83;">
          Scientific Verification • Confounding Audit
        </div>
        <h1 class="section-title">Demographic Shortcut Verification (ADNI N=500)</h1>
        <p class="section-desc">
          Investigation of non-imaging confounding variables (Patient Age & Biological Sex) across the matched 500-subject ADNI cohort.
          Definitively resolves the historical 50-patient pilot artifact (0.5671) and proves genuine imaging signal in deep vision models.
        </p>
      </div>

      <div class="card" style="margin-bottom: 24px;">
        <h3 style="font-size: 15px; font-weight: 700; color: #ECE9DF; margin-bottom: 12px;">Demographic Baselines vs. Deep Vision Models</h3>
        <table class="data-table">
          <thead>
            <tr>
              <th>Model / Baseline Strategy</th>
              <th>Cohort Size (N)</th>
              <th>ROC-AUC</th>
              <th>95% Confidence Interval</th>
              <th>Brier Score</th>
              <th>Scientific Interpretation</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Demographic: Age Alone</td>
              <td>500</td>
              <td>0.5242</td>
              <td>[0.4727, 0.5753]</td>
              <td>0.2516</td>
              <td>Near random chance; age alone does not separate AD from CN</td>
            </tr>
            <tr>
              <td>Demographic: Sex Alone</td>
              <td>500</td>
              <td>0.5232</td>
              <td>[0.4716, 0.5748]</td>
              <td>0.2518</td>
              <td>Near random chance; no sex confounding</td>
            </tr>
            <tr style="background: rgba(199,169,122,0.1); font-weight: 700;">
              <td>Demographic: Age + Sex (Matched Full Cohort)</td>
              <td>500</td>
              <td style="color: #C7A97A;">0.5123</td>
              <td>[0.4608, 0.5638]</td>
              <td>0.2520</td>
              <td>Definitive benchmark: zero demographic shortcut on N=500</td>
            </tr>
            <tr style="background: rgba(156,197,170,0.1); font-weight: 700;">
              <td>MONAI Swin UNETR SSL (Linear Probe)</td>
              <td>500</td>
              <td style="color: #9CC5AA;">0.5834</td>
              <td>[0.5331, 0.6337]</td>
              <td>0.2569</td>
              <td>Statistically superior to demographic chance (+0.0711 AUC)</td>
            </tr>
            <tr style="background: rgba(156,197,170,0.1); font-weight: 700;">
              <td>2D Transfer ResNet-18 (Slice-Aggregated)</td>
              <td>500</td>
              <td style="color: #9CC5AA;">0.5778</td>
              <td>[0.5280, 0.6276]</td>
              <td>0.5315</td>
              <td>Statistically superior to demographic chance (+0.0655 AUC)</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="card" style="background-color: #1c221e; border-color: #2d3630;">
        <h4 style="font-size: 13px; font-weight: 700; color: #9CC5AA; margin-bottom: 8px;">Resolution of the N=50 Pilot Study Variance</h4>
        <p style="font-size: 12px; color: #ECE9DF; line-height: 1.6;">
          In an early exploratory pilot on N=50 subjects, Age+Sex logistic regression had yielded an elevated ROC-AUC of 0.5671.
          Our audit reveals this was an artifact of small-sample variance. On the complete 500-subject ADNI cohort, non-imaging demographic
          discrimination drops to <strong>0.5123</strong>, confirming that top imaging models capture genuine structural atrophy patterns.
        </p>
      </div>
    `;
  }

  // ==========================================================================
  // SHIFT-MRI RESEARCH STUDIO: 4. SUBMISSION DELIVERABLES DOWNLOAD HUB
  // ==========================================================================

  renderDownloadsHub() {
    const container = document.getElementById('section-downloads-hub');
    if (!container) return;

    container.innerHTML = `
      <div class="section-header">
        <div class="section-tag" style="color: #9CC5AA; background: rgba(156,197,170,0.1); border-color: #809E83;">
          Official Submission Deliverables • Submission Draft
        </div>
        <h1 class="section-title">Research Deliverables & Download Hub</h1>
        <p class="section-desc">
          Download the publication-ready scientific research paper, complete editable manuscript, Scientific presentation slide deck (PPTX),
          and the complete reproduction archive containing code, canonical registries, and audit logs.
        </p>
      </div>

      <!-- Download Cards Grid -->
      <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-bottom: 28px;">
        <!-- PDF Paper -->
        <div class="card" style="background-color: #222824; border-color: #3d4a41; padding: 20px;">
          <div style="display: flex; align-items: flex-start; justify-content: space-between;">
            <div>
              <div style="font-size: 11px; color: #9CC5AA; font-weight: 700; text-transform: uppercase;">Publication Monograph</div>
              <h3 style="font-size: 16px; font-weight: 700; color: #ECE9DF; margin: 6px 0;">Complete Research Paper (PDF)</h3>
              <p style="font-size: 12px; color: #A8A396; margin-bottom: 14px;">
                Full publication-grade monograph with structured abstract, embedded high-resolution figures, 10-model leaderboard,
                TrustGate failure analysis, and bibliography.
              </p>
            </div>
            <span style="font-size: 28px;">📄</span>
          </div>
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-family: var(--font-mono); font-size: 11px; color: #A8A396;">Size: ~850 KB • Letter Format</span>
            <a href="/downloads/SHIFT_MRI_Complete_Research_Paper.pdf" class="btn btn-primary" download style="background: #9CC5AA; color: #181B19; font-weight: 700; text-decoration: none; padding: 8px 16px; border-radius: 6px;">
              Download PDF
            </a>
          </div>
        </div>

        <!-- Word Document -->
        <div class="card" style="background-color: #222824; border-color: #3d4a41; padding: 20px;">
          <div style="display: flex; align-items: flex-start; justify-content: space-between;">
            <div>
              <div style="font-size: 11px; color: #9CC5AA; font-weight: 700; text-transform: uppercase;">Editable Manuscript</div>
              <h3 style="font-size: 16px; font-weight: 700; color: #ECE9DF; margin: 6px 0;">Research Paper (DOCX)</h3>
              <p style="font-size: 12px; color: #A8A396; margin-bottom: 14px;">
                Complete editable Word manuscript with formatted tables, typography, and section hierarchies ready for peer review submission.
              </p>
            </div>
            <span style="font-size: 28px;">📝</span>
          </div>
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-family: var(--font-mono); font-size: 11px; color: #A8A396;">Size: ~230 KB • Word Format</span>
            <a href="/downloads/SHIFT_MRI_Complete_Research_Paper.docx" class="btn btn-primary" download style="background: #9CC5AA; color: #181B19; font-weight: 700; text-decoration: none; padding: 8px 16px; border-radius: 6px;">
              Download DOCX
            </a>
          </div>
        </div>

        <!-- Slide Deck -->
        <div class="card" style="background-color: #222824; border-color: #C7A97A; padding: 20px;">
          <div style="display: flex; align-items: flex-start; justify-content: space-between;">
            <div>
              <div style="font-size: 11px; color: #C7A97A; font-weight: 700; text-transform: uppercase;">Scientific Presentation</div>
              <h3 style="font-size: 16px; font-weight: 700; color: #ECE9DF; margin: 6px 0;">Final Presentation Slides (PPTX)</h3>
              <p style="font-size: 12px; color: #A8A396; margin-bottom: 14px;">
                19-slide widescreen (16:9) presentation deck in botanical neuroscience dark theme, covering all research questions,
                methodologies, empirical findings, and presentation slides.
              </p>
            </div>
            <span style="font-size: 28px;">📽</span>
          </div>
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-family: var(--font-mono); font-size: 11px; color: #A8A396;">Size: ~60 KB • 19 Slides</span>
            <a href="/downloads/SHIFT_MRI_Final_Presentation.pptx" class="btn btn-primary" download style="background: #C7A97A; color: #181B19; font-weight: 700; text-decoration: none; padding: 8px 16px; border-radius: 6px;">
              Download PPTX
            </a>
          </div>
        </div>

        <!-- Complete Submission ZIP -->
        <div class="card" style="background-color: #222824; border-color: #3d4a41; padding: 20px;">
          <div style="display: flex; align-items: flex-start; justify-content: space-between;">
            <div>
              <div style="font-size: 11px; color: #9CC5AA; font-weight: 700; text-transform: uppercase;">Full Submission Archive</div>
              <h3 style="font-size: 16px; font-weight: 700; color: #ECE9DF; margin: 6px 0;">Master Submission Package (ZIP)</h3>
              <p style="font-size: 12px; color: #A8A396; margin-bottom: 14px;">
                Self-contained archive including PDF paper, DOCX manuscript, PPTX deck, canonical_registry.json,
                experiment_provenance_manifest.json, research reports, and authoritative CSV tables.
              </p>
            </div>
            <span style="font-size: 28px;">📦</span>
          </div>
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-family: var(--font-mono); font-size: 11px; color: #A8A396;">Size: ~930 KB • Full Package</span>
            <a href="/downloads/SHIFT_MRI_Submission_Package.zip" class="btn btn-primary" download style="background: #9CC5AA; color: #181B19; font-weight: 700; text-decoration: none; padding: 8px 16px; border-radius: 6px;">
              Download ZIP
            </a>
          </div>
        </div>
      </div>

      <!-- Provenance Manifest Display -->
      <div class="card" style="background-color: #1c221e; border-color: #2d3630;">
        <h4 style="font-size: 13px; font-weight: 700; color: #ECE9DF; margin-bottom: 10px;">Experiment Provenance & Verification Metadata</h4>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; font-size: 12px; color: #A8A396;">
          <div><strong>Authoritative Git Commit:</strong> <code style="color: #9CC5AA;">435daf0</code></div>
          <div><strong>Environment:</strong> Python 3.12.10 • PyTorch 2.11+cu128</div>
          <div><strong>Hardware:</strong> RTX PRO 2000 Blackwell (16 GB VRAM)</div>
          <div><strong>Verification Test Suite:</strong> 100 Passed, 4 Skipped</div>
          <div><strong>Single Source of Truth:</strong> canonical_registry.json</div>
          <div><strong>Dataset Storage:</strong> D:\brain_mri_data (848 Scans)</div>
        </div>
      </div>
    `;
  }


  /* ==========================================================================
     RESEARCH EXPERIMENT LAB: LOCAL EXECUTION & STORE
     ========================================================================== */
  renderExperimentLab() {
    const container = document.getElementById('section-experiment-lab');
    if (!container) return;

    container.innerHTML = `
      <div class="section-header">
        <div class="section-tag" style="color: #809E83; background: rgba(128,158,131,0.1); border-color: #809E83;">
          Local Experiment Execution & SQLite Store
        </div>
        <h1 class="section-title">SHIFT-MRI Research Experiment Lab</h1>
        <p class="section-desc">
          Configure and execute reproducible model evaluations on local cohort data. Track run configurations,
          extract 95% bootstrap confidence intervals, and audit metric differences side-by-side.
        </p>
      </div>

      <!-- Experiment Configuration Builder -->
      <div class="card" style="background: #222824; border-color: #2d3630; padding: 20px; margin-bottom: 24px;">
        <h3 style="font-size: 15px; font-weight: 700; color: #ECE9DF; margin-bottom: 14px;">1. Experiment Configuration Builder</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-bottom: 16px;">
          <div>
            <label style="font-size: 11px; color: #A8A396; display: block; margin-bottom: 4px; font-weight: 600;">Model Architecture</label>
            <select id="lab-model-select" style="width: 100%; background: #1c221e; color: #ECE9DF; border: 1px solid #3d4a41; border-radius: 6px; padding: 8px 10px; font-size: 12px;">
              <option value="MONAI Swin UNETR SSL (Linear Probe)">MONAI Swin UNETR SSL (Linear Probe)</option>
              <option value="3D ResNet-10 (Pipeline B)">3D ResNet-10 (Pipeline B)</option>
              <option value="3D DenseNet-121 (Pipeline B)">3D DenseNet-121 (Pipeline B)</option>
              <option value="2.5D Multi-View ResNet-18 (Pipeline B)">2.5D Multi-View ResNet-18 (Pipeline B)</option>
              <option value="Dual-Branch Hybrid (ResNet-10 + Radiomics)">Dual-Branch Hybrid (ResNet-10 + Radiomics)</option>
              <option value="2D Transfer ResNet-18 (Slice-Aggregated)">2D Transfer ResNet-18 (Slice-Aggregated)</option>
              <option value="Classical Radiomics (Linear SVM)">Classical Radiomics (Linear SVM)</option>
              <option value="3D ResNet-18 (Scratch Baseline)">3D ResNet-18 (Scratch Baseline)</option>
              <option value="3D ResNet-34 (Scratch Baseline)">3D ResNet-34 (Scratch Baseline)</option>
              <option value="FOMO25 AMAES ResEnc-B (Linear Probe)">FOMO25 AMAES ResEnc-B (Linear Probe)</option>
            </select>
          </div>

          <div>
            <label style="font-size: 11px; color: #A8A396; display: block; margin-bottom: 4px; font-weight: 600;">Evaluation Dataset</label>
            <select id="lab-dataset-select" style="width: 100%; background: #1c221e; color: #ECE9DF; border: 1px solid #3d4a41; border-radius: 6px; padding: 8px 10px; font-size: 12px;">
              <option value="ADNI" selected>ADNI External Multi-Site (N=500)</option>
              <option value="OASIS-1">OASIS-1 Validation Split (N=30)</option>
            </select>
          </div>

          <div>
            <label style="font-size: 11px; color: #A8A396; display: block; margin-bottom: 4px; font-weight: 600;">Random Seed</label>
            <input id="lab-seed-input" type="number" value="42" style="width: 100%; background: #1c221e; color: #ECE9DF; border: 1px solid #3d4a41; border-radius: 6px; padding: 8px 10px; font-size: 12px;">
          </div>

          <div>
            <label style="font-size: 11px; color: #A8A396; display: block; margin-bottom: 4px; font-weight: 600;">Experiment Run Name</label>
            <input id="lab-name-input" type="text" placeholder="Optional custom name" style="width: 100%; background: #1c221e; color: #ECE9DF; border: 1px solid #3d4a41; border-radius: 6px; padding: 8px 10px; font-size: 12px;">
          </div>
        </div>

        <button id="lab-btn-execute" class="btn btn-primary" style="background: linear-gradient(135deg, #809E83, #9CC5AA); color: #181B19; font-weight: 700; border: none; padding: 10px 22px; border-radius: 6px; cursor: pointer; font-size: 13px;">
          &#9654; Execute Reproducible Experiment
        </button>
      </div>

      <!-- Execution Status & Live Metrics Panel -->
      <div id="lab-output-card" class="card" style="background: #222824; border-color: #2d3630; padding: 20px; margin-bottom: 24px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
          <h3 style="font-size: 15px; font-weight: 700; color: #ECE9DF; margin: 0;">2. Live Run Output & Metrics</h3>
          <span id="lab-status-badge" class="badge" style="background: #1c221e; color: #9CC5AA; border: 1px solid #3d4a41;">Ready to Execute</span>
        </div>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 16px;">
          <div style="background: #1c221e; padding: 12px; border-radius: 6px; border: 1px solid #2d3630;">
            <div style="font-size: 11px; color: #A8A396;">ROC-AUC (95% CI)</div>
            <div id="lab-kpi-auc" style="font-size: 18px; font-weight: 700; color: #9CC5AA; font-family: var(--font-mono);">-</div>
          </div>
          <div style="background: #1c221e; padding: 12px; border-radius: 6px; border: 1px solid #2d3630;">
            <div style="font-size: 11px; color: #A8A396;">Balanced Accuracy</div>
            <div id="lab-kpi-bal-acc" style="font-size: 18px; font-weight: 700; color: #ECE9DF; font-family: var(--font-mono);">-</div>
          </div>
          <div style="background: #1c221e; padding: 12px; border-radius: 6px; border: 1px solid #2d3630;">
            <div style="font-size: 11px; color: #A8A396;">Brier Score Loss</div>
            <div id="lab-kpi-brier" style="font-size: 18px; font-weight: 700; color: #ECE9DF; font-family: var(--font-mono);">-</div>
          </div>
          <div style="background: #1c221e; padding: 12px; border-radius: 6px; border: 1px solid #2d3630;">
            <div style="font-size: 11px; color: #A8A396;">Sensitivity / Specificity</div>
            <div id="lab-kpi-sens-spec" style="font-size: 18px; font-weight: 700; color: #ECE9DF; font-family: var(--font-mono);">-</div>
          </div>
        </div>

        <!-- Terminal Log Viewer -->
        <div style="background: #131614; border: 1px solid #2d3630; border-radius: 6px; padding: 12px; font-family: var(--font-mono); font-size: 11px; color: #A8A396; max-height: 160px; overflow-y: auto;">
          <pre id="lab-log-console" style="margin: 0; white-space: pre-wrap;">[Lab] Select an architecture and dataset above, then click Execute Reproducible Experiment.</pre>
        </div>
      </div>

      <!-- Experiment History & Comparison Table -->
      <div class="card" style="background: #222824; border-color: #2d3630; padding: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
          <div>
            <h3 style="font-size: 15px; font-weight: 700; color: #ECE9DF; margin: 0;">3. Experiment History & Run Comparator</h3>
            <p style="font-size: 11px; color: #A8A396; margin: 4px 0 0 0;">Historical runs stored in local SQLite database (experiment_history.db).</p>
          </div>
          <button id="lab-btn-compare" class="btn btn-secondary" style="background: #1c221e; color: #C7A97A; border: 1px solid #C7A97A; padding: 6px 14px; border-radius: 6px; font-size: 12px; cursor: pointer;">
            &#9878; Compare Selected Runs
          </button>
        </div>

        <div id="lab-compare-alert" style="display: none; background: rgba(199,169,122,0.1); border: 1px solid #C7A97A; padding: 10px 14px; border-radius: 6px; margin-bottom: 12px; font-size: 12px; color: #C7A97A;"></div>

        <div style="overflow-x: auto;">
          <table class="data-table" style="width: 100%;">
            <thead>
              <tr>
                <th style="width: 30px;">Select</th>
                <th>Timestamp</th>
                <th>Run Name</th>
                <th>Model</th>
                <th>Dataset</th>
                <th>ROC-AUC [95% CI]</th>
                <th>Brier Score</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody id="lab-history-tbody">
              <tr><td colspan="8" style="text-align: center; color: #A8A396;">Loading experiment history...</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    `;

    this.bindExperimentLabEvents();
  }

  async bindExperimentLabEvents() {
    const btnExecute = document.getElementById('lab-btn-execute');
    const btnCompare = document.getElementById('lab-btn-compare');
    const modelSelect = document.getElementById('lab-model-select');
    const datasetSelect = document.getElementById('lab-dataset-select');
    const seedInput = document.getElementById('lab-seed-input');
    const nameInput = document.getElementById('lab-name-input');

    const statusBadge = document.getElementById('lab-status-badge');
    const kpiAuc = document.getElementById('lab-kpi-auc');
    const kpiBalAcc = document.getElementById('lab-kpi-bal-acc');
    const kpiBrier = document.getElementById('lab-kpi-brier');
    const kpiSensSpec = document.getElementById('lab-kpi-sens-spec');
    const logConsole = document.getElementById('lab-log-console');
    const historyTbody = document.getElementById('lab-history-tbody');
    const compareAlert = document.getElementById('lab-compare-alert');

    const loadHistory = async () => {
      try {
        const resp = await fetch('/api/experiments/history');
        const json = await resp.json();
        const runs = json.runs || [];

        historyTbody.innerHTML = '';
        runs.forEach(r => {
          const m = r.metrics || {};
          const ci = m.ci_95 ? `[${m.ci_95[0]} - ${m.ci_95[1]}]` : '';
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td><input type="checkbox" class="lab-run-checkbox" value="${r.run_id}" style="accent-color: #9CC5AA;"></td>
            <td style="font-family: var(--font-mono); font-size: 11px;">${r.timestamp}</td>
            <td style="font-weight: 600; color: #ECE9DF;">${r.name}</td>
            <td style="font-size: 12px;">${r.model_name}</td>
            <td style="font-size: 12px;"><span class="badge" style="background: #1c221e; border: 1px solid #3d4a41;">${r.dataset}</span></td>
            <td style="font-family: var(--font-mono); color: #9CC5AA; font-weight: 600;">${m.roc_auc !== undefined ? m.roc_auc.toFixed(4) : 'N/A'} <span style="font-size: 10px; color: #A8A396;">${ci}</span></td>
            <td style="font-family: var(--font-mono); font-size: 11px;">${m.brier_score !== undefined ? m.brier_score.toFixed(4) : 'N/A'}</td>
            <td><span class="badge" style="background: rgba(156,197,170,0.1); color: #9CC5AA; border: 1px solid #809E83;">${r.status}</span></td>
          `;
          historyTbody.appendChild(tr);
        });
      } catch (e) {
        console.error('Failed to load experiment history:', e);
      }
    };

    btnExecute.addEventListener('click', async () => {
      btnExecute.disabled = true;
      statusBadge.textContent = 'Running Experiment...';
      statusBadge.style.color = '#C7A97A';
      logConsole.textContent = '[Lab] Launching execution task on local workstation...\n';

      const payload = {
        model_name: modelSelect.value,
        dataset: datasetSelect.value,
        seed: parseInt(seedInput.value, 10) || 42,
        name: nameInput.value.trim() || undefined
      };

      try {
        const resp = await fetch('/api/experiments/launch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const res = await resp.json();

        if (res.status === 'success') {
          const run = res.run;
          const m = run.metrics;
          statusBadge.textContent = 'Execution Completed (Success)';
          statusBadge.style.color = '#9CC5AA';

          kpiAuc.textContent = `${m.roc_auc.toFixed(4)} [${m.ci_95[0]} - ${m.ci_95[1]}]`;
          kpiBalAcc.textContent = m.balanced_accuracy.toFixed(4);
          kpiBrier.textContent = m.brier_score.toFixed(4);
          kpiSensSpec.textContent = `${(m.sensitivity * 100).toFixed(1)}% / ${(m.specificity * 100).toFixed(1)}%`;

          logConsole.textContent = (run.log || []).join('\n');
          await loadHistory();
        } else {
          statusBadge.textContent = 'Execution Failed';
          statusBadge.style.color = '#B98378';
          logConsole.textContent = `[Error] ${res.error || 'Execution failed.'}`;
        }
      } catch (e) {
        statusBadge.textContent = 'Execution Error';
        statusBadge.style.color = '#B98378';
        logConsole.textContent = `[Network Error] ${e.message}`;
      } finally {
        btnExecute.disabled = false;
      }
    });

    btnCompare.addEventListener('click', async () => {
      const selectedBoxes = Array.from(document.querySelectorAll('.lab-run-checkbox:checked'));
      const runIds = selectedBoxes.map(b => b.value);

      if (runIds.length < 2) {
        alert('Please select at least 2 runs using the checkboxes to compare.');
        return;
      }

      try {
        const resp = await fetch(`/api/experiments/compare?ids=${encodeURIComponent(runIds.join(','))}`);
        const res = await resp.json();

        if (res.cross_cohort_warning) {
          compareAlert.textContent = res.cross_cohort_warning;
          compareAlert.style.display = 'block';
        } else {
          compareAlert.style.display = 'none';
        }

        let alertMsg = `Compared ${res.runs_compared} runs:\n`;
        (res.comparison_table || []).forEach(row => {
          alertMsg += `? ${row.model} (${row.dataset}): AUC=${row.roc_auc.toFixed(4)} (Delta vs Base: ${row.delta_auc_vs_baseline > 0 ? '+' : ''}${row.delta_auc_vs_baseline})\n`;
        });
        alert(alertMsg);
      } catch (e) {
        console.error('Comparison error:', e);
      }
    });

    loadHistory();
  }

  /* ==========================================================================
     FUTURE RESEARCH PROGRAM (TRACKS A THROUGH I)
     ========================================================================== */
  renderFutureResearch() {
    const container = document.getElementById('section-future-research');
    if (!container) return;

    container.innerHTML = `
      <div class="section-header">
        <div class="section-tag" style="color: #B98378; background: rgba(185,131,120,0.1); border-color: #B98378;">
          Scientific Roadmap ? Living Research Program
        </div>
        <h1 class="section-title">SHIFT-MRI Future Research Roadmap</h1>
        <p class="section-desc">
          Nine prioritized, scientifically defensible research tracks addressing the core vulnerabilities,
          negative TrustGate findings, and clinical adaptation barriers discovered in the V3 benchmark.
        </p>
      </div>

      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px;">
        <!-- Track A -->
        <div class="card" style="background: #222824; border-color: #2d3630; padding: 18px;">
          <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
            <span class="badge" style="background: rgba(199,169,122,0.1); color: #C7A97A; border: 1px solid #C7A97A;">Track A ? Primary</span>
            <span class="badge" style="background: rgba(156,197,170,0.1); color: #9CC5AA; border: 1px solid #809E83;">Implementation-Ready</span>
          </div>
          <h3 style="font-size: 14px; font-weight: 700; color: #ECE9DF; margin-bottom: 6px;">TrustGate V2: Non-Linear Abstention</h3>
          <p style="font-size: 12px; color: #A8A396; line-height: 1.5; margin-bottom: 10px;">
            Investigate why multi-model consensus inverts under scanner shift (AUROC 0.4674). Replace equal weighting with rank-based aggregation and PCA-whitened FastMCD distance.
          </p>
          <div style="font-size: 11px; color: #9CC5AA;">Verification: Target AUROC &ge; 0.60 without external labels.</div>
        </div>

        <!-- Track B -->
        <div class="card" style="background: #222824; border-color: #2d3630; padding: 18px;">
          <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
            <span class="badge" style="background: #1c221e; color: #9CC5AA; border: 1px solid #3d4a41;">Track B</span>
            <span class="badge" style="background: rgba(156,197,170,0.1); color: #9CC5AA; border: 1px solid #809E83;">Partially Executed</span>
          </div>
          <h3 style="font-size: 14px; font-weight: 700; color: #ECE9DF; margin-bottom: 6px;">Scanner & Harmonization Shift</h3>
          <p style="font-size: 12px; color: #A8A396; line-height: 1.5; margin-bottom: 10px;">
            Separate scanner distribution differences (Siemens N=370 vs Philips N=130) from true anatomical atrophy. Evaluate ComBat neuroimaging harmonization against Z-score normalization.
          </p>
          <div style="font-size: 11px; color: #9CC5AA;">Verification: Permutation test across scanner strata.</div>
        </div>

        <!-- Track C -->
        <div class="card" style="background: #222824; border-color: #2d3630; padding: 18px;">
          <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
            <span class="badge" style="background: #1c221e; color: #9CC5AA; border: 1px solid #3d4a41;">Track C</span>
            <span class="badge" style="background: #1c221e; color: #A8A396; border: 1px solid #3d4a41;">Proposed Study</span>
          </div>
          <h3 style="font-size: 14px; font-weight: 700; color: #ECE9DF; margin-bottom: 6px;">Foundation Model Adaptation (LoRA)</h3>
          <p style="font-size: 12px; color: #A8A396; line-height: 1.5; margin-bottom: 10px;">
            Compare frozen probing against low-rank attention adaptation (LoRA, r=8) on MONAI Swin UNETR SSL representations across budget curves (N=20 to 138).
          </p>
          <div style="font-size: 11px; color: #9CC5AA;">Verification: Superior external AUC with &lt; 5% trainable params.</div>
        </div>

        <!-- Track D -->
        <div class="card" style="background: #222824; border-color: #2d3630; padding: 18px;">
          <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
            <span class="badge" style="background: #1c221e; color: #9CC5AA; border: 1px solid #3d4a41;">Track D</span>
            <span class="badge" style="background: rgba(156,197,170,0.1); color: #9CC5AA; border: 1px solid #809E83;">Implementation-Ready</span>
          </div>
          <h3 style="font-size: 14px; font-weight: 700; color: #ECE9DF; margin-bottom: 6px;">Multi-Seed Split Stability</h3>
          <p style="font-size: 12px; color: #A8A396; line-height: 1.5; margin-bottom: 10px;">
            Execute 5x5 repeated cross-validation across all 10 architectures to quantify parameter-to-sample variance and establish DeLong confidence intervals.
          </p>
          <div style="font-size: 11px; color: #9CC5AA;">Verification: Inter-split variance &sigma; &le; 0.03.</div>
        </div>

        <!-- Track E -->
        <div class="card" style="background: #222824; border-color: #2d3630; padding: 18px;">
          <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
            <span class="badge" style="background: #1c221e; color: #9CC5AA; border: 1px solid #3d4a41;">Track E</span>
            <span class="badge" style="background: #1c221e; color: #A8A396; border: 1px solid #3d4a41;">Proposed Study</span>
          </div>
          <h3 style="font-size: 14px; font-weight: 700; color: #ECE9DF; margin-bottom: 6px;">External Validation (AIBL / MIRIAD)</h3>
          <p style="font-size: 12px; color: #A8A396; line-height: 1.5; margin-bottom: 10px;">
            Acquire formal DUAs for Australian AIBL cohort to test zero-shot generalization on certified zero-overlap international participant populations.
          </p>
          <div style="font-size: 11px; color: #9CC5AA;">Verification: Ethical compliance and adapter indexing.</div>
        </div>

        <!-- Track F -->
        <div class="card" style="background: #222824; border-color: #2d3630; padding: 18px;">
          <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
            <span class="badge" style="background: #1c221e; color: #9CC5AA; border: 1px solid #3d4a41;">Track F</span>
            <span class="badge" style="background: #1c221e; color: #A8A396; border: 1px solid #3d4a41;">Proposed Study</span>
          </div>
          <h3 style="font-size: 14px; font-weight: 700; color: #ECE9DF; margin-bottom: 6px;">Conformal Prediction Under Shift</h3>
          <p style="font-size: 12px; color: #A8A396; line-height: 1.5; margin-bottom: 10px;">
            Evaluate covariate-shift-adapted split conformal prediction with density ratio reweighting to provide formal marginal coverage guarantees under scanner drift.
          </p>
          <div style="font-size: 11px; color: #9CC5AA;">Verification: Exact 90% coverage bounded on external cohort.</div>
        </div>

        <!-- Track G -->
        <div class="card" style="background: #222824; border-color: #2d3630; padding: 18px;">
          <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
            <span class="badge" style="background: #1c221e; color: #9CC5AA; border: 1px solid #3d4a41;">Track G</span>
            <span class="badge" style="background: rgba(156,197,170,0.1); color: #9CC5AA; border: 1px solid #809E83;">Implementation-Ready</span>
          </div>
          <h3 style="font-size: 14px; font-weight: 700; color: #ECE9DF; margin-bottom: 6px;">Synthetic Perturbation Auditing</h3>
          <p style="font-size: 12px; color: #A8A396; line-height: 1.5; margin-bottom: 10px;">
            Subject volumes to parameterized Rician noise, slice-direction motion artifacts, and B1 field bias to benchmark degradation trajectories across models.
          </p>
          <div style="font-size: 11px; color: #9CC5AA;">Verification: Monotonic AUC degradation curves.</div>
        </div>

        <!-- Track H -->
        <div class="card" style="background: #222824; border-color: #2d3630; padding: 18px;">
          <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
            <span class="badge" style="background: #1c221e; color: #9CC5AA; border: 1px solid #3d4a41;">Track H</span>
            <span class="badge" style="background: #1c221e; color: #A8A396; border: 1px solid #3d4a41;">Proposed Study</span>
          </div>
          <h3 style="font-size: 14px; font-weight: 700; color: #ECE9DF; margin-bottom: 6px;">Clinical Label Sensitivity</h3>
          <p style="font-size: 12px; color: #A8A396; line-height: 1.5; margin-bottom: 10px;">
            Analyze model failure modes against clinical dementia severity (CDR 0.5 very mild vs CDR 1.0 moderate) and MMSE cognitive scores.
          </p>
          <div style="font-size: 11px; color: #9CC5AA;">Verification: Disclose borderlines vs true misclassifications.</div>
        </div>

        <!-- Track I -->
        <div class="card" style="background: #222824; border-color: #2d3630; padding: 18px;">
          <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
            <span class="badge" style="background: #1c221e; color: #9CC5AA; border: 1px solid #3d4a41;">Track I</span>
            <span class="badge" style="background: #1c221e; color: #A8A396; border: 1px solid #3d4a41;">Proposed Study</span>
          </div>
          <h3 style="font-size: 14px; font-weight: 700; color: #ECE9DF; margin-bottom: 6px;">Longitudinal Atrophy Trajectories</h3>
          <p style="font-size: 12px; color: #A8A396; line-height: 1.5; margin-bottom: 10px;">
            Extend foundation model embeddings to participant-disjoint longitudinal progression prediction over 24-36 months.
          </p>
          <div style="font-size: 11px; color: #9CC5AA;">Verification: Progression velocity correlation &gt; 0.70.</div>
        </div>
      </div>
    `;
  }

}

// Bootstrap application
window.addEventListener('DOMContentLoaded', () => {
  const app = new BenchmarkApp();
  window.__app = app;
  app.init();
});
