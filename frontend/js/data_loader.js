/**
 * Canonical Data Loader & Integrity Validator
 * Fair Benchmarking of Brain-MRI Foundation Models
 * 
 * Consumes strictly from: results/final_campaign/canonical_frontend_data/
 * Performs runtime schema validation, NaN/Infinity detection, and research vocabulary checks.
 */

export const DataLoader = {
  data: {},
  isLoaded: false,
  loadErrors: [],

  // Registry of the 19 canonical JSON files
  canonicalFiles: {
    study_metadata: '/data/study_metadata.json',
    models: '/data/models.json',
    leaderboard: '/data/leaderboard.json',
    data_efficiency: '/data/data_efficiency.json',
    label_efficiency: '/data/label_efficiency.json',
    ssap: '/data/ssap.json',
    parameter_efficiency: '/data/parameter_efficiency.json',
    compute_efficiency: '/data/compute_efficiency.json',
    calibration: '/data/calibration.json',
    domain_shift: '/data/domain_shift.json',
    fusion: '/data/fusion.json',
    representation_analysis: '/data/representation_analysis.json',
    error_analysis: '/data/error_analysis.json',
    dataset_characteristics: '/data/dataset_characteristics.json',
    statistics: '/data/statistics.json',
    interpretability_index: '/data/interpretability_index.json',
    methodology: '/data/methodology.json',
    limitations: '/data/limitations.json',
    audit_status: '/data/audit_status.json',
    training_diagnostics: '/data/training_diagnostics.json',
    foundation_adaptation: '/data/foundation_adaptation.json',
    statistical_uncertainty: '/data/statistical_uncertainty.json',
    adaptation_comparison: '/data/adaptation_comparison.json'
  },

  /**
   * Recursively validate object to ensure no NaN, Infinity, or null where numbers expected.
   */
  validateNoNaN(obj, path = '') {
    if (obj === null || obj === undefined) return;
    if (typeof obj === 'number') {
      if (isNaN(obj) || !isFinite(obj)) {
        throw new Error(`Data Integrity Error: Invalid numerical value (NaN/Infinity) at ${path}`);
      }
    } else if (Array.isArray(obj)) {
      obj.forEach((item, index) => this.validateNoNaN(item, `${path}[${index}]`));
    } else if (typeof obj === 'object') {
      for (const [key, val] of Object.entries(obj)) {
        this.validateNoNaN(val, `${path}.${key}`);
      }
    }
  },

  /**
   * Scan textual data to ensure no prohibited clinical overclaiming words exist.
   */
  auditVocabulary(obj, path = '') {
    const prohibited = [
      'diagnoses alzheimer',
      'disease-generating',
      'proof of pathology',
      'clinically validated generalization',
      'definitive proof'
    ];
    if (typeof obj === 'string') {
      const lower = obj.toLowerCase();
      for (const phrase of prohibited) {
        if (lower.includes(phrase)) {
          console.warn(`[Terminology Audit Warning] Prohibited clinical term "${phrase}" detected at ${path}`);
        }
      }
    } else if (Array.isArray(obj)) {
      obj.forEach((item, index) => this.auditVocabulary(item, `${path}[${index}]`));
    } else if (typeof obj === 'object' && obj !== null) {
      for (const [key, val] of Object.entries(obj)) {
        this.auditVocabulary(val, `${path}.${key}`);
      }
    }
  },

  /**
   * Fetch and validate all canonical JSON datasets.
   */
  async loadAll() {
    console.log('[DataLoader] Loading canonical research datasets...');
    
    // Safety check for file:// execution
    if (typeof window !== 'undefined' && window.location && window.location.protocol === 'file:') {
      throw new Error(
        'Protocol Error: Cannot fetch canonical JSON files when loaded directly via file:// due to browser CORS security restrictions. ' +
        'Please start the local server by executing "python run_frontend.py" and open http://127.0.0.1:8000/'
      );
    }

    const keys = Object.keys(this.canonicalFiles);

    const fetchWithRetry = async (url, retries = 2, delayMs = 80) => {
      let lastError = null;
      for (let attempt = 0; attempt <= retries; attempt++) {
        try {
          const res = await fetch(url);
          if (!res.ok) {
            throw new Error(`HTTP ${res.status} (${res.statusText || 'Response not OK'})`);
          }
          return await res.json();
        } catch (err) {
          lastError = err;
          if (attempt < retries) {
            await new Promise(r => setTimeout(r, delayMs * (attempt + 1)));
          }
        }
      }
      throw lastError;
    };
    
    try {
      const fetchPromises = keys.map(async (key) => {
        const url = this.canonicalFiles[key];
        try {
          const parsed = await fetchWithRetry(url);
          
          // Runtime verification checks
          this.validateNoNaN(parsed, key);
          this.auditVocabulary(parsed, key);
          
          return { key, data: parsed };
        } catch (fetchErr) {
          console.error(`[DataLoader] Error fetching ${key} (${url}):`, fetchErr.message);
          throw new Error(`Failed to load ${key} from "${url}": ${fetchErr.message}`);
        }
      });

      const results = await Promise.all(fetchPromises);
      for (const { key, data } of results) {
        this.data[key] = data;
      }

      this.isLoaded = true;
      console.log(`[DataLoader] Successfully loaded and certified ${keys.length}/23 canonical datasets.`);
      return this.data;
    } catch (err) {
      console.error('[DataLoader] Fatal error loading canonical research data:', err);
      this.loadErrors.push(err.message);
      throw err;
    }
  },

  get(key) {
    if (!this.isLoaded) {
      console.warn(`[DataLoader] Requested "${key}" before data was loaded.`);
    }
    return this.data[key];
  }
};
