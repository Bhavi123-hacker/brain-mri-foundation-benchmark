/**
 * Scientific Visualization Engine (Chart.js wrapper)
 * Fair Benchmarking of Brain-MRI Foundation Models
 * 
 * Provides publication-grade styling, exact hover tooltips, and strict evaluation labeling.
 */

export const ChartEngine = {
  instances: {},

  destroy(canvasId) {
    if (this.instances[canvasId]) {
      this.instances[canvasId].destroy();
      delete this.instances[canvasId];
    }
  },

  defaultOptions(title, xLabel, yLabel, yMin = 0.4, yMax = 1.05) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 350 },
      plugins: {
        legend: {
          position: 'top',
          labels: {
            color: '#94a3b8',
            font: { family: 'Inter', size: 12, weight: '500' },
            boxWidth: 14,
            usePointStyle: true
          }
        },
        title: {
          display: !!title,
          text: title,
          color: '#f1f5f9',
          font: { family: 'Inter', size: 14, weight: '600' },
          padding: { bottom: 16 }
        },
        tooltip: {
          backgroundColor: '#0f172a',
          titleColor: '#93c5fd',
          bodyColor: '#f1f5f9',
          borderColor: '#3b82f6',
          borderWidth: 1,
          padding: 12,
          boxPadding: 6,
          usePointStyle: true,
          callbacks: {
            label: function (ctx) {
              const val = ctx.parsed.y !== undefined ? ctx.parsed.y : ctx.parsed.x;
              return ` ${ctx.dataset.label}: ${typeof val === 'number' ? val.toFixed(4) : val}`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(30, 45, 77, 0.4)' },
          ticks: { color: '#94a3b8', font: { family: 'Inter', size: 11 } },
          title: {
            display: !!xLabel,
            text: xLabel,
            color: '#94a3b8',
            font: { family: 'Inter', size: 12, weight: '500' }
          }
        },
        y: {
          min: yMin,
          max: yMax,
          grid: { color: 'rgba(30, 45, 77, 0.4)' },
          ticks: { color: '#94a3b8', font: { family: 'Inter', size: 11 } },
          title: {
            display: !!yLabel,
            text: yLabel,
            color: '#94a3b8',
            font: { family: 'Inter', size: 12, weight: '500' }
          }
        }
      }
    };
  },

  /**
   * Multi-metric comparison across models
   */
  renderModelComparison(canvasId, models) {
    this.destroy(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const labels = models.map(m => m.display_name.replace('Supervised 3D ', '').replace('MONAI ', ''));
    
    this.instances[canvasId] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Development CV ROC-AUC (N=42)',
            data: models.map(m => m.development_cv_auc),
            backgroundColor: 'rgba(59, 130, 246, 0.85)',
            borderColor: '#3b82f6',
            borderWidth: 1,
            borderRadius: 4
          },
          {
            label: 'Locked Holdout ROC-AUC (N=8)',
            data: models.map(m => m.locked_test_auc),
            backgroundColor: 'rgba(16, 185, 129, 0.85)',
            borderColor: '#10b981',
            borderWidth: 1,
            borderRadius: 4
          },
          {
            label: 'External ADNI Transfer ROC-AUC (N=8)',
            data: models.map(m => m.ADNI_transfer_auc),
            backgroundColor: 'rgba(245, 158, 11, 0.85)',
            borderColor: '#f59e0b',
            borderWidth: 1,
            borderRadius: 4
          }
        ]
      },
      options: this.defaultOptions('Cross-Context Model Performance Comparison', 'Evaluated Model', 'Area Under ROC (AUC)', 0.4, 1.05)
    });
  },

  /**
   * Data efficiency curves across label fractions
   */
  renderDataEfficiency(canvasId, dataEfficiency) {
    this.destroy(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    // Group by model
    const models = [...new Set(dataEfficiency.map(d => d.model))];
    const colors = {
      '3D ResNet-18': '#3b82f6',
      '3D ResNet-34': '#6366f1',
      'BrainIAC': '#06b6d4',
      'FOMO25 Winner': '#f59e0b',
      'MONAI Swin UNETR SSL': '#ec4899',
      'ResNet-34 + Swin Ensemble': '#10b981'
    };

    const datasets = models.map(model => {
      const records = dataEfficiency.filter(d => d.model === model).sort((a, b) => a.n_train - b.n_train);
      const color = colors[model] || '#a855f7';
      return {
        label: model,
        data: records.map(r => ({ x: r.n_train, y: r.mean_auc, std: r.std_auc, acc: r.mean_accuracy })),
        borderColor: color,
        backgroundColor: color,
        borderWidth: 2.5,
        tension: 0.15,
        pointRadius: 5,
        pointHoverRadius: 7
      };
    });

    const opts = this.defaultOptions('Sample-Size Scaling Trajectories (10% to 100% Training Pool)', 'Labeled Training Subjects (N)', 'Mean ROC-AUC across Seeds', 0.4, 1.02);
    opts.plugins.tooltip.callbacks.afterLabel = function(ctx) {
      const raw = ctx.raw;
      return `Std Dev: ±${(raw.std || 0).toFixed(4)} | Mean Acc: ${(raw.acc * 100).toFixed(1)}%`;
    };

    this.instances[canvasId] = new Chart(ctx, {
      type: 'line',
      data: { datasets },
      options: opts
    });
  },

  /**
   * Parameter efficiency Pareto frontier
   */
  renderParameterFrontier(canvasId, paramData) {
    this.destroy(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const datasets = [
      {
        label: 'Candidate Models (Trainable Params vs CV AUC)',
        data: paramData.map(p => ({
          x: p.trainable_params,
          y: p.cv_auc,
          name: p.model,
          paradigm: p.paradigm,
          total: p.total_params,
          ratio: p.trainable_ratio_pct
        })),
        backgroundColor: paramData.map(p => p.pareto_optimal ? '#10b981' : '#3b82f6'),
        borderColor: '#fff',
        borderWidth: 1,
        pointRadius: 8,
        pointHoverRadius: 10
      }
    ];

    const opts = this.defaultOptions('Parameter Efficiency: Trainable Parameters vs In-Domain CV AUC', 'Trainable Parameters (Log Scale)', 'Development CV ROC-AUC', 0.45, 1.05);
    opts.scales.x.type = 'logarithmic';
    opts.plugins.tooltip.callbacks.title = function(ctx) {
      return ctx[0].raw.name;
    };
    opts.plugins.tooltip.callbacks.label = function(ctx) {
      const d = ctx.raw;
      return [
        `Paradigm: ${d.paradigm}`,
        `Trainable Params: ${d.x.toLocaleString()} (${d.ratio}% of total)`,
        `Total Params: ${d.total.toLocaleString()}`,
        `Development CV AUC: ${d.y.toFixed(4)}`
      ];
    };

    this.instances[canvasId] = new Chart(ctx, {
      type: 'scatter',
      data: { datasets },
      options: opts
    });
  },

  /**
   * Reliability Diagram / Calibration Curve
   */
  renderCalibrationChart(canvasId, calibrationData) {
    this.destroy(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const models = calibrationData.map(c => c.model);
    
    this.instances[canvasId] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: models,
        datasets: [
          {
            label: 'Raw Brier Score (Lower is Better)',
            data: calibrationData.map(c => c.raw_brier_score),
            backgroundColor: 'rgba(239, 68, 68, 0.8)',
            borderColor: '#ef4444',
            borderWidth: 1,
            borderRadius: 4
          },
          {
            label: 'Recomputed Out-of-Sample ECE (Lower is Better)',
            data: calibrationData.map(c => c.recomputed_ece),
            backgroundColor: 'rgba(59, 130, 246, 0.8)',
            borderColor: '#3b82f6',
            borderWidth: 1,
            borderRadius: 4
          }
        ]
      },
      options: this.defaultOptions('Calibration Error Audit across Models', 'Model Architecture', 'Error Magnitude', 0, 0.45)
    });
  },

  /**
   * Domain Shift comparison
   */
  renderDomainShiftChart(canvasId, domainData) {
    this.destroy(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    this.instances[canvasId] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: domainData.map(d => d.model),
        datasets: [
          {
            label: 'ADNI External Transfer AUC (3.0T)',
            data: domainData.map(d => d.auc),
            backgroundColor: 'rgba(16, 185, 129, 0.85)',
            borderColor: '#10b981',
            borderWidth: 1,
            borderRadius: 4
          },
          {
            label: 'ADNI Balanced Accuracy (Dev Threshold)',
            data: domainData.map(d => d.balanced_accuracy),
            backgroundColor: 'rgba(59, 130, 246, 0.85)',
            borderColor: '#3b82f6',
            borderWidth: 1,
            borderRadius: 4
          }
        ]
      },
      options: this.defaultOptions('Cross-Dataset Generalization: OASIS-1 (1.5T) to ADNI (3.0T)', 'Model Architecture', 'Transfer Performance (N=8 Test Cohort)', 0.4, 1.05)
    });
  },

  /**
   * Seed-Stability Adjusted Performance (SSAP)
   */
  renderSSAPChart(canvasId, ssapData) {
    this.destroy(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    // Filter to 100% data fraction
    const fullData = ssapData.filter(s => s.fraction === '100%');

    this.instances[canvasId] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: fullData.map(s => s.model),
        datasets: [
          {
            label: 'Mean AUC (Seeds 42-46)',
            data: fullData.map(s => s.mean_auc),
            backgroundColor: 'rgba(59, 130, 246, 0.75)',
            borderColor: '#3b82f6',
            borderWidth: 1,
            borderRadius: 4
          },
          {
            label: 'SSAP (λ = 0.50 Penalty)',
            data: fullData.map(s => s.ssap_lambda_0_50),
            backgroundColor: 'rgba(16, 185, 129, 0.85)',
            borderColor: '#10b981',
            borderWidth: 1,
            borderRadius: 4
          }
        ]
      },
      options: this.defaultOptions('Seed-Stability Adjusted Performance (SSAP = μ_AUC − λ·σ_AUC)', 'Model', 'Score', 0.4, 1.02)
    });
  }
};
