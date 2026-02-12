"""HTML report generation for fmridenoiser participant-level analyses.

This module generates standalone HTML reports with:
- Denoising parameters documentation
- Confound regression details
- Denoising quality histograms
- FD-based censoring visualization
- Resampling information
- Reproducibility command
"""

import base64
import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fmridenoiser.core.version import __version__

logger = logging.getLogger(__name__)


REPORT_CSS = """
<style>
:root {
    --primary-color: #0891b2;
    --primary-dark: #0e7490;
    --secondary-color: #6366f1;
    --success-color: #10b981;
    --warning-color: #f59e0b;
    --danger-color: #ef4444;
    --gray-50: #f9fafb;
    --gray-100: #f3f4f6;
    --gray-200: #e5e7eb;
    --gray-600: #4b5563;
    --gray-700: #374151;
    --gray-800: #1f2937;
    --gray-900: #111827;
}
* { box-sizing: border-box; }
body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6; color: var(--gray-800);
    background-color: var(--gray-50); margin: 0; padding: 0;
}
.container { max-width: 1200px; margin: 0 auto; padding: 20px; }
.nav-bar {
    position: sticky; top: 0; background: white;
    border-bottom: 1px solid var(--gray-200); padding: 15px 20px;
    z-index: 100; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.nav-content {
    max-width: 1200px; margin: 0 auto;
    display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;
}
.nav-brand { font-weight: 700; font-size: 1.25em; color: var(--primary-color); }
.nav-links { display: flex; gap: 20px; flex-wrap: wrap; }
.nav-links a { color: var(--gray-600); text-decoration: none; font-size: 0.9em; }
.nav-links a:hover { color: var(--primary-color); }
.header {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    color: white; padding: 40px; margin-bottom: 30px; border-radius: 12px;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
}
.header h1 { margin: 0 0 10px 0; font-size: 2.2em; font-weight: 700; }
.header .subtitle { font-size: 1.1em; opacity: 0.9; }
.header .meta-info { margin-top: 20px; display: flex; gap: 30px; flex-wrap: wrap; font-size: 0.95em; }
.section {
    background: white; padding: 30px; border-radius: 12px;
    margin-bottom: 25px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.section h2 {
    color: var(--gray-800); border-bottom: 2px solid var(--primary-color);
    padding-bottom: 12px; margin-top: 0; margin-bottom: 25px; font-size: 1.5em;
}
.section h3 { color: var(--gray-700); margin-top: 25px; margin-bottom: 15px; font-size: 1.2em; }
.metrics-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 20px; margin: 20px 0;
}
.metric-card {
    background: var(--gray-50); padding: 20px; border-radius: 10px;
    text-align: center; border-left: 4px solid var(--primary-color);
}
.metric-value { font-size: 2em; font-weight: 700; color: var(--primary-color); }
.metric-label { color: var(--gray-600); font-size: 0.9em; margin-top: 5px; }
.param-table { width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 0.95em; }
.param-table th, .param-table td { padding: 12px 15px; text-align: left; border-bottom: 1px solid var(--gray-200); }
.param-table th { background: var(--gray-100); font-weight: 600; color: var(--gray-700); }
.param-table code { background: var(--gray-100); padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 0.9em; }
.figure-container { margin: 25px 0; text-align: center; }
.figure-wrapper {
    display: inline-block; background: white; padding: 15px; border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.figure-wrapper img { max-width: 100%; height: auto; border-radius: 6px; }
.figure-caption { color: var(--gray-600); font-style: italic; margin-top: 12px; font-size: 0.95em; }
.code-block {
    background: var(--gray-900); color: #e5e7eb; padding: 20px; border-radius: 8px;
    overflow-x: auto; font-family: monospace; font-size: 0.9em; margin: 15px 0;
}
.alert {
    padding: 15px 20px; border-radius: 8px; margin: 15px 0;
    display: flex; align-items: flex-start; gap: 12px;
}
.alert-success { background: #d1fae5; border: 1px solid var(--success-color); color: #065f46; }
.alert-warning { background: #fef3c7; border: 1px solid var(--warning-color); color: #92400e; }
.alert-danger { background: #fee2e2; border: 1px solid var(--danger-color); color: #991b1b; }
.alert-info { background: #dbeafe; border: 1px solid var(--primary-color); color: #1e40af; }
.info-box {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border: 1px solid #bae6fd; border-left: 4px solid var(--primary-color);
    border-radius: 8px; padding: 20px; margin: 20px 0;
}
.info-box h4 { margin-top: 0; color: var(--primary-dark); }
.toc { background: white; padding: 25px; border-radius: 12px; margin-bottom: 30px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.toc h2 { margin-top: 0; color: var(--gray-800); border-bottom: 2px solid var(--primary-color); padding-bottom: 10px; }
.toc-list { list-style: none; padding: 0; margin: 0; columns: 2; column-gap: 40px; }
.toc-list li { margin-bottom: 8px; }
.toc-list a { color: var(--gray-700); text-decoration: none; }
.toc-list a:hover { color: var(--primary-color); }
.toc-number { background: var(--gray-100); color: var(--gray-600); padding: 2px 8px; border-radius: 4px; font-size: 0.85em; font-weight: 600; }
.reference-item { margin-bottom: 15px; padding-left: 25px; position: relative; }
.reference-item::before { content: "\\2022"; position: absolute; left: 8px; color: var(--primary-color); }
.reference-item a { color: var(--primary-color); text-decoration: none; }
.footer { text-align: center; padding: 30px 20px; color: var(--gray-600); font-size: 0.9em; border-top: 1px solid var(--gray-200); margin-top: 40px; }
.footer a { color: var(--primary-color); text-decoration: none; }
</style>
"""


class DenoisingReportGenerator:
    """Generate HTML reports for fmridenoiser participant-level analyses."""

    def __init__(
        self,
        subject_id: str,
        config: Any,
        output_dir: Path,
        confounds_df: Optional[pd.DataFrame] = None,
        selected_confounds: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
        desc: Optional[str] = None,
        label: Optional[str] = None,
        censoring_summary: Optional[Dict[str, Any]] = None,
        resampling_info: Optional[Dict[str, Any]] = None,
    ):
        self.subject_id = subject_id
        self.config = config
        self.output_dir = Path(output_dir)
        self.desc = desc
        self.label = label
        self._logger = logger or logging.getLogger(__name__)

        self.confounds_used: List[str] = selected_confounds or []
        self.confounds_df: Optional[pd.DataFrame] = confounds_df
        self.denoising_histogram_data: Optional[Dict[str, Any]] = None

        self.censoring_summary: Optional[Dict[str, Any]] = censoring_summary
        self.resampling_info: Optional[Dict[str, Any]] = resampling_info

        self.toc_items: List[Tuple[str, str]] = []
        self.command_line: Optional[str] = None
        self.figures_dir: Optional[Path] = None

    def add_denoising_histogram_data(self, data: Dict[str, Any]) -> None:
        self.denoising_histogram_data = data

    def set_command_line(self, command: str) -> None:
        self.command_line = command

    def _figure_to_base64(self, fig: plt.Figure, dpi: int = 150) -> str:
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        buffer.seek(0)
        img_data = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        return img_data

    def generate(self) -> Path:
        """Generate the complete HTML report."""
        self._logger.info(f"Generating denoising report for {self.subject_id}")

        # Determine output path
        if self.subject_id.startswith('sub-'):
            parts = self.subject_id.split('_')
            sub_part = parts[0]
            ses_part = None
            for p in parts[1:]:
                if p.startswith('ses-'):
                    ses_part = p
            output_path = self.output_dir / sub_part
            if ses_part:
                output_path = output_path / ses_part
        else:
            output_path = self.output_dir / f"sub-{self.subject_id}"

        output_path.mkdir(parents=True, exist_ok=True)
        self.figures_dir = output_path / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Build sections
        sections_html = ""
        sections_html += self._build_overview_section()
        sections_html += self._build_parameters_section()
        sections_html += self._build_resampling_section()
        sections_html += self._build_confounds_section()
        sections_html += self._build_censoring_section()
        sections_html += self._build_qa_section()
        sections_html += self._build_command_section()
        sections_html += self._build_references_section()

        nav_html = self._build_nav_bar()
        toc_html = self._build_toc()
        header_html = self._build_header()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        footer_html = f'''
        <div class="footer">
            <p>Generated by <strong>fmridenoiser v{__version__}</strong></p>
            <p>{timestamp}</p>
            <p><a href="https://github.com/ln2t/fmridenoiser" target="_blank">GitHub</a></p>
        </div>'''

        title_label = self.subject_id if self.subject_id.startswith('sub-') else f"sub-{self.subject_id}"

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>fmridenoiser Report - {title_label}</title>
    {REPORT_CSS}
</head>
<body>
    {nav_html}
    <div class="container">
        {header_html}
        {toc_html}
        {sections_html}
        {footer_html}
    </div>
</body>
</html>'''

        # Build report filename
        report_name_parts = [self.subject_id]
        if self.desc:
            report_name_parts.append(f"desc-{self.desc}")
        if self.label:
            report_name_parts.append(f"label-{self.label}")
        report_name_parts.append("denoising-report.html")

        report_filename = "_".join(report_name_parts)
        report_path = output_path / report_filename

        with report_path.open('w') as f:
            f.write(html)

        self._logger.info(f"Report saved to {report_path}")
        return report_path

    def _build_nav_bar(self) -> str:
        links = "".join(
            f'<a href="#{item_id}">{title}</a>'
            for item_id, title in self.toc_items
        )
        return f'''
        <div class="nav-bar">
            <div class="nav-content">
                <span class="nav-brand">fmridenoiser v{__version__}</span>
                <div class="nav-links">{links}</div>
            </div>
        </div>'''

    def _build_toc(self) -> str:
        items = ""
        for i, (item_id, title) in enumerate(self.toc_items, 1):
            items += f'<li><a href="#{item_id}"><span class="toc-number">{i}</span> {title}</a></li>\n'
        return f'''
        <div class="toc">
            <h2>Contents</h2>
            <ul class="toc-list">{items}</ul>
        </div>'''

    def _build_header(self) -> str:
        title = self.subject_id if self.subject_id.startswith('sub-') else f"sub-{self.subject_id}"
        strategy = self.config.denoising_strategy or "custom"

        meta = f'''
        <div class="meta-info">
            <span>Strategy: <strong>{strategy}</strong></span>
            <span>High-pass: <strong>{self.config.high_pass} Hz</strong></span>
            <span>Low-pass: <strong>{self.config.low_pass} Hz</strong></span>
            <span>Confounds: <strong>{len(self.confounds_used)}</strong></span>
        </div>'''

        return f'''
        <div class="header">
            <h1>fMRI Denoising Report</h1>
            <div class="subtitle">{title}</div>
            {meta}
        </div>'''

    def _build_overview_section(self) -> str:
        self.toc_items.append(("overview", "Overview"))

        strategy = self.config.denoising_strategy or "custom"
        censoring_status = "Enabled" if self.censoring_summary and self.censoring_summary.get('enabled') else "Disabled"

        return f'''
        <div class="section" id="overview">
            <h2>1. Overview</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{strategy}</div>
                    <div class="metric-label">Denoising Strategy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(self.confounds_used)}</div>
                    <div class="metric-label">Confounds Regressed</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{self.config.high_pass}-{self.config.low_pass} Hz</div>
                    <div class="metric-label">Bandpass Filter</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{censoring_status}</div>
                    <div class="metric-label">Motion Censoring</div>
                </div>
            </div>
        </div>'''

    def _build_parameters_section(self) -> str:
        self.toc_items.append(("parameters", "Parameters"))

        confounds_str = ", ".join(f"<code>{c}</code>" for c in self.confounds_used[:20])
        if len(self.confounds_used) > 20:
            confounds_str += f" ... and {len(self.confounds_used) - 20} more"

        rows = f'''
            <tr><td>Denoising Strategy</td><td><code>{self.config.denoising_strategy or "custom"}</code></td></tr>
            <tr><td>High-pass Filter</td><td>{self.config.high_pass} Hz</td></tr>
            <tr><td>Low-pass Filter</td><td>{self.config.low_pass} Hz</td></tr>
            <tr><td>Standardization</td><td>Z-score (mean=0, std=1)</td></tr>
            <tr><td>Detrending</td><td>Linear</td></tr>
            <tr><td>Confounds ({len(self.confounds_used)})</td><td>{confounds_str}</td></tr>'''

        return f'''
        <div class="section" id="parameters">
            <h2>2. Denoising Parameters</h2>
            <table class="param-table">
                <tr><th>Parameter</th><th>Value</th></tr>
                {rows}
            </table>
        </div>'''

    def _build_resampling_section(self) -> str:
        self.toc_items.append(("resampling", "Resampling"))

        if not self.resampling_info or not self.resampling_info.get('resampled'):
            return '''
            <div class="section" id="resampling">
                <h2>3. Resampling</h2>
                <div class="alert alert-success">
                    <span>&#10003;</span>
                    <span>All images have consistent geometry - no resampling needed.</span>
                </div>
            </div>'''

        ri = self.resampling_info
        orig_shape = ri.get('original_shape', ['?', '?', '?'])
        orig_voxel = ri.get('original_voxel_size', ['?', '?', '?'])
        ref_shape = ri.get('reference_shape', ['?', '?', '?'])
        ref_voxel = ri.get('reference_voxel_size', ['?', '?', '?'])

        return f'''
        <div class="section" id="resampling">
            <h2>3. Resampling</h2>
            <div class="alert alert-info">
                <span>&#8505;</span>
                <span>Images were resampled to match reference geometry.</span>
            </div>
            <table class="param-table">
                <tr><th>Property</th><th>Original</th><th>Reference</th></tr>
                <tr><td>Shape (voxels)</td><td>{orig_shape}</td><td>{ref_shape}</td></tr>
                <tr><td>Voxel size (mm)</td><td>{[f"{v:.2f}" for v in orig_voxel]}</td><td>{[f"{v:.2f}" for v in ref_voxel]}</td></tr>
            </table>
        </div>'''

    def _build_confounds_section(self) -> str:
        self.toc_items.append(("confounds", "Confounds"))

        html = '''
        <div class="section" id="confounds">
            <h2>4. Confound Regression</h2>'''

        if self.confounds_df is not None and self.confounds_used:
            try:
                fig = self._plot_confounds_timeseries()
                if fig is not None:
                    img_data = self._figure_to_base64(fig)
                    plt.close(fig)
                    html += f'''
                    <h3>Confound Time Series</h3>
                    <div class="figure-container">
                        <div class="figure-wrapper">
                            <img src="data:image/png;base64,{img_data}" alt="Confound time series">
                        </div>
                        <div class="figure-caption">Selected confound regressors over time (z-scored for display).</div>
                    </div>'''
            except Exception as e:
                self._logger.warning(f"Could not generate confound time series plot: {e}")

            try:
                fig = self._plot_confounds_correlation()
                if fig is not None:
                    img_data = self._figure_to_base64(fig)
                    plt.close(fig)
                    html += f'''
                    <h3>Confound Correlation Matrix</h3>
                    <div class="figure-container">
                        <div class="figure-wrapper">
                            <img src="data:image/png;base64,{img_data}" alt="Confound correlation matrix">
                        </div>
                        <div class="figure-caption">Pearson correlation between selected confound regressors.
                        High correlations between confounds may indicate collinearity.</div>
                    </div>'''
            except Exception as e:
                self._logger.warning(f"Could not generate confound correlation plot: {e}")

        html += '</div>'
        return html

    def _plot_confounds_timeseries(self) -> Optional[plt.Figure]:
        """Plot confound time series."""
        if self.confounds_df is None or not self.confounds_used:
            return None

        # Select subset of confounds for plotting
        plot_confounds = self.confounds_used[:8]
        available = [c for c in plot_confounds if c in self.confounds_df.columns]

        if not available:
            return None

        fig, axes = plt.subplots(len(available), 1, figsize=(12, 2 * len(available)), sharex=True)
        if len(available) == 1:
            axes = [axes]

        for i, confound in enumerate(available):
            values = self.confounds_df[confound].values
            values = (values - np.nanmean(values)) / (np.nanstd(values) + 1e-10)
            axes[i].plot(values, linewidth=0.8, color='#0891b2')
            axes[i].set_ylabel(confound, fontsize=8)
            axes[i].tick_params(labelsize=7)
            axes[i].axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

        axes[-1].set_xlabel('Volume')
        fig.suptitle('Confound Regressors (z-scored)', fontsize=12, fontweight='bold')
        fig.tight_layout()
        return fig

    def _plot_confounds_correlation(self) -> Optional[plt.Figure]:
        """Plot correlation matrix of selected confounds."""
        if self.confounds_df is None or not self.confounds_used:
            return None

        available = [c for c in self.confounds_used if c in self.confounds_df.columns]
        if len(available) < 2:
            return None

        # Compute correlation matrix
        corr = self.confounds_df[available].corr()

        # Determine figure size based on number of confounds
        n = len(available)
        size = max(6, min(14, n * 0.5 + 2))
        fig, ax = plt.subplots(figsize=(size, size))

        # Plot heatmap
        im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Pearson r', fontsize=10)

        # Configure ticks
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        # Shorten long confound names for readability
        short_labels = [c if len(c) <= 20 else c[:17] + '...' for c in available]
        ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(short_labels, fontsize=7)

        # Add correlation values in cells if not too many confounds
        if n <= 20:
            for i in range(n):
                for j in range(n):
                    val = corr.values[i, j]
                    color = 'white' if abs(val) > 0.6 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=max(5, 8 - n // 5), color=color)

        ax.set_title('Confound Correlation Matrix', fontsize=12, fontweight='bold', pad=15)
        fig.tight_layout()
        return fig

    def _build_censoring_section(self) -> str:
        self.toc_items.append(("censoring", "Temporal Censoring"))

        if not self.censoring_summary or not self.censoring_summary.get('enabled'):
            return '''
            <div class="section" id="censoring">
                <h2>5. Temporal Censoring</h2>
                <div class="alert alert-info">
                    <span>&#8505;</span>
                    <span>FD-based temporal censoring was not enabled for this analysis.</span>
                </div>
            </div>'''

        cs = self.censoring_summary
        n_original = cs.get('n_original', 0)
        n_retained = cs.get('n_retained', 0)
        n_censored = cs.get('n_censored', 0)
        fraction = cs.get('fraction_retained', 0)
        reason_counts = cs.get('reason_counts', {})

        reasons_html = ""
        for reason, count in reason_counts.items():
            reasons_html += f"<tr><td><code>{reason}</code></td><td>{count}</td></tr>\n"

        alert_class = "alert-success" if fraction >= 0.7 else ("alert-warning" if fraction >= 0.3 else "alert-danger")

        html = f'''
        <div class="section" id="censoring">
            <h2>5. Temporal Censoring (FD-based)</h2>
            <div class="{alert_class}">
                <span>{"&#10003;" if fraction >= 0.5 else "&#9888;"}</span>
                <span>{n_retained}/{n_original} volumes retained ({fraction:.1%})</span>
            </div>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{n_original}</div>
                    <div class="metric-label">Original Volumes</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{n_retained}</div>
                    <div class="metric-label">Retained</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{n_censored}</div>
                    <div class="metric-label">Censored</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{fraction:.1%}</div>
                    <div class="metric-label">Retention Rate</div>
                </div>
            </div>'''

        if reasons_html:
            html += f'''
            <h3>Censoring Reasons</h3>
            <table class="param-table">
                <tr><th>Reason</th><th>Volumes Affected</th></tr>
                {reasons_html}
            </table>'''

        # Plot censoring mask
        try:
            mask = cs.get('mask')
            if mask and self.confounds_df is not None:
                fig = self._plot_censoring_mask(mask)
                if fig is not None:
                    img_data = self._figure_to_base64(fig)
                    plt.close(fig)
                    html += f'''
                    <h3>Censoring Mask</h3>
                    <div class="figure-container">
                        <div class="figure-wrapper">
                            <img src="data:image/png;base64,{img_data}" alt="Censoring mask">
                        </div>
                        <div class="figure-caption">
                            Top: Framewise displacement with threshold. Bottom: Retained (green) and censored (red) volumes.
                        </div>
                    </div>'''
        except Exception as e:
            self._logger.warning(f"Could not generate censoring mask plot: {e}")

        html += '</div>'
        return html

    def _plot_censoring_mask(self, mask: list) -> Optional[plt.Figure]:
        """Plot FD trace and censoring mask."""
        mask_arr = np.array(mask)
        n_volumes = len(mask_arr)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 4), sharex=True,
                                        gridspec_kw={'height_ratios': [3, 1]})

        # FD trace
        if self.confounds_df is not None and 'framewise_displacement' in self.confounds_df.columns:
            fd = self.confounds_df['framewise_displacement'].values
            fd = np.nan_to_num(fd, nan=0.0)
            ax1.plot(fd, linewidth=0.8, color='#0891b2', label='FD')
            if self.config.censoring.motion_censoring.enabled:
                threshold = self.config.censoring.motion_censoring.fd_threshold
                ax1.axhline(y=threshold, color='#ef4444', linewidth=1, linestyle='--',
                           label=f'Threshold ({threshold} cm)')
            ax1.set_ylabel('FD (cm)')
            ax1.legend(fontsize=8)
        else:
            ax1.text(0.5, 0.5, 'FD data not available', transform=ax1.transAxes, ha='center')

        # Censoring mask
        for i in range(n_volumes):
            color = '#10b981' if mask_arr[i] else '#ef4444'
            ax2.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.7)
        ax2.set_xlim(-0.5, n_volumes - 0.5)
        ax2.set_yticks([])
        ax2.set_xlabel('Volume')
        ax2.set_ylabel('Status')

        fig.suptitle('Temporal Censoring', fontsize=12, fontweight='bold')
        fig.tight_layout()
        return fig

    def _build_qa_section(self) -> str:
        self.toc_items.append(("qa", "Quality Assessment"))

        html = '''
        <div class="section" id="qa">
            <h2>6. Quality Assessment</h2>'''

        if self.denoising_histogram_data is not None:
            try:
                fig = self._plot_denoising_histogram()
                if fig is not None:
                    img_data = self._figure_to_base64(fig)
                    plt.close(fig)
                    html += f'''
                    <h3>Denoising Effect on Voxel Intensity Distribution</h3>
                    <div class="figure-container">
                        <div class="figure-wrapper">
                            <img src="data:image/png;base64,{img_data}" alt="Denoising histogram">
                        </div>
                        <div class="figure-caption">
                            Distribution of voxel values before (z-scored) and after denoising.
                            A narrower distribution after denoising indicates effective noise removal.
                        </div>
                    </div>'''

                    # Add statistics
                    orig = self.denoising_histogram_data.get('original_stats', {})
                    den = self.denoising_histogram_data.get('denoised_stats', {})
                    html += f'''
                    <table class="param-table">
                        <tr><th>Statistic</th><th>Before (z-scored)</th><th>After Denoising</th></tr>
                        <tr><td>Mean</td><td>{orig.get("mean", "N/A"):.4f}</td><td>{den.get("mean", "N/A"):.4f}</td></tr>
                        <tr><td>Std</td><td>{orig.get("std", "N/A"):.4f}</td><td>{den.get("std", "N/A"):.4f}</td></tr>
                    </table>'''
            except Exception as e:
                self._logger.warning(f"Could not generate denoising histogram: {e}")

        html += '</div>'
        return html

    def _plot_denoising_histogram(self) -> Optional[plt.Figure]:
        """Plot before/after denoising histogram."""
        if self.denoising_histogram_data is None:
            return None

        orig = self.denoising_histogram_data['original_data']
        den = self.denoising_histogram_data['denoised_data']

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(orig, bins=100, alpha=0.5, label='Before (z-scored)', color='#94a3b8', density=True)
        ax.hist(den, bins=100, alpha=0.5, label='After denoising', color='#0891b2', density=True)
        ax.set_xlabel('Voxel Value')
        ax.set_ylabel('Density')
        ax.set_title('Denoising Effect on Voxel Intensity Distribution')
        ax.legend()
        ax.set_xlim(-5, 5)
        fig.tight_layout()
        return fig

    def _build_command_section(self) -> str:
        self.toc_items.append(("command", "Reproducibility"))

        cmd = self.command_line or "fmridenoiser [args not recorded]"

        return f'''
        <div class="section" id="command">
            <h2>7. Reproducibility</h2>
            <h3>Command Line</h3>
            <div class="code-block">{cmd}</div>
            <div class="info-box">
                <h4>Configuration Backup</h4>
                <p>A copy of the configuration used for this analysis is saved in
                <code>{self.output_dir}/config/backups/</code>.</p>
            </div>
        </div>'''

    def _build_references_section(self) -> str:
        self.toc_items.append(("references", "References"))

        return '''
        <div class="section" id="references">
            <h2>8. References</h2>
            <div class="reference-item">
                <strong>fMRIPrep:</strong> Esteban et al. (2019). fMRIPrep: a robust preprocessing pipeline
                for functional MRI. <em>Nature Methods</em>, 16, 111-116.
                <a href="https://doi.org/10.1038/s41592-018-0235-4" target="_blank">doi:10.1038/s41592-018-0235-4</a>
            </div>
            <div class="reference-item">
                <strong>Nilearn:</strong> Abraham et al. (2014). Machine learning for neuroimaging with
                scikit-learn. <em>Frontiers in Neuroinformatics</em>, 8, 14.
                <a href="https://doi.org/10.3389/fninf.2014.00014" target="_blank">doi:10.3389/fninf.2014.00014</a>
            </div>
            <div class="reference-item">
                <strong>Denoising strategies:</strong> Wang et al. (2024). Benchmarking fMRI denoising
                strategies for functional connectomics.
            </div>
            <div class="reference-item">
                <strong>Motion scrubbing:</strong> Power et al. (2012). Spurious but systematic correlations
                in functional connectivity MRI networks arise from subject motion.
                <em>NeuroImage</em>, 59, 2142-2154.
                <a href="https://doi.org/10.1016/j.neuroimage.2011.10.018" target="_blank">doi:10.1016/j.neuroimage.2011.10.018</a>
            </div>
        </div>'''
