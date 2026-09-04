import numpy as np

from ai_safety.models.diagnostic.aggregation import aggregate_to_scan_level
from ai_safety.models.monitor.aggregation import aggregate_dual_pooling_to_scan_level


def compute_flow_data(df, subtype, level, d_thresh, m_thresh):
    if level == "slice_level":
        diag_gt = df[f"label_{subtype}_diag"].values
        diag_prob = df[f"prob_{subtype}_diag"].values
        mon_prob = df[f"prob_{subtype}_mon"].values
    else:
        series_ids = df["series_id"].values
        orig_diag_gts = df[f"label_{subtype}_diag"].values
        orig_diag_probs = df[f"prob_{subtype}_diag"].values

        _, diag_prob, diag_gt = aggregate_to_scan_level(series_ids, orig_diag_probs, orig_diag_gts, k=3)
        _, mon_prob = aggregate_dual_pooling_to_scan_level(
            series_ids, diag_probs=orig_diag_probs, mon_probs=df[f"prob_{subtype}_mon"].values, diag_threshold=d_thresh, k=3
        )

    diag_pred = (diag_prob >= d_thresh).astype(int)
    mon_pred = (mon_prob >= m_thresh).astype(int)

    is_pos = diag_gt == 1
    is_neg = ~is_pos
    diag_correct = diag_pred == diag_gt
    mon_flagged = mon_pred == 1

    res = {
        "total": len(diag_gt),
        "dis_pos": {
            "total": int(is_pos.sum()),
            "diag_tp": {
                "total": int((is_pos & diag_correct).sum()),
                "mon_fp": int((is_pos & diag_correct & mon_flagged).sum()),
                "mon_tn": int((is_pos & diag_correct & ~mon_flagged).sum()),
            },
            "diag_fn": {
                "total": int((is_pos & ~diag_correct).sum()),
                "mon_tp": int((is_pos & ~diag_correct & mon_flagged).sum()),
                "mon_fn": int((is_pos & ~diag_correct & ~mon_flagged).sum()),
            },
        },
        "dis_neg": {
            "total": int(is_neg.sum()),
            "diag_tn": {
                "total": int((is_neg & diag_correct).sum()),
                "mon_fp": int((is_neg & diag_correct & mon_flagged).sum()),
                "mon_tn": int((is_neg & diag_correct & ~mon_flagged).sum()),
            },
            "diag_fp": {
                "total": int((is_neg & ~diag_correct).sum()),
                "mon_tp": int((is_neg & ~diag_correct & mon_flagged).sum()),
                "mon_fn": int((is_neg & ~diag_correct & ~mon_flagged).sum()),
            },
        },
    }

    return res


def render_mermaid_html(mermaid_code: str, container_id: str, download_filename: str) -> str:
    svg_id = f"mermaid-svg-{container_id}"
    return f"""
        <div style="text-align: right; margin-bottom: 8px;">
            <a id="download-{container_id}" href="#" download="{download_filename}" style="color: #cad3f5; text-decoration: none; font-family: sans-serif; background-color: #363a4f; padding: 6px 12px; border-radius: 6px; font-size: 13px;">
                Download SVG
            </a>
        </div>
        <div id="{container_id}" style="display: flex; justify-content: center; width: 100%;">
        </div>
        
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ 
                startOnLoad: false, 
                theme: 'dark',
                securityLevel: 'loose',
                htmlLabels: true
            }});
            
            const graphDefinition = `{mermaid_code}`;
            
            const render = () => {{
                mermaid.render('{svg_id}', graphDefinition).then((result) => {{
                    let svg = result.svg.replace(/<br>/g, '<br/>');
                    const container = document.getElementById('{container_id}');
                    if (container) {{
                        container.innerHTML = svg;
                    }}
                    const svgElement = document.getElementById('{svg_id}');
                    if (svgElement) {{
                        svgElement.style.maxWidth = '100%';
                        svgElement.style.height = 'auto';
                    }}
                    const blob = new Blob([svg], {{ type: 'image/svg+xml' }});
                    const link = document.getElementById('download-{container_id}');
                    if (link) {{
                        link.href = URL.createObjectURL(blob);
                    }}
                }}).catch((err) => {{
                    console.error(err);
                }});
            }};

            const observer = new IntersectionObserver((entries) => {{
                if (entries[0].isIntersecting) {{
                    render();
                    observer.disconnect();
                }}
            }});
            observer.observe(document.body);
        </script>
    """
