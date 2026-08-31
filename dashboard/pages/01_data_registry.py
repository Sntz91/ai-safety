import json
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

st.set_page_config(page_title="Data Registry", layout="wide")
st.title("Data Registry Overview")

st.markdown("""
This dynamically generates an interactive Venn diagram visualizing our exact dataset distributions and their slice overlaps.
""")

try:
    with open("splits/stats.json", "r") as f:
        stats = json.load(f)
except FileNotFoundError:
    st.error("Could not find splits/stats.json")
    st.stop()

# Aggregate totals for Venn diagram
dataset_totals = {}
rsna_sinoct_overlap = 0

if "rsna" in stats:
    rsna_total = stats["rsna"].get("all.parquet", {}).get("slices", 0)
    rsna_sinoct_overlap = stats["rsna"].get("match-sinoct-all.parquet", {}).get("slices", 0)
    dataset_totals["RSNA"] = rsna_total

if "sinoct" in stats:
    sinoct_total = stats["sinoct"].get("all.parquet", {}).get("slices", 0)
    dataset_totals["SINOCT"] = sinoct_total

if "bhx" in stats:
    bhx_total = stats["bhx"].get("all.parquet", {}).get("slices", 0)
    dataset_totals["BHX"] = bhx_total

smaller_circle = min(dataset_totals.get('RSNA', 0), dataset_totals.get('SINOCT', 0))
visual_overlap = min(rsna_sinoct_overlap, int(smaller_circle * 0.70)) if smaller_circle > 0 else rsna_sinoct_overlap

lines = [
    "venn-beta",
    "  title \"Dataset Overlaps (Slice Counts)\"",
    f"  set rsna[\"RSNA\"]:{dataset_totals.get('RSNA', 0)}",
    f"    text rsna_t[\"({dataset_totals.get('RSNA', 0)//1000}k)\"]",
    f"  set sinoct[\"SINOCT\"]:{dataset_totals.get('SINOCT', 0)}",
    f"    text sinoct_t[\"({dataset_totals.get('SINOCT', 0)//1000}k)\"]",
    f"  set bhx[\"BHX\"]:{dataset_totals.get('BHX', 0)}",
    f"    text bhx_t[\"({dataset_totals.get('BHX', 0)//1000}k)\"]",
    f"  union rsna,sinoct[\"Overlap\"]:{visual_overlap}",
    f"    text overlap_t[\"({rsna_sinoct_overlap//1000}k slices)\"]",
    "  style rsna fill:#a6da95,fill-opacity:0.3",
    "  style sinoct fill:#8aadf4,fill-opacity:0.3",
    "  style bhx fill:#ed8796,fill-opacity:0.3"
]
mermaid_code = "\n".join(lines)
html_code = f"""
    <div style="text-align: right; margin-bottom: 10px;">
        <a id="download-link" href="#" download="registry_venn.svg" style="color: #cad3f5; text-decoration: none; font-family: sans-serif; background-color: #363a4f; padding: 8px 12px; border-radius: 6px; font-size: 14px;">
            ⬇️ Download SVG
        </a>
    </div>
    <div id="graph-container" style="display: flex; justify-content: center; width: 100%;">
    </div>
    
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ 
            startOnLoad: false, 
            theme: 'dark',
            securityLevel: 'loose'
        }});
        
        const graphDefinition = `{mermaid_code}`;
        
        mermaid.render('mermaid-svg', graphDefinition).then((result) => {{
            let svg = result.svg;
            svg = svg.replace(/<br>/g, '<br/>');
            document.getElementById('graph-container').innerHTML = svg;
            const svgElement = document.getElementById('mermaid-svg');
            if(svgElement) {{
                svgElement.style.maxWidth = '100%';
                svgElement.style.height = 'auto';
            }}
            const blob = new Blob([svg], {{ type: 'image/svg+xml' }});
            const url = URL.createObjectURL(blob);
            document.getElementById('download-link').href = url;
        }});
    </script>
"""
components.html(html_code, height=600, scrolling=True)

st.markdown("---")
st.title("Dataset Split Statistics")

# Create a tab for each dataset
datasets = sorted(stats.keys())
tabs = st.tabs(datasets)

for tab, dataset_name in zip(tabs, datasets):
    with tab:
        st.header(dataset_name.upper())
        splits = stats[dataset_name]
        
        summary_data = []
        for split_name, split_data in splits.items():
            summary_data.append({
                "Split": split_name,
                "Patients": split_data.get("patients", "N/A"),
                "Series": split_data.get("series", "N/A"),
                "Slices": split_data["slices"],
            })
            
        summary_df = pd.DataFrame(summary_data).set_index("Split")
        st.subheader("Overview")
        st.dataframe(summary_df, use_container_width=True)
        
        st.subheader("Label Distributions")
        for split_name, split_data in splits.items():
            labels = split_data.get("labels", {})
            if not labels:
                continue
                
            label_data = []
            for label_name, metrics in labels.items():
                label_data.append({
                    "Label": label_name,
                    "Count": metrics["count"],
                    "Prevalence (%)": round(metrics["prevalence"] * 100, 2)
                })
                
            if label_data:
                st.write(f"**{split_name}**")
                label_df = pd.DataFrame(label_data).set_index("Label")
                st.dataframe(label_df, use_container_width=True)
