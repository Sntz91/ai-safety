import json
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

st.set_page_config(page_title="Data Registry", layout="wide")
st.title("Overview of used Datasets")

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
    "",
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
            Download SVG
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
                svgElement.style.maxWidth = 'auto';
                svgElement.style.height = '300';
            }}
            const blob = new Blob([svg], {{ type: 'image/svg+xml' }});
            const url = URL.createObjectURL(blob);
            document.getElementById('download-link').href = url;
        }});
    </script>
"""
components.html(html_code, height=330, scrolling=False)

st.markdown("---")
st.title("Dataset Split Statistics")

# Create tabs for dataset info and statistics
datasets = sorted(stats.keys())
tab_names = ["Info"] + [d.upper() for d in datasets]
tabs = st.tabs(tab_names)

with tabs[0]:
    st.header("Dataset Information & Literature Sources")
    st.markdown("""
    * **RSNA-IHD**: [Paper](https://pubs.rsna.org/doi/full/10.1148/ryai.2020190211), [Dataset](https://registry.opendata.aws/rsna-intracranial-hemorrhage-detection/)
        * 27,861 CT scans with slice annotations
        * Three institutions
        * Labeled by 60 radiologists (test dataset), 1 radiologist (training data)
        * Labels: Any, Epidural, Subdural, Intraparenchymal, Intraventricular, Subarachnoid
    * **SinoCT**: [Paper](https://pubs.rsna.org/doi/abs/10.1148/ryai.2021200229), [Dataset](https://aimi.stanford.edu/datasets/sinoct)
        * 9000 scans with scan annotations
        * Labels: Normal / Abnormal
        * Substantial overlap with RSNA-IHD
    * **CQ500/BHX**: [Paper](https://arxiv.org/abs/1803.05854), [Dataset](https://www.kaggle.com/datasets/crawford/qureai-headct), [BHX Extension](https://physionet.org/content/bhx-brain-bounding-box/1.1/):
        * 491 CT scans of 6 different scanners
        * Labeled by 3 radiologists
        * Scan-level annotations with slice-level bounding boxes through BHX extension
        * Labels: Any, Epidural, Subdural, Intraparenchymal, Intraventricular, Subarachnoid
        * Literature mostly uses CQ500 for external validation
    * **PHE-SICH-CT-IDS**: [Paper](https://arxiv.org/abs/2308.10521), [Dataset](https://figshare.com/articles/dataset/PHE-SICH-CT-IDS/23957937?file=42152730)
        * 120 CT scans of 1 scanner with segmentation
        * Labeled by 3 Radiologists
        * Label: ICH? (binary)
    * **PhysioNet-CT-ICH**: [Paper](https://www.mdpi.com/2306-5729/5/1/14), [Dataset](https://physionet.org/content/ct-ich/1.3.1/)
        * 82 CT scans of 1 scanner with segmentation
        * Labeled by 2 radiologists
        * Label: Intraventricular, Intraparenchymal, subarachnoid, epidural, subdural, fracture
    """)

for tab, dataset_name in zip(tabs[1:], datasets):
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
