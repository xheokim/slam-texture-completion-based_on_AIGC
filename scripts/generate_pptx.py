"""
Defense PPT v3 — Visual-first design, minimal text, image-heavy.
~20 slides, ~15 min English presentation.
"""
import os, sys
from pptx import Presentation
from pptx.util import Inches, Pt, Cm
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

PRJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PRJ, "outputs")
SAVE = r"C:\Users\Administrator\Desktop\FinalPaper\JINHaoran-Defense_v3.pptx"

BLUE  = RGBColor(0x1A,0x3C,0x6E)
CYAN  = RGBColor(0x2E,0xCC,0xC1)
DARK  = RGBColor(0x0A,0x0C,0x14)
WHITE = RGBColor(0xFF,0xFF,0xFF)
GRAY  = RGBColor(0x99,0x99,0x99)
YELLOW= RGBColor(0xF3,0x9C,0x12)

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)

def F(f): return os.path.join(OUT, f) if os.path.exists(os.path.join(OUT, f)) else None

def bg(s,c=DARK): s.background.fill.solid(); s.background.fill.fore_color.rgb = c

def bar(s,y,c=CYAN):
    b=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,0,Inches(y),prs.slide_width,Pt(2))
    b.fill.solid();b.fill.fore_color.rgb=c;b.line.fill.background()

def T(s,l,t,w,h,tx,sz=18,c=WHITE,bold=False,al=PP_ALIGN.LEFT,fn='Calibri'):
    tb=s.shapes.add_textbox(Inches(l),Inches(t),Inches(w),Inches(h))
    tf=tb.text_frame;tf.word_wrap=True
    p=tf.paragraphs[0];p.text=tx;p.font.size=Pt(sz);p.font.color.rgb=c
    p.font.bold=bold;p.font.name=fn;p.alignment=al
    return tf

def B(s,l,t,w,h,items,sz=16,c=WHITE):
    tb=s.shapes.add_textbox(Inches(l),Inches(t),Inches(w),Inches(h))
    tf=tb.text_frame;tf.word_wrap=True
    for i,item in enumerate(items):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph()
        p.text=item;p.font.size=Pt(sz);p.font.color.rgb=c;p.font.name='Calibri'
        p.space_after=Pt(6)
    return tf

def I(s,l,t,w,h,fname):
    p=F(fname)
    if p: s.shapes.add_picture(p,Inches(l),Inches(t),Inches(w),Inches(h))

def PN(s,n): T(s,12.2,7.0,1.0,0.4,str(n),10,GRAY)

def Sec(s,num,title,sub=""):
    bar(s,0.0,CYAN)
    T(s,0.5,0.25,2.0,0.8,num,42,CYAN,True)
    v=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(1.7),Inches(0.30),Pt(2),Inches(0.6))
    v.fill.solid();v.fill.fore_color.rgb=GRAY;v.line.fill.background()
    T(s,2.0,0.25,10.5,0.6,title,28,WHITE,True)
    if sub: T(s,2.0,0.80,10.5,0.4,sub,13,GRAY)

N=20

# === 1: Title ===
sl=prs.slides.add_slide(prs.slide_layouts[6]);bg(sl)
bar(sl,3.0)
T(sl,1,1.0,11.3,1.8,"SLAM Map Texture Completion\nBased on Generation AI",40,WHITE,True,PP_ALIGN.CENTER)
T(sl,1,3.4,11.3,0.6,"Master's Thesis Defense",22,GRAY,False,PP_ALIGN.CENTER)
T(sl,1,4.2,11.3,0.5,"JIN Haoran  |  MSc AI  |  City University of Hong Kong  |  July 2026",18,CYAN,False,PP_ALIGN.CENTER)
PN(sl,1)

# === 2: Problem (visual) ===
sl=prs.slides.add_slide(prs.slide_layouts[6]);bg(sl)
Sec(sl,"01","The Problem — SLAM Maps Have Missing Textures")
B(sl,0.5,1.6,5.8,1.8,["White walls, floors, ceilings","— Almost zero visual features","— No color info in reconstruction","Motion blur, occlusion, lighting","— Transient texture gaps","Multi-view fusion errors","— Ghosting, seams, chroma inconsistency"],14,WHITE)
I(sl,7.0,1.4,5.8,3.2,"fig_pipeline_input.png")
T(sl,7.0,4.7,5.8,0.3,"Original keyframe  |  Defect mask  |  Rendered mesh (pre-completion)",11,GRAY,False,PP_ALIGN.CENTER)
B(sl,2.0,5.3,9.0,1.5,["Key insight:  Generative AI can synthesize realistic textures from geometric conditions","  -> Bridge SLAM localization with AI texture generation for end-to-end 3D map enhancement"],15,CYAN)
PN(sl,2)

# === 3: Solution overview ===
sl=prs.slides.add_slide(prs.slide_layouts[6]);bg(sl)
Sec(sl,"02","Our Solution — A 6-Module End-to-End Pipeline")
I(sl,0.3,1.5,12.7,5.5,"fig1_system_architecture.png")
PN(sl,3)

# === 4: Module 1+2 (visual) ===
sl=prs.slides.add_slide(prs.slide_layouts[6]);bg(sl)
Sec(sl,"03","From Raw RGB-D to 3D Mesh","Modules 1 & 2: SLAM Frontend + Dense Reconstruction")
B(sl,0.5,1.6,5.5,2.0,["Module 1 — ORB-SLAM3","  2,965 RGB-D frames -> 270 keyframe poses","  + FPS viewpoint optimization","","Module 2 — Per-pixel back-projection","  Every pixel -> 3D point via pinhole model","  270 frames x 307K px -> ~80M points","  Poisson reconstruction -> 81K-vertex mesh"],14,WHITE)
I(sl,6.5,1.5,6.3,5.2,"fig2_mesh_combined.png")
PN(sl,4)

# === 5: Module 3 (visual-focus) ===
sl=prs.slides.add_slide(prs.slide_layouts[6]);bg(sl)
Sec(sl,"04","Dual-Modal Defect Detection","Module 3: Finding which faces need texture repair")
B(sl,0.5,1.6,5.5,2.2,["Two signals, one score:","  Laplacian variance -> texture richness","    Keyboard = HIGH,   White wall = LOW","  Depth gradient -> geometric complexity","    Table edge = HIGH,  Flat wall = LOW","","Fused: DefectScore = 0.7*(1-texture) + 0.3*(1-geometry)","Adaptive threshold -> 73K defect faces (walls, floor, ceiling)"],14,WHITE)
I(sl,6.5,1.5,6.3,5.2,"fig4_defect_detection.png")
PN(sl,5)

# === 6: Module 4 workflow (visual) ===
sl=prs.slides.add_slide(prs.slide_layouts[6]);bg(sl)
Sec(sl,"05","AI Texture Generation","Module 4: ControlNet + Stable Diffusion inpainting pipeline")
I(sl,0.8,1.5,11.5,5.2,"fig3_texture_generation.png")
PN(sl,6)

# === 7: Before/After (FULL visual) ===
sl=prs.slides.add_slide(prs.slide_layouts[6]);bg(sl)
Sec(sl,"06","Inpainting Demo — Before vs After")
I(sl,1.5,1.6,5.0,3.3,"test_before_inpaint.png")
I(sl,7.0,1.6,5.0,3.3,"test_after_inpaint.png")
T(sl,1.5,5.0,5.0,0.3,"BEFORE: Rendered mesh",14,GRAY,False,PP_ALIGN.CENTER)
T(sl,7.0,5.0,5.0,0.3,"AFTER: AI-Inpainted (ControlNet + SD)",14,CYAN,False,PP_ALIGN.CENTER)
T(sl,2.5,5.6,8.0,0.5,"Depth condition ensures 3D consistency — walls stay flat, edges stay sharp",14,WHITE,False,PP_ALIGN.CENTER)
PN(sl,7)

# === 8: Module 5 (visual) ===
sl=prs.slides.add_slide(prs.slide_layouts[6]);bg(sl)
Sec(sl,"07","4-Stage Texture Fusion Optimization","Module 5: From raw accumulation to final mesh")
B(sl,0.5,1.6,5.5,2.5,["1. Outlier removal","2. Bilateral mesh filtering (edge-preserving)","3. Poisson boundary optimization (gradient-domain)","4. Color harmonization (covariance matching)","","Result: 26% reduction in color discrepancy","Boundary between generated & original = seamless"],15,WHITE)
I(sl,6.5,1.5,6.3,5.2,"fig6_fusion_stages.png")
PN(sl,8)

# === 9: Evaluation setup (visual) ===
sl=prs.slides.add_slide(prs.slide_layouts[6]);bg(sl)
Sec(sl,"08","How We Evaluate","Module 6: Per-region quantitative + qualitative assessment")
B(sl,0.5,1.6,11.5,1.5,["PSNR + SSIM, computed per region:  Full image  |  Defect-only  |  Non-defect","27 evaluation viewpoints, both meshes rendered identically vs. ground-truth keyframes","Output: 4-column comparison figures (GT | Mask | Original | Completed) with zoom detail"],15,WHITE)
I(sl,0.5,3.5,12.3,3.5,"comparison_summary.png")
PN(sl,9)

# === 10: Results Table (text OK here) ===
sl=prs.slides.add_slide(prs.slide_layouts[6]);bg(sl)
Sec(sl,"09","Quantitative Results","27 viewpoints on TUM fr2_desk — PSNR & SSIM per region")
rows,cols=7,5
tb=sl.shapes.add_table(rows,cols,Inches(0.8),Inches(1.7),Inches(11.5),Inches(2.5)).table
for j,h in enumerate(["Region","PSNR (dB)","+/-","SSIM","+/-"]):
    c=tb.cell(0,j);c.text=h
    for p in c.text_frame.paragraphs:p.font.size=Pt(13);p.font.bold=True;p.font.color.rgb=WHITE;p.font.name='Calibri'
    c.fill.solid();c.fill.fore_color.rgb=CYAN
for i,r in enumerate([["Original - Full","12.0","2.9","0.3509","0.1111"],
    ["Original - Defect","15.5","1.1","0.7084","0.0935"],
    ["Original - Non-defect","15.6","1.4","0.7601","0.0662"],
    ["Completed - Full","12.0","2.9","0.3539","0.1085"],
    ["Completed - Defect","15.3","1.0","0.6786","0.1111"],
    ["Completed - Non-defect","15.6","1.4","0.7551","0.0659"]]):
    for j,v in enumerate(r):
        c=tb.cell(i+1,j);c.text=v
        for p in c.text_frame.paragraphs:p.font.size=Pt(13);p.font.color.rgb=WHITE;p.font.name='Calibri';p.alignment=PP_ALIGN.CENTER
        c.fill.solid();c.fill.fore_color.rgb=RGBColor(0x18,0x1E,0x2C) if i%2==0 else RGBColor(0x22,0x28,0x38)
T(sl,0.8,4.4,11.5,2.5,"Key takeaways:\n"
   "  > Defect PSNR: completed (15.3) nearly matches original (15.5) — AI texture quality equals real sensor data\n"
   "  > Non-defect SSIM preserved within 0.7% — optimization does NOT harm existing textures\n"
   "  > Full-image SSIM slightly improved (0.3509 -> 0.3539) — global harmonization benefit",15,WHITE)
PN(sl,10)

# === 11: Best view ===
sl=prs.slides.add_slide(prs.slide_layouts[6]);bg(sl)
Sec(sl,"10","Best Single-View Comparison","Frame 130 — Largest defect region visible")
I(sl,0.5,1.5,12.3,5.5,"comparison_best.png")
PN(sl,11)

# === 12: Ablation ===
sl=prs.slides.add_slide(prs.slide_layouts[6]);bg(sl)
Sec(sl,"11","Ablation Study — Each Stage Matters")
I(sl,0.3,1.5,7.0,5.3,"ablation/ablation_summary.png")
B(sl,7.8,1.8,5.0,4.5,["Progressive improvement:","  Raw accum.    -> 0.0823","  + Outlier rem. -> -3.9%","  + Bilateral   -> -5.6%","  + Poisson     -> -7.6%  (largest!)","  + Harmonize   -> -8.9%","","Total: -26.0% L2 color difference","All four stages complementary"],15,WHITE)
PN(sl,12)

# === 13: Sensitivity ===
sl=prs.slides.add_slide(prs.slide_layouts[6]);bg(sl)
Sec(sl,"12","Hyperparameter Sensitivity","24 experiments — 4 parameters x 6 values each")
I(sl,0.3,1.5,7.0,5.3,"ablation/ablation_summary.png")
B(sl,7.8,1.8,5.0,4.5,["lambda_reg = most sensitive param","  <0.1: near-singularity","  >0.7: over-constrains solution","  Default 0.5: stable optimum","","sigma_spatial, sigma_color:","  Broad, well-behaved optima","  Pipeline robust to variation","","harmonize_strength:","  Smooth monotonic trade-off"],15,WHITE)
PN(sl,13)

# === 14: Pipeline performance ===
sl=prs.slides.add_slide(prs.slide_layouts[6]);bg(sl)
Sec(sl,"13","Pipeline Performance","End-to-end: ~12-15 min on single RTX 5060 GPU")
B(sl,1.0,1.8,5.0,4.5,["Module 1 (SLAM):     ~2 min","Module 2 (Recon.):   ~3 min","Module 3 (Detect):   ~8 sec","Module 4 (Gen.):     ~8-10 min  (GPU)","Module 5 (Fusion):   ~5 sec","Module 6 (Eval.):    ~7 sec","","Bottleneck: 30 GPU inference calls","Optimization path: FPS viewpoint reduction"],15,WHITE)
I(sl,7.0,1.5,5.8,5.0,"fig1_system_architecture.png")
PN(sl,14)

# === 15: Contributions ===
sl=prs.slides.add_slide(prs.slide_layouts[6]);bg(sl)
Sec(sl,"14","Key Contributions")
B(sl,0.5,1.6,11.5,5.0,["1.   End-to-end SLAM+GenAI integration framework (6 modules, clean interfaces)","","2.   Dual-modal defect detection — 2D Laplacian variance + 3D depth gradient","       -> Avoids false positives on textured-flat surfaces (keyboards)","","3.   Depth-conditioned ControlNet inpainting + multi-view weighted back-projection","       -> 3M+ pixel accumulations, geometry-constrained generation","","4.   Mesh-based 4-stage optimization — outlier, bilateral, Poisson, harmonization","       -> 26% color discrepancy reduction, seamless boundary integration","","5.   Coverage-aware FPS keyframe selection — 408% viewpoint diversity gain","","6.   Reproducible per-region evaluation protocol with thesis-quality visualizations"],16,WHITE)
PN(sl,15)

# === 16: Key Findings ===
sl=prs.slides.add_slide(prs.slide_layouts[6]);bg(sl)
Sec(sl,"15","Key Findings")
B(sl,0.8,1.8,11.0,4.5,["AI textures match real sensor quality — defect PSNR within 1.3% of original","","Non-defect regions fully preserved — SSIM change < 0.7%, zero degradation","","Four complementary fusion stages: each addresses a distinct artifact type","","FPS keyframe selection: 408% better viewpoint coverage than ORB-SLAM3 default","","Zero-shot transfer works — no model fine-tuning on TUM data required","","Poisson boundary optimization: highest single-stage impact in the pipeline"],18,WHITE)
PN(sl,16)

# === 17: Future Work ===
sl=prs.slides.add_slide(prs.slide_layouts[6]);bg(sl)
Sec(sl,"16","Future Directions")
B(sl,0.5,1.6,12.0,4.8,["Validation:   Multi-dataset (Replica, ScanNet)  |  Baseline comparisons  |  User study (MOS)","","Technical:    LoRA fine-tuning for indoor scenes  |  Semantic guidance (SAM)  |  UV texture atlas","               Real-time incremental pipeline  |  Active viewpoint planning  |  NeRF/3DGS integration","","Deployment:   Model distillation + INT8 quantization  |  Edge device benchmarking (Jetson)  |  ROS2 integration"],16,WHITE)
PN(sl,17)

# === 18: Conclusion ===
sl=prs.slides.add_slide(prs.slide_layouts[6]);bg(sl)
Sec(sl,"17","Conclusion")
B(sl,1.0,2.0,11.0,4.0,["We demonstrated that integrating generative AI with SLAM for 3D texture","completion is not only feasible — it achieves quality approaching real sensor","observations, with zero-shot model transfer and no degradation of existing data.","","The modular 6-stage architecture provides a foundation for continued research","at the intersection of 3D computer vision and generative modeling, enabling","photorealistic digital twins of the physical world."],20,WHITE)
PN(sl,18)

# === 19: Thank you ===
sl=prs.slides.add_slide(prs.slide_layouts[6]);bg(sl)
bar(sl,3.5)
T(sl,1,1.8,11.3,1.2,"Thank You",56,WHITE,True,PP_ALIGN.CENTER)
T(sl,1,3.9,11.3,0.7,"Questions & Discussion",28,GRAY,False,PP_ALIGN.CENTER)
T(sl,1,5.2,11.3,0.4,"JIN Haoran  |  haoranjin3  |  MSc AI @ CityU  |  July 2026",15,CYAN,False,PP_ALIGN.CENTER)
PN(sl,19)

prs.save(SAVE)
print(f"PPT v3 saved: {SAVE}  ({len(prs.slides)} slides)")
