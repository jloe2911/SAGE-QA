"""
Generate SAGE-QA 4-slide PowerPoint presentation.
Run: .venv/bin/python scripts/make_slides.py
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt
import pptx.oxml.ns as nsmap
from lxml import etree

# ── Color palette ──────────────────────────────────────────────────────────────
DARK_BLUE   = RGBColor(0x1A, 0x37, 0x6C)   # title / headers
MID_BLUE    = RGBColor(0x2E, 0x6D, 0xA8)   # accent boxes
LIGHT_BLUE  = RGBColor(0xD6, 0xE8, 0xF7)   # box fills
TEAL        = RGBColor(0x1E, 0x7F, 0x8E)   # second accent
ORANGE      = RGBColor(0xE8, 0x6B, 0x2A)   # highlight numbers
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY  = RGBColor(0xF4, 0xF6, 0xF9)
MID_GRAY    = RGBColor(0x6B, 0x7C, 0x93)
DARK_GRAY   = RGBColor(0x2D, 0x3A, 0x4A)
GREEN       = RGBColor(0x27, 0xAE, 0x60)
RED         = RGBColor(0xC0, 0x39, 0x2B)

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)


# ── Helpers ────────────────────────────────────────────────────────────────────

def new_prs():
    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H
    return prs


def blank_slide(prs):
    layout = prs.slide_layouts[6]   # completely blank
    return prs.slides.add_slide(layout)


def add_rect(slide, x, y, w, h, fill_color, line_color=None, line_width=Pt(0)):
    shape = slide.shapes.add_shape(
        pptx.enum.shapes.MSO_SHAPE_TYPE.AUTO_SHAPE if False else 1,  # rectangle = 1
        x, y, w, h
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = line_width
    else:
        shape.line.fill.background()
    return shape


def add_textbox(slide, x, y, w, h, text, font_size=Pt(14), bold=False,
                color=DARK_GRAY, align=PP_ALIGN.LEFT, wrap=True):
    txBox = slide.shapes.add_textbox(x, y, w, h)
    tf    = txBox.text_frame
    tf.word_wrap = wrap
    p  = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size  = font_size
    run.font.bold  = bold
    run.font.color.rgb = color
    return txBox


def add_para(tf, text, font_size=Pt(13), bold=False, color=DARK_GRAY,
             align=PP_ALIGN.LEFT, space_before=Pt(4)):
    from pptx.util import Pt as _Pt
    p = tf.add_paragraph()
    p.alignment = align
    p.space_before = space_before
    run = p.add_run()
    run.text = text
    run.font.size  = font_size
    run.font.bold  = bold
    run.font.color.rgb = color
    return p


def header_bar(slide, title, subtitle=None):
    """Full-width top bar with title."""
    bar = add_rect(slide, 0, 0, SLIDE_W, Inches(1.15), DARK_BLUE)
    add_textbox(slide, Inches(0.35), Inches(0.12), Inches(12.6), Inches(0.6),
                title, font_size=Pt(28), bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    if subtitle:
        add_textbox(slide, Inches(0.35), Inches(0.68), Inches(12.6), Inches(0.35),
                    subtitle, font_size=Pt(14), bold=False, color=RGBColor(0xB0, 0xC8, 0xE8),
                    align=PP_ALIGN.LEFT)


def footer(slide, label):
    add_rect(slide, 0, Inches(7.2), SLIDE_W, Inches(0.3), DARK_BLUE)
    add_textbox(slide, Inches(0.3), Inches(7.21), Inches(8), Inches(0.25),
                label, font_size=Pt(9), color=RGBColor(0xB0, 0xC8, 0xE8))
    add_textbox(slide, Inches(11.5), Inches(7.21), Inches(1.5), Inches(0.25),
                "SAGE-QA", font_size=Pt(9), bold=True,
                color=RGBColor(0xB0, 0xC8, 0xE8), align=PP_ALIGN.RIGHT)


def box_with_title(slide, x, y, w, h, title, lines,
                   title_color=WHITE, title_bg=MID_BLUE, body_bg=LIGHT_BLUE,
                   title_size=Pt(13), body_size=Pt(12)):
    """Rounded-corner-style content box (title bar + body)."""
    title_h = Inches(0.34)
    add_rect(slide, x, y, w, title_h, title_bg)
    add_textbox(slide, x + Inches(0.08), y + Inches(0.04),
                w - Inches(0.16), title_h,
                title, font_size=title_size, bold=True, color=title_color)
    body_h = h - title_h
    add_rect(slide, x, y + title_h, w, body_h, body_bg,
             line_color=MID_BLUE, line_width=Pt(0.75))
    txBox = slide.shapes.add_textbox(
        x + Inches(0.1), y + title_h + Inches(0.06),
        w - Inches(0.2), body_h - Inches(0.1))
    tf = txBox.text_frame
    tf.word_wrap = True
    first = True
    for line in lines:
        if first:
            p   = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        p.space_before = Pt(3)
        run = p.add_run()
        run.text = line
        run.font.size  = body_size
        run.font.color.rgb = DARK_GRAY


def metric_box(slide, x, y, w, h, value, label, delta=None,
               val_color=DARK_BLUE, bg=LIGHT_GRAY):
    add_rect(slide, x, y, w, h, bg, line_color=MID_BLUE, line_width=Pt(0.5))
    add_textbox(slide, x, y + Inches(0.06), w, Inches(0.48),
                value, font_size=Pt(32), bold=True, color=val_color,
                align=PP_ALIGN.CENTER)
    add_textbox(slide, x, y + Inches(0.52), w, Inches(0.26),
                label, font_size=Pt(10), color=MID_GRAY,
                align=PP_ALIGN.CENTER)
    if delta:
        d_color = GREEN if delta.startswith("+") else RED
        add_textbox(slide, x, y + Inches(0.74), w, Inches(0.2),
                    delta, font_size=Pt(10), bold=True, color=d_color,
                    align=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 0 — Motivation
# ══════════════════════════════════════════════════════════════════════════════

def slide0(prs):
    sl = blank_slide(prs)
    add_rect(sl, 0, 0, SLIDE_W, SLIDE_H, LIGHT_GRAY)

    header_bar(sl, "Motivation",
               "Why combining neural retrieval with symbolic reasoning matters for QA")
    footer(sl, "Slide 1 / 4  —  Motivation")

    # ── Top band: the gap statement ───────────────────────────────────────────
    add_rect(sl, 0, Inches(1.15), SLIDE_W, Inches(0.62), RGBColor(0xEA, 0xF2, 0xFB))
    add_textbox(sl, Inches(0.4), Inches(1.22), Inches(12.5), Inches(0.45),
                "Knowledge graphs and ontologies encode structured world knowledge. "
                "Question answering over them often requires chaining multiple facts and rules — "
                "yet today's best neural retrievers treat each candidate independently, "
                "leaving the assembly problem unsolved.",
                font_size=Pt(13), color=DARK_GRAY, wrap=True)

    # ── Three-column tension row ──────────────────────────────────────────────
    col_y = Inches(1.95)
    col_h = Inches(2.45)

    # Column A — Pure neural
    box_with_title(sl, Inches(0.25), col_y, Inches(3.85), col_h,
        "Pure Neural Retrieval",
        [
            "Learned dense embeddings score",
            "candidate subgraphs against the",
            "question representation.",
            "",
            "✓  Generalises across domains",
            "✓  Handles surface-form variation",
            "✗  Scores pieces independently —",
            "    no chain / proof awareness",
            "✗  EM@1 < 3 % on 2-hop text QA",
        ],
        title_bg=MID_BLUE, body_bg=LIGHT_BLUE, body_size=Pt(12))

    # Arrow
    add_textbox(sl, Inches(4.13), col_y + Inches(0.95), Inches(0.5), Inches(0.4),
                "⚡", font_size=Pt(22), color=ORANGE, align=PP_ALIGN.CENTER)

    # Column B — Pure symbolic
    box_with_title(sl, Inches(4.65), col_y, Inches(3.85), col_h,
        "Pure Symbolic / Rule-Based",
        [
            "Hand-crafted inference rules over",
            "knowledge graph triples or OWL",
            "ontology axioms.",
            "",
            "✓  Fully interpretable chains",
            "✓  Correct under formal semantics",
            "✗  Brittle to noisy / incomplete KGs",
            "✗  Cannot rank ambiguous candidates",
            "✗  No learning from training data",
        ],
        title_bg=TEAL, body_bg=RGBColor(0xE0, 0xF4, 0xF6), body_size=Pt(12))

    # Arrow
    add_textbox(sl, Inches(8.53), col_y + Inches(0.95), Inches(0.5), Inches(0.4),
                "⚡", font_size=Pt(22), color=ORANGE, align=PP_ALIGN.CENTER)

    # Column C — NeSy opportunity
    box_with_title(sl, Inches(9.05), col_y, Inches(3.98), col_h,
        "NeSy Opportunity",
        [
            "Use the GNN to score relevance;",
            "use symbolic structure to compose",
            "the best reasoning chain.",
            "",
            "✓  Neural scoring handles noise",
            "    and lexical variation",
            "✓  Symbolic composer ensures the",
            "    selected chain is connected,",
            "    bridging, and proof-complete",
            "✓  No extra labelled data needed",
        ],
        title_bg=DARK_BLUE, body_bg=RGBColor(0xEA, 0xF2, 0xFB), body_size=Pt(12))

    # ── Bottom row: two concrete motivating examples ───────────────────────────
    add_textbox(sl, Inches(0.3), Inches(4.52), Inches(12.7), Inches(0.28),
                "Concrete motivating examples",
                font_size=Pt(12), bold=True, color=DARK_BLUE)

    ex_y = Inches(4.82)
    ex_h = Inches(2.25)

    # Example 1 — text multi-hop
    box_with_title(sl, Inches(0.25), ex_y, Inches(6.15), ex_h,
        "Text Multi-Hop  (2WikiMultiHopQA)",
        [
            'Q: "Where was the director of film Sea Sorrow born?"',
            "",
            "Required chain:",
            "  SENT::Sea Sorrow::0  →  mentions director Angelina Jolie",
            "  SENT::Angelina Jolie::0  →  born in Los Angeles",
            "",
            "GNN alone: scores each sentence in isolation — misses",
            "the cross-article bridge.   SAGE-QA: detects the bridge",
            "(Angelina Jolie appears in Sea Sorrow text) and selects",
            "both sentences.   LLM answer: Los Angeles  ✓",
        ],
        title_bg=MID_BLUE, body_bg=LIGHT_BLUE, body_size=Pt(11))

    # Example 2 — OWL proof
    box_with_title(sl, Inches(6.65), ex_y, Inches(6.4), ex_h,
        "OWL Ontology Proof  (FamilyOWL-2hop)",
        [
            'Q: "Is Mary Green a blood relative of Rebecca Green?"',
            "",
            "Required proof:",
            "  Fact:  mary_green_1803 isSiblingOf rebecca_green_1800",
            "  Rule:  isSiblingOf rdfs:subPropertyOf isBloodrelationOf",
            "",
            "GNN alone: retrieves the fact triple but not the axiom",
            "that makes it entail the answer.  SAGE-QA Proof mode:",
            "pairs fact + rule into a complete OWL entailment.",
            "LLM answer: Yes  ✓",
        ],
        title_bg=TEAL, body_bg=RGBColor(0xE0, 0xF4, 0xF6), body_size=Pt(11))


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — Problem & Motivation
# ══════════════════════════════════════════════════════════════════════════════

def slide1(prs):
    sl = blank_slide(prs)
    add_rect(sl, 0, 0, SLIDE_W, SLIDE_H, LIGHT_GRAY)  # background

    header_bar(sl, "SAGE-QA: Symbolic-Augmented Evidence Graph QA",
               "Why multi-hop QA needs more than a GNN")
    footer(sl, "Slide 2 / 4  —  Problem & Motivation")

    # ── Left column: the challenge ────────────────────────────────────────────
    box_with_title(sl, Inches(0.3), Inches(1.3), Inches(4.1), Inches(5.65),
        "The Multi-Hop QA Challenge",
        [
            "Multi-hop questions require chaining evidence across",
            "multiple sentences or ontology axioms:",
            "",
            "• Bridging:  film → director → birthplace",
            "• Comparison:  who lived longer, X or Y?",
            "• OWL Proof:  isParentOf ⊑ hasAncestor (subPropertyOf)",
            "",
            "A single retrieved sentence rarely suffices —",
            "the answer emerges only when the right pieces",
            "are assembled into a coherent reasoning chain.",
        ],
        title_bg=DARK_BLUE, body_bg=RGBColor(0xEA, 0xF2, 0xFB),
        body_size=Pt(12.5))

    # ── Center column: GNN limitation ─────────────────────────────────────────
    box_with_title(sl, Inches(4.6), Inches(1.3), Inches(4.1), Inches(5.65),
        "GNN Subgraph Retrieval — Strong but Incomplete",
        [
            "A GNN encoder scores candidate subgraphs",
            "by matching question embeddings against a",
            "KG-style evidence graph.",
            "",
            "✓  Excellent at identifying relevant axioms",
            "     and sentences individually",
            "",
            "✗  Does not reason about how pieces connect",
            "✗  Picks the highest-scored single candidate",
            "     — often a fragment, not a full proof",
            "",
            "Result: EM@1 < 3% on HotpotQA / 2WikiMHQA",
            "even when individual candidates are correct.",
        ],
        title_bg=MID_BLUE, body_bg=LIGHT_BLUE, body_size=Pt(12.5))

    # ── Right column: insight ─────────────────────────────────────────────────
    box_with_title(sl, Inches(8.9), Inches(1.3), Inches(4.1), Inches(5.65),
        "Key Insight",
        [
            "Symbolic structure is the missing link.",
            "",
            "Text benchmarks: sentences from different",
            "Wikipedia articles must be bridged — the",
            "answer entity appears in article A but the",
            "question asks about it via article B.",
            "",
            "OWL benchmarks: a fact triple is only valid",
            "proof when paired with the right axiom",
            "(inverseOf, subPropertyOf, Symmetric…).",
            "",
            "→  Reranking that understands these structural",
            "    patterns can assemble the correct chain.",
        ],
        title_bg=TEAL, body_bg=RGBColor(0xE0, 0xF4, 0xF6), body_size=Pt(12.5))


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — SAGE-QA Architecture
# ══════════════════════════════════════════════════════════════════════════════

def slide2(prs):
    sl = blank_slide(prs)
    add_rect(sl, 0, 0, SLIDE_W, SLIDE_H, LIGHT_GRAY)

    header_bar(sl, "SAGE-QA Architecture",
               "Three-stage pipeline with mode-aware symbolic reranking")
    footer(sl, "Slide 3 / 4  —  Architecture")

    # ── Pipeline arrow row ────────────────────────────────────────────────────
    stage_y  = Inches(1.35)
    stage_h  = Inches(1.1)
    arrow_y  = stage_y + Inches(0.37)

    stages = [
        (Inches(0.25),  Inches(2.3),  MID_BLUE,  "① Evidence Graph",   "KG-style graph\nbuilt from question\n+ context sentences"),
        (Inches(3.05),  Inches(2.3),  MID_BLUE,  "② GNN Retriever",    "Scores all candidate\nsubgraphs; outputs\nper-axiom probability"),
        (Inches(5.85),  Inches(2.3),  DARK_BLUE, "③ SymbolicComposer", "Beam search over\naxiom pool; assembles\nbest reasoning chain"),
        (Inches(8.65),  Inches(2.3),  TEAL,      "④ LLM Answer",       "Gemma / GPT reads\ncomposed context;\ngenerates final answer"),
    ]
    for sx, sw, sc, title, body in stages:
        add_rect(sl, sx, stage_y, sw, stage_h, sc)
        add_textbox(sl, sx + Inches(0.08), stage_y + Inches(0.04),
                    sw - Inches(0.16), Inches(0.32),
                    title, font_size=Pt(12), bold=True, color=WHITE)
        add_textbox(sl, sx + Inches(0.08), stage_y + Inches(0.36),
                    sw - Inches(0.16), Inches(0.68),
                    body, font_size=Pt(10.5), color=WHITE)

    # Arrows between stages
    for ax in [Inches(2.58), Inches(5.38), Inches(8.18)]:
        add_textbox(sl, ax, arrow_y, Inches(0.45), Inches(0.35),
                    "▶", font_size=Pt(18), bold=True,
                    color=DARK_BLUE, align=PP_ALIGN.CENTER)

    # ── Two-mode detail boxes ─────────────────────────────────────────────────
    mode_y = Inches(2.72)
    # Divider label
    add_textbox(sl, Inches(0.3), mode_y, Inches(12.7), Inches(0.3),
                "SymbolicComposer — Two Operating Modes",
                font_size=Pt(13), bold=True, color=DARK_BLUE)

    box_with_title(sl, Inches(0.3), mode_y + Inches(0.32), Inches(6.15), Inches(4.15),
        "Text-Chain Mode  (HotpotQA, 2WikiMultiHopQA)",
        [
            "Input: SENT::<article>::<idx>::<sentence text>",
            "",
            "Edge types built between sentence nodes:",
            "  • Same-article adjacency  (weight 0.55–0.65)",
            "  • Cross-entity bridge  (weight 0.75)",
            "    entity_A appears in text of sentence from entity_B",
            "  • Query-entity virtual bridge  (weight 0.52)",
            "    for comparison questions: both compared entities",
            "    get an explicit edge so beam search explores them",
            "",
            "Scoring bonuses: cross-page (+), bridge (+),",
            "query entity coverage (+), same-page redundancy (−),",
            "oversize >4 units (−)",
        ],
        title_bg=MID_BLUE, body_bg=LIGHT_BLUE, body_size=Pt(11.5))

    box_with_title(sl, Inches(6.65), mode_y + Inches(0.32), Inches(6.35), Inches(4.15),
        "Proof Mode  (FamilyOWL, PizzaOWL)",
        [
            "Input: OWL axioms (facts & schema rules)",
            "  Fact:   subject predicate object",
            "  Rule:   SymmetricObjectProperty(P)",
            "          InverseObjectProperties(P, Q)",
            "          SubObjectPropertyOf(P, Q)",
            "",
            "Edge types: shared entities/properties between axioms",
            "",
            "Ideal subgraph = 1 fact + 1 rule",
            "  (together they entail the query under OWL semantics)",
            "",
            "Scoring bonuses: connectivity (+), fact-rule mix (+),",
            "query entity/property alignment (+),",
            "oversize >2 units (−)",
            "",
            "Note: GNN alone achieves 98.6% Hit@1 on OWL —",
            "Proof mode is used for borderline cases.",
        ],
        title_bg=TEAL, body_bg=RGBColor(0xE0, 0xF4, 0xF6), body_size=Pt(11.5))


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — Results
# ══════════════════════════════════════════════════════════════════════════════

def slide3(prs):
    sl = blank_slide(prs)
    add_rect(sl, 0, 0, SLIDE_W, SLIDE_H, LIGHT_GRAY)

    header_bar(sl, "Evaluation Results",
               "200-example test sets  |  Retrieval metrics: EM@1 and Hit@1 (Jaccard ≥ 0.5)")
    footer(sl, "Slide 4 / 4  —  Results")

    # ── 2WikiMultiHopQA ───────────────────────────────────────────────────────
    box_with_title(sl, Inches(0.25), Inches(1.3), Inches(6.1), Inches(2.7),
        "2WikiMultiHopQA  (n=200)",
        [], title_bg=DARK_BLUE, body_bg=RGBColor(0xEA, 0xF2, 0xFB))

    metrics_2w = [
        (Inches(0.35), "2.5%",  "GNN EM@1",    None,       DARK_GRAY),
        (Inches(1.65), "33.5%", "SAGE EM@1",   "+31.0 pp", GREEN),
        (Inches(3.05), "22.0%", "GNN Hit@1",   None,       DARK_GRAY),
        (Inches(4.35), "68.0%", "SAGE Hit@1",  "+46.0 pp", GREEN),
    ]
    for mx, val, lbl, delta, vc in metrics_2w:
        metric_box(sl, mx, Inches(1.78), Inches(1.15), Inches(1.0),
                   val, lbl, delta, val_color=vc, bg=WHITE)

    # Per-type breakdown
    add_textbox(sl, Inches(0.35), Inches(2.88), Inches(5.8), Inches(0.22),
                "EM@1 by question type (SAGE-QA):", font_size=Pt(10.5),
                bold=True, color=DARK_BLUE)
    rows_2w = [
        ("comparison",        "47", "63.8%", "+61.7 pp"),
        ("compositional",     "91", "33.0%", "+29.7 pp"),
        ("inference",         "20", "25.0%", "+25.0 pp"),
        ("bridge_comparison", "42",  "4.8%",  "+2.4 pp"),
    ]
    for i, (qt, n, em, d) in enumerate(rows_2w):
        ry = Inches(3.1) + i * Inches(0.27)
        c = GREEN if float(em.rstrip('%')) > 10 else ORANGE
        add_textbox(sl, Inches(0.4), ry, Inches(2.5), Inches(0.24),
                    f"{qt}", font_size=Pt(10), color=DARK_GRAY)
        add_textbox(sl, Inches(2.9), ry, Inches(0.5), Inches(0.24),
                    f"n={n}", font_size=Pt(10), color=MID_GRAY)
        add_textbox(sl, Inches(3.5), ry, Inches(0.7), Inches(0.24),
                    em, font_size=Pt(10), bold=True, color=c)
        add_textbox(sl, Inches(4.3), ry, Inches(1.0), Inches(0.24),
                    d, font_size=Pt(10), color=c)

    # ── HotpotQA ──────────────────────────────────────────────────────────────
    box_with_title(sl, Inches(6.6), Inches(1.3), Inches(6.45), Inches(2.7),
        "HotpotQA  (n=200)",
        [], title_bg=DARK_BLUE, body_bg=RGBColor(0xEA, 0xF2, 0xFB))

    metrics_hp = [
        (Inches(6.70), "2.0%",  "GNN EM@1",   None,       DARK_GRAY),
        (Inches(8.00), "35.5%", "SAGE EM@1",  "+33.5 pp", GREEN),
        (Inches(9.40), "29.0%", "GNN Hit@1",  None,       DARK_GRAY),
        (Inches(10.70),"57.0%", "SAGE Hit@1", "+28.0 pp", GREEN),
    ]
    for mx, val, lbl, delta, vc in metrics_hp:
        metric_box(sl, mx, Inches(1.78), Inches(1.15), Inches(1.0),
                   val, lbl, delta, val_color=vc, bg=WHITE)

    add_textbox(sl, Inches(6.7), Inches(2.88), Inches(6.2), Inches(0.22),
                "Improvement breakdown (HotpotQA, n=200):", font_size=Pt(10.5),
                bold=True, color=DARK_BLUE)
    rows_hp = [
        ("Improved",   "135 / 200", GREEN),
        ("Unchanged",  " 37 / 200", MID_GRAY),
        ("Regressed",  " 28 / 200", ORANGE),
    ]
    for i, (lbl, val, c) in enumerate(rows_hp):
        ry = Inches(3.1) + i * Inches(0.27)
        add_textbox(sl, Inches(6.75), ry, Inches(2.5), Inches(0.24),
                    lbl, font_size=Pt(10), color=DARK_GRAY)
        add_textbox(sl, Inches(9.5), ry, Inches(1.5), Inches(0.24),
                    val, font_size=Pt(10), bold=True, color=c)

    # ── OWL observation ───────────────────────────────────────────────────────
    box_with_title(sl, Inches(0.25), Inches(4.2), Inches(12.8), Inches(2.65),
        "FamilyOWL_2hop — OWL Benchmarks: GNN is Already Strong",
        [], title_bg=TEAL, body_bg=RGBColor(0xE0, 0xF4, 0xF6))

    # OWL metrics row
    owl_metrics = [
        (Inches(0.4),  "98.6%", "GNN Hit@1",    None,       GREEN),
        (Inches(1.7),  "55.2%", "GNN EM@1",     None,       DARK_BLUE),
        (Inches(3.0),  "63.3%", "SAGE Hit@1",   "−35.3 pp", RED),
        (Inches(4.3),  "44.8%", "SAGE EM@1",    "−10.5 pp", RED),
    ]
    for mx, val, lbl, delta, vc in owl_metrics:
        metric_box(sl, mx, Inches(4.67), Inches(1.15), Inches(0.95),
                   val, lbl, delta, val_color=vc, bg=WHITE)

    add_textbox(sl, Inches(5.7), Inches(4.67), Inches(7.1), Inches(1.85),
                "GNN learns OWL proof structure directly during training and already achieves near-perfect "
                "retrieval (98.6% Hit@1). Applying SymbolicComposer reranking introduces noise: the "
                "fact-rule-mix bonus incentivizes adding extra axioms even when the GNN's 2-unit "
                "selection is already the correct proof.\n\n"
                "→  For OWL benchmarks, the production pipeline uses GNN neural scores directly "
                "(or lightweight row-level sageqa_proof_adjustment), not global SymbolicComposer "
                "reranking.",
                font_size=Pt(11.5), color=DARK_GRAY)


# ── Main ───────────────────────────────────────────────────────────────────────

prs = new_prs()
slide0(prs)
slide1(prs)
slide2(prs)
slide3(prs)

out = "outputs/SAGE_QA_slides.pptx"
import pathlib; pathlib.Path("outputs").mkdir(exist_ok=True)
prs.save(out)
print(f"Saved: {out}")
