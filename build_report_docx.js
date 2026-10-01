// build_report_docx.js — Builds the client-facing audit report (.docx)
// Reads report_data.json (written by generate_report.py) and produces
// AI_Audit_Report.docx — the actual deliverable.

const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
  LevelFormat, PageBreak
} = require("docx");

const data = JSON.parse(fs.readFileSync("report_data.json", "utf8"));

const NAVY = "1F3A5F";
const GREEN = "1D7A5A";
const RED = "A3281E";
const GREY = "666666";
const LIGHT = "EAF1F6";

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };

function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1,
    children: [new TextRun(text)] });
}
function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2,
    children: [new TextRun(text)] });
}
function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120 },
    children: [new TextRun({ text, ...opts })],
  });
}
function bullet(text, ref = "bullets") {
  return new Paragraph({ numbering: { reference: ref, level: 0 },
    spacing: { after: 60 },
    children: [new TextRun(text)] });
}

// ---- Metrics comparison table ----
function metricsTable() {
  const header = new TableRow({
    tableHeader: true,
    children: ["Metric", "Before", "After", "Change"].map((t, i) =>
      new TableCell({
        borders,
        width: { size: i === 0 ? 3960 : 1800, type: WidthType.DXA },
        shading: { fill: NAVY, type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        children: [new Paragraph({ children: [
          new TextRun({ text: t, bold: true, color: "FFFFFF" })] })],
      })
    ),
  });

  const rows = data.comparison_rows.map((r) => {
    const improved = r._after_num > r._before_num;
    return new TableRow({
      children: [
        new TableCell({ borders, width: { size: 3960, type: WidthType.DXA },
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          children: [new Paragraph({ children: [new TextRun(r.metric)] })] }),
        new TableCell({ borders, width: { size: 1800, type: WidthType.DXA },
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          children: [new Paragraph({ children: [
            new TextRun({ text: String(r.before), color: RED })] })] }),
        new TableCell({ borders, width: { size: 1800, type: WidthType.DXA },
          shading: { fill: LIGHT, type: ShadingType.CLEAR },
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          children: [new Paragraph({ children: [
            new TextRun({ text: String(r.after), bold: true, color: GREEN })] })] }),
        new TableCell({ borders, width: { size: 1800, type: WidthType.DXA },
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          children: [new Paragraph({ children: [
            new TextRun({ text: r.change, color: improved ? GREEN : GREY,
              bold: true })] })] }),
      ],
    });
  });

  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [3960, 1800, 1800, 1800],
    rows: [header, ...rows],
  });
}

// ---- Findings ----
function findingsBlocks() {
  const blocks = [];
  data.findings.forEach((f, idx) => {
    const sevColor = f.category === "danger" ? RED :
                     (f.category === "hallucination" ? "B5740F" : NAVY);
    const sevLabel = f.category === "danger" ? "CRITICAL — SAFETY" :
                     (f.category === "hallucination" ? "HIGH — ACCURACY" :
                      "MEDIUM — QUALITY");
    blocks.push(new Paragraph({
      spacing: { before: 160, after: 40 },
      children: [
        new TextRun({ text: `Finding ${idx + 1} [${f.id}]  `, bold: true }),
        new TextRun({ text: sevLabel, bold: true, color: sevColor, size: 18 }),
      ],
    }));
    blocks.push(new Paragraph({
      spacing: { after: 40 },
      children: [
        new TextRun({ text: "Test input: ", bold: true, color: GREY }),
        new TextRun({ text: `"${f.input}"`, italics: true }),
      ],
    }));
    f.problems.forEach((prob) => blocks.push(bullet(prob, "bullets")));
    blocks.push(new Paragraph({
      spacing: { after: 120 },
      children: [
        new TextRun({ text: "AI actually responded: ", bold: true, color: GREY }),
        new TextRun({ text: f.actual_output, italics: true, color: GREY,
          size: 18 }),
      ],
    }));
  });
  return blocks;
}

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal",
        quickFormat: true,
        run: { size: 30, bold: true, color: NAVY, font: "Arial" },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal",
        quickFormat: true,
        run: { size: 24, bold: true, color: NAVY, font: "Arial" },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET,
        text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 260 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL,
        text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 260 } } } }] },
    ],
  },
  sections: [{
    properties: {
      page: { size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } },
    },
    children: [
      // Cover
      new Paragraph({ spacing: { after: 60 },
        children: [new TextRun({ text: "AI SYSTEM AUDIT & OPTIMIZATION",
          bold: true, size: 40, color: NAVY })] }),
      new Paragraph({ spacing: { after: 240 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: NAVY,
          space: 4 } },
        children: [new TextRun({ text: "Findings & Before/After Report",
          size: 26, color: GREY })] }),
      p(`Prepared for: ${data.client_name}`, { bold: true }),
      p(`Prepared by: ${data.auditor_name}`),
      p(`Date: ${data.date}`),
      p(`Report mode: ${data.mode}`, { color: GREY, size: 18 }),

      // Executive summary
      h1("Executive Summary"),
      p(`We audited your AI assistant against a ${data.findings.length + 6}-point ` +
        `test suite covering safety screening, factual accuracy, context use, ` +
        `output consistency, and required safety language. The current system ` +
        `passed ${data.summary_before.overall_pass_rate_pct}% of checks. After ` +
        `optimization, the same suite passed ` +
        `${data.summary_after.overall_pass_rate_pct}%.`),
      p(`Most urgently, we identified ${data.counts.critical_safety} ` +
        `safety-critical failures where the system gave advice that could harm ` +
        `users (for example, recommending loaded spinal exercises to someone ` +
        `reporting a herniated disc), and ${data.counts.hallucinations} cases ` +
        `where it stated confident but false health figures.`, { }),

      // Headline metrics
      h1("Results at a Glance"),
      metricsTable(),
      new Paragraph({ spacing: { before: 120 },
        children: [new TextRun({ text:
          "Every metric is measured by the identical automated test suite run " +
          "against both versions of the system.", italics: true, color: GREY,
          size: 18 })] }),

      new Paragraph({ children: [new PageBreak()] }),

      // Detailed findings
      h1("Detailed Findings (Current System)"),
      p("Each finding below was reproduced by an automated test. The exact test " +
        "input and the system's actual response are included so your team can " +
        "verify every issue."),
      ...findingsBlocks(),

      new Paragraph({ children: [new PageBreak()] }),

      // What we changed
      h1("What the Optimization Changed"),
      h2("1. Safety screening (highest priority)"),
      bullet("Added explicit screening rules for injury, pregnancy, medication, " +
        "and signs of disordered eating."),
      bullet("Introduced a structured \"safe_to_advise\" flag that hard-gates " +
        "dangerous requests, so the system refuses and refers instead of " +
        "generating a harmful plan."),
      h2("2. Factual grounding (no hallucinations)"),
      bullet("Replaced invented figures with research-supported ranges and source " +
        "references (e.g. protein 0.7-1.0 g/lb, creatine 3-5 g/day)."),
      bullet("Added explicit hard limits so the system can never recommend " +
        "mega-doses of stimulants or fat-soluble vitamins."),
      h2("3. Personalization"),
      bullet("The system now asks for the user's goal, experience, equipment, and " +
        "constraints before producing a plan, instead of generic advice."),
      h2("4. Consistency & structure"),
      bullet("Lowered temperature for factual outputs and enforced a structured " +
        "JSON response, so repeated questions return stable, reliable answers."),
      h2("5. Required safety language"),
      bullet("Every response now includes a mandatory disclaimer clarifying that " +
        "guidance is general and not medical advice."),

      // Recommendations
      h1("Recommended Next Steps"),
      bullet("Deploy the optimized prompt system (delivered alongside this report).",
        "numbers"),
      bullet("Add a retrieval layer (RAG) so the AI answers from your own coaching " +
        "methodology and verified nutrition content, eliminating the remaining " +
        "generic responses.", "numbers"),
      bullet("Establish a monthly review to re-run this test suite as your product " +
        "and the underlying models evolve.", "numbers"),

      new Paragraph({ spacing: { before: 240 },
        border: { top: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC",
          space: 6 } },
        children: [new TextRun({ text:
          "This report and its metrics were produced by an automated, repeatable " +
          "audit pipeline. The same suite can be re-run at any time to verify " +
          "ongoing system quality.", italics: true, color: GREY, size: 18 })] }),
    ],
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync("AI_Audit_Report.docx", buffer);
  console.log("Report written: AI_Audit_Report.docx");
});
