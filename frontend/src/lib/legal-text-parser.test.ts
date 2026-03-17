import { describe, it, expect } from "vitest";
import { parseTextRuns, ART_RUN_RE, EXTERNAL_CONTEXT_RE } from "./legal-text-parser";

describe("ART_RUN_RE", () => {
  it("matches English singular 'Article N'", () => {
    ART_RUN_RE.lastIndex = 0;
    expect(ART_RUN_RE.test("see Article 92 for details")).toBe(true);
  });

  it("matches English plural 'Articles N and M'", () => {
    ART_RUN_RE.lastIndex = 0;
    expect(ART_RUN_RE.test("see Articles 92 and 93")).toBe(true);
  });

  it("matches Italian singular 'Articolo N'", () => {
    ART_RUN_RE.lastIndex = 0;
    expect(ART_RUN_RE.test("vedi Articolo 92")).toBe(true);
  });

  it("matches Italian plural 'Articoli N e M'", () => {
    ART_RUN_RE.lastIndex = 0;
    expect(ART_RUN_RE.test("vedi Articoli 92 e 93")).toBe(true);
  });
});

describe("EXTERNAL_CONTEXT_RE", () => {
  it("excludes English 'of Regulation ...'", () => {
    expect(EXTERNAL_CONTEXT_RE.test(" of Regulation (EU) No 648/2012")).toBe(true);
  });

  it("excludes English 'of Directive ...'", () => {
    expect(EXTERNAL_CONTEXT_RE.test(" of Directive 2013/36/EU")).toBe(true);
  });

  it("excludes Italian 'del Regolamento ...'", () => {
    expect(EXTERNAL_CONTEXT_RE.test(" del Regolamento (UE) n. 648/2012")).toBe(true);
  });

  it("excludes Italian 'della Direttiva ...'", () => {
    expect(EXTERNAL_CONTEXT_RE.test(" della Direttiva 2013/36/UE")).toBe(true);
  });

  it("does not exclude plain article reference", () => {
    expect(EXTERNAL_CONTEXT_RE.test(" on own funds")).toBe(false);
  });
});

describe("parseTextRuns", () => {
  it("linkifies English 'Article 92'", () => {
    const runs = parseTextRuns("see Article 92 for details");
    expect(runs.some((r) => r.kind === "article" && r.num === "92")).toBe(true);
  });

  it("linkifies Italian 'Articolo 92'", () => {
    const runs = parseTextRuns("vedi Articolo 92 per i dettagli");
    expect(runs.some((r) => r.kind === "article" && r.num === "92")).toBe(true);
  });

  it("linkifies Italian 'Articoli 92 e 93'", () => {
    const runs = parseTextRuns("vedi Articoli 92 e 93");
    const articleNums = runs.filter((r) => r.kind === "article").map((r) => r.num);
    expect(articleNums).toEqual(["92", "93"]);
  });

  it("does NOT linkify article in Italian external ref", () => {
    const runs = parseTextRuns("Articolo 4 del Regolamento (UE) n. 648/2012");
    expect(runs.every((r) => r.kind !== "article")).toBe(true);
  });

  it("does NOT linkify article in English external ref", () => {
    const runs = parseTextRuns("Article 4 of Regulation (EU) No 648/2012");
    expect(runs.every((r) => r.kind !== "article")).toBe(true);
  });
});
