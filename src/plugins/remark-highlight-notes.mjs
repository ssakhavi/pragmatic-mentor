import { visit } from "unist-util-visit";

const HIGHLIGHT_TAIL_RE = /==([\s\S]+?)==$/;
const GLOSSARY_RE = /==([\s\S]+?)==\(\(([a-z][a-z0-9-]*)\)\)/g;

function noteTypeForLabel(label) {
  for (const prefix of ["info", "warn", "cite"]) {
    if (label.startsWith(prefix)) return prefix;
  }
  return "hl";
}

// A label containing "hover" anywhere (e.g. "info-hover-1", "hover2")
// renders as a hover-only popup instead of a persistent entry in the
// notes column. This is the only thing that controls display style —
// no separate frontmatter field needed, same lightweight convention
// already used for color/type.
function noteStyleForLabel(label) {
  return label.includes("hover") ? "hover" : "sidebar";
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// Splits a text node's value on the ==term==((slug)) glossary syntax and
// returns an array of text/html nodes. Returns null if there's no match,
// so callers can skip splicing when nothing changed.
function splitGlossaryRefs(value, usedSlugs) {
  GLOSSARY_RE.lastIndex = 0;
  if (!GLOSSARY_RE.test(value)) return null;
  GLOSSARY_RE.lastIndex = 0;

  const parts = [];
  let lastIndex = 0;
  let match;
  while ((match = GLOSSARY_RE.exec(value))) {
    if (match.index > lastIndex) {
      parts.push({ type: "text", value: value.slice(lastIndex, match.index) });
    }
    const [, phrase, slug] = match;
    usedSlugs.add(slug);
    parts.push({
      type: "html",
      value: `<span class="term" data-term="${slug}">${phrase}<sup class="term-mark">°</sup></span>`,
    });
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < value.length) {
    parts.push({ type: "text", value: value.slice(lastIndex) });
  }
  return parts;
}

export default function remarkHighlightNotes() {
  return (tree, file) => {
    const definitions = {};
    visit(tree, "footnoteDefinition", (node) => {
      let text = "";
      visit(node, "text", (t) => {
        text += t.value;
      });
      definitions[node.identifier] = text.trim();
    });

    const usedNotes = [];

    visit(tree, (node) => Array.isArray(node.children), (parent) => {
      const children = parent.children;
      for (let i = 0; i < children.length; i++) {
        const node = children[i];
        if (node.type !== "footnoteReference") continue;

        const prev = children[i - 1];
        if (!prev || prev.type !== "text") continue;

        const match = HIGHLIGHT_TAIL_RE.exec(prev.value);
        if (!match) continue;

        const before = prev.value.slice(0, match.index);
        const phrase = match[1];
        const label = node.identifier;
        const noteType = noteTypeForLabel(label);
        const noteStyle = noteStyleForLabel(label);
        const text = definitions[label] || "";

        usedNotes.push({ label, type: noteType, style: noteStyle, text });

        const replacement = [];
        if (before) replacement.push({ type: "text", value: before });
        replacement.push({
          type: "html",
          value: `<mark data-type="${noteType}" data-style="${noteStyle}" data-note="${label}" id="note-ref-${label}" data-note-text="${escapeHtml(text)}">${phrase}</mark>`,
        });

        children.splice(i - 1, 2, ...replacement);
        i = i - 1 + replacement.length - 1;
      }
    });

    tree.children = tree.children.filter((n) => n.type !== "footnoteDefinition");

    // Second pass: glossary term references, ==phrase==((slug)). These
    // live in plain text nodes (no footnote syntax involved), so they
    // survive untouched by the pass above and can be handled independently.
    const usedGlossarySlugs = new Set();
    visit(tree, (node) => Array.isArray(node.children), (parent) => {
      const children = parent.children;
      for (let i = 0; i < children.length; i++) {
        const node = children[i];
        if (node.type !== "text") continue;
        const parts = splitGlossaryRefs(node.value, usedGlossarySlugs);
        if (!parts) continue;
        children.splice(i, 1, ...parts);
        i = i + parts.length - 1;
      }
    });

    file.data.astro = file.data.astro || {};
    file.data.astro.frontmatter = file.data.astro.frontmatter || {};
    file.data.astro.frontmatter.notes = usedNotes;
    file.data.astro.frontmatter.glossaryTerms = Array.from(usedGlossarySlugs);
  };
}
