import { visit } from "unist-util-visit";

const HIGHLIGHT_TAIL_RE = /==([\s\S]+?)==$/;

function noteTypeForLabel(label) {
  for (const prefix of ["info", "warn", "cite"]) {
    if (label.startsWith(prefix)) return prefix;
  }
  return "hl";
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

        usedNotes.push({ label, type: noteType, text: definitions[label] || "" });

        const replacement = [];
        if (before) replacement.push({ type: "text", value: before });
        replacement.push({
          type: "html",
          value: `<mark data-type="${noteType}" data-note="${label}">${phrase}</mark>`,
        });

        children.splice(i - 1, 2, ...replacement);
        i = i - 1 + replacement.length - 1;
      }
    });

    tree.children = tree.children.filter((n) => n.type !== "footnoteDefinition");

    file.data.astro = file.data.astro || {};
    file.data.astro.frontmatter = file.data.astro.frontmatter || {};
    file.data.astro.frontmatter.notes = usedNotes;
  };
}
