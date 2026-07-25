import { defineConfig } from "astro/config";
import remarkGfm from "remark-gfm";
import remarkHighlightNotes from "./src/plugins/remark-highlight-notes.mjs";

export default defineConfig({
  site: "https://www.siavashsakhavi.com",
  markdown: {
    remarkPlugins: [remarkGfm, remarkHighlightNotes],
  },
  build: {
    // Keep page CSS inlined in the HTML instead of extracted to /_astro/*.css.
    // Extracted stylesheets use root-absolute hrefs that only resolve when
    // served by a webserver; opening the built HTML directly as a local
    // file (file://) 404s on that link and silently falls back to
    // unstyled markup. Inlining keeps each page self-contained.
    inlineStylesheets: "always",
  },
});
