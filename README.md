# The Pragmatic Mentor

Siavash Sakhavi's personal blog, built with [Astro](https://astro.build) and
[bun](https://bun.sh). Migrated from a Hugo/Congo site — the old version is
preserved on the `hugo-legacy` branch.

## Quick start

```bash
bun install
bun run dev       # http://localhost:4321
bun run build     # outputs to dist/
bun run preview   # serve the production build locally
```

## Project structure

```
src/
  content/
    blog/            # blog posts (markdown)
    pages/            # About, Favorites (markdown)
    glossary/         # reusable term definitions (markdown)
    config.ts         # content collection schemas
  layouts/
    PostLayout.astro  # blog post template (outline + body + notes)
    PageLayout.astro  # simple page template (About, Favorites, glossary terms)
  pages/
    index.astro       # homepage: bio + post list
    about.astro        # renders content/pages/about.md
    favorites.astro   # renders content/pages/favorites.md
    blog/[slug].astro # renders content/blog/*.md
    glossary/[slug].astro # renders content/glossary/*.md
  plugins/
    remark-highlight-notes.mjs  # powers ==highlight==[^note] and ==term==((slug))
public/
  CNAME               # custom domain for GitHub Pages
.github/workflows/
  gh-pages.yml        # builds with bun/Astro, deploys to GitHub Pages
```

## Features

### Three-column post layout

Blog posts (`PostLayout.astro`) render as three columns on wide screens:

- **Left — Contents.** Auto-generated from the post's `##` headings (plus a
  synthetic "Introduction" entry pointing at the top of the article). A
  small scroll-spy script tracks which section is currently in view and
  puts a dot next to the active entry, in addition to click-to-jump.
- **Center — the article** itself, in a readable ~680px column.
- **Right — Notes.** Sidebar-style highlighted phrases (see below) get a
  matching note here. Notes are dim by default and light up when you
  hover or click the highlighted phrase in the body (and vice versa). Each
  note has a small `^` backlink that jumps to where it's referenced in
  the text.

This collapses gracefully: the outline disappears under ~1150px, and the
notes column drops below the article under ~860px, so the site is fully
usable on mobile.

Pages that don't need the outline/notes machinery (About, Favorites) use
the simpler `PageLayout.astro` instead — same nav/footer/typography, single
column.

### Highlight + typed notes syntax

This is a custom Markdown convention (implemented in
`src/plugins/remark-highlight-notes.mjs`) for adding sidenote-style
annotations to a highlighted phrase:

```markdown
Glacial ice cores let scientists read ==80,000 years of atmospheric
history==[^info1] in a single vertical column of ice.

[^info1]: Trapped air bubbles preserve ancient CO2 levels; layer counting
works much like tree rings.
```

- Wrap the phrase you want annotated in `==double equals==`.
- Immediately follow it with a footnote reference `[^label]`.
- Define the note anywhere in the file with `[^label]: text`.

The label controls both the color/category and how the note is displayed
— there's no separate frontmatter field, it's all read from the label
text itself:

| Label contains | Effect |
|-----------------|--------|
| starts with `info` | info color (blue) |
| starts with `warn` | warning color (amber) |
| starts with `cite` | citation color (gray) |
| *(anything else)*  | generic highlight color (red) |
| contains `hover` anywhere | **display style**: shows as a floating popup right where you're reading, like a footnote — no entry in the sidebar |
| *(no `hover`)* | **display style** (default): shows as a persistent entry in the right-hand notes column |

So `[^info1]` is a sidebar-style info note, `[^info-hover1]` is the exact
same info-colored highlight but shown as a hover popup instead, and
`[^hover1]` is a hover-style popup with the generic highlight color. Mix
and match per note depending on whether the annotation is worth a
permanent spot in the margin or just a quick aside.

See `src/content/blog/mlaas-vs-mlops.md` for both variants in use.

### Glossary terms (hover-preview cards)

A second, separate syntax for reusable definitions — inspired by
LessWrong's wiki-tag hover cards — that references a standalone glossary
entry instead of an inline footnote:

```markdown
I watched a video on ==MLOps==((mlops)) vs ML-as-a-Service.
```

- Wrap the phrase in `==double equals==`, immediately followed by
  `((slug))`, where `slug` matches a file in `src/content/glossary/`.
- Hovering (or tapping) the phrase shows a card with the term's `term` and
  `summary` frontmatter fields, a "Read more" link to a full page for that
  term (`/glossary/slug/`), and — if the glossary entry sets a `tag` that
  matches one or more post tags — a "Related posts" list.

Glossary entries live in `src/content/glossary/*.md`:

```markdown
---
term: "MLOps"
summary: "One or two sentences for the hover card."
tag: "Machine Learning"   # optional — matches a blog post `tag` for "Related posts"
---

The full write-up, shown on the term's own page (`/glossary/mlops/`).
```

Unlike footnote notes, glossary terms are reusable across posts — define
the entry once, reference `==Term==((slug))` from as many posts as you
like.

### Content collections

Two Astro content collections, defined in `src/content/config.ts`:

- **`blog`** — schema: `title`, `date`, `tags` (array), `description`
  (optional). Rendered through `PostLayout.astro`.
- **`pages`** — schema: `title`, `lead` (optional). Currently just About
  and Favorites, rendered through `PageLayout.astro`.
- **`glossary`** — schema: `term`, `summary`, `tag` (optional). Powers the
  hover-preview definition cards; see "Glossary terms" above.

### Writing a new post

Add a file to `src/content/blog/your-slug.md`:

```markdown
---
title: "Your Title"
date: "2026-01-01"
tags: ["Tag One", "Tag Two"]
description: "One line for the homepage list (optional)"
---

Your content here. Use `##` headings if you want them to show up in the
outline. Cross-link other posts with relative paths, e.g.
`[other post](../other-post-slug/)`, not absolute `/blog/...` links — that
way they keep working regardless of whether the site is deployed at the
domain root or under a subpath.
```

The filename (minus `.md`) becomes the URL slug: `/blog/your-slug/`.

## Deployment

Deploys via GitHub Actions (`.github/workflows/gh-pages.yml`) on every push
to `main`: installs with bun, runs `bun run build`, and publishes `dist/`
to GitHub Pages.

The site is configured for the custom domain **www.siavashsakhavi.com**
(`astro.config.mjs` → `site`, and `public/CNAME`). For this to actually
resolve, two things need to be true outside this repo:

1. DNS: a `CNAME` record for `www` pointing at `ssakhavi.github.io`
   (configured at your registrar, e.g. Namecheap's Advanced DNS tab).
2. GitHub: Settings → Pages → Custom domain → `www.siavashsakhavi.com`,
   then "Enforce HTTPS" once it verifies.

If you ever want to fall back to the default GitHub Pages URL instead of
the custom domain, set in `astro.config.mjs`:

```js
site: "https://ssakhavi.github.io",
base: "/pragmatic-mentor/",
```

and remove `public/CNAME`. No template changes are needed either way —
every internal link is built from `import.meta.env.BASE_URL`, which
resolves correctly whether `base` is set or not.

## Branches

- **`main`** — the current Astro site.
- **`hugo-legacy`** — a frozen snapshot of the old Hugo/Congo site, kept
  for reference. Not deployed.
