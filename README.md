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
    config.ts         # content collection schemas
  layouts/
    PostLayout.astro  # blog post template (outline + body + notes)
    PageLayout.astro  # simple page template (About, Favorites)
  pages/
    index.astro       # homepage: bio + post list
    about.astro        # renders content/pages/about.md
    favorites.astro   # renders content/pages/favorites.md
    blog/[slug].astro # renders content/blog/*.md
  plugins/
    remark-highlight-notes.mjs  # powers the ==highlight==[^note] syntax
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
- **Right — Notes.** Any highlighted phrases in the post (see below) get a
  matching note here. Notes are dim by default and light up when you
  hover or click the highlighted phrase in the body (and vice versa).

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

The prefix of the label controls the color/category shown in the notes
column:

| Prefix   | Meaning                          |
|----------|-----------------------------------|
| `info`   | supporting detail, explanation    |
| `warn`   | caveat, risk, "read carefully"    |
| `cite`   | source, reference, citation       |
| *(other)*| falls back to a generic highlight |

None of the current posts use this yet (they were plain prose in Hugo), but
it's ready to use in any future post.

### Content collections

Two Astro content collections, defined in `src/content/config.ts`:

- **`blog`** — schema: `title`, `date`, `tags` (array), `description`
  (optional). Rendered through `PostLayout.astro`.
- **`pages`** — schema: `title`, `lead` (optional). Currently just About
  and Favorites, rendered through `PageLayout.astro`.

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
