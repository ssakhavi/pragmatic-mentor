import { defineCollection, z } from "astro:content";

const blog = defineCollection({
  type: "content",
  schema: z.object({
    title: z.string(),
    date: z.string(),
    tags: z.array(z.string()).default([]),
    description: z.string().optional(),
  }),
});

const pages = defineCollection({
  type: "content",
  schema: z.object({
    title: z.string(),
    lead: z.string().optional(),
  }),
});

const glossary = defineCollection({
  type: "content",
  schema: z.object({
    term: z.string(),
    summary: z.string(),
    // Optional: matches a blog post `tag` exactly, used to compute the
    // "related posts" list shown in the hover card.
    tag: z.string().optional(),
  }),
});

const microblog = defineCollection({
  type: "content",
  schema: z.object({
    title: z.string().optional(),
    date: z.string(),
  }),
});

export const collections = { blog, pages, glossary, microblog };
