// Turns a display tag like "ML-as-a-Service" or "Software Engineering"
// into a URL-safe slug ("ml-as-a-service", "software-engineering").
// Used to route /tags/[tag]/ pages and to link tags from post pages.
export function slugifyTag(tag: string): string {
  return tag
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}
