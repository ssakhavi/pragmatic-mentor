// Sorts content collection items by date in descending order (newest first)
export function sortByDate<T extends { data: { date: string } }>(items: T[]): T[] {
  return items.sort(
    (a, b) => new Date(b.data.date).valueOf() - new Date(a.data.date).valueOf()
  );
}
