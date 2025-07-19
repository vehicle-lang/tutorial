import type { CollectionEntry, InferEntrySchema } from "astro:content";
import path from "path";

/** Page title. */
export const pageTitle: string = "Vehicle Tutorial";

/** Chapters. */
export type Chapter = CollectionEntry<"chapters">;
export type ChapterData = InferEntrySchema<"chapters">;

export function getChapterNumber(chapter: Chapter): number {
  if (chapter.filePath !== undefined) {
    const basename = path.basename(chapter.filePath, ".md");
    const match = basename.match(/^(?<chapterNumber>\d+)-/);
    const chapterNumber = Number(match?.groups?.chapterNumber);
    return chapterNumber;
  }
  return 0;
}

export function byChapterNumber(chapter1: Chapter, chapter2: Chapter): number {
  return getChapterNumber(chapter1) - getChapterNumber(chapter2);
}
