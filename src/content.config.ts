import { glob } from "astro/loaders";
import { z, defineCollection } from "astro:content";

const chapters = defineCollection({
  loader: glob({
    pattern: "**/[^_]*.md",
    base: "./src/chapters",
    generateId: ({ entry }) =>
      entry
        .toLowerCase()
        .replace(/^([0-9]+-)/,'')
        .replace(/\.md$/,''),
  }),
  schema: z.object({
    title: z.string(),
  }),
});

export const collections = { chapters };
