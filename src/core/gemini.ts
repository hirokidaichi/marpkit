import { GoogleGenerativeAI } from "@google/generative-ai";
import fs from "node:fs";
import path from "node:path";

// Latest Gemini embedding model (3072 dimensions, multilingual support)
const EMBEDDING_MODEL = "gemini-embedding-001";
const VISION_MODEL = "gemini-2.0-flash";

// Maximum batch size for batchEmbedContents API (API limit: 100)
const EMBEDDING_BATCH_SIZE = 100;
// Number of parallel batch requests (to maximize throughput within 5M tokens/min limit)
const PARALLEL_BATCHES = 3;

export class GeminiClient {
  private genAI: GoogleGenerativeAI;

  constructor(apiKey: string) {
    this.genAI = new GoogleGenerativeAI(apiKey);
  }

  async embed(text: string): Promise<Float32Array> {
    const model = this.genAI.getGenerativeModel({ model: EMBEDDING_MODEL });
    const result = await model.embedContent(text);
    return new Float32Array(result.embedding.values);
  }

  async embedBatch(
    texts: string[],
    onProgress?: (completed: number, total: number) => void
  ): Promise<Float32Array[]> {
    const model = this.genAI.getGenerativeModel({ model: EMBEDDING_MODEL });
    const results: Float32Array[] = new Array(texts.length);
    let completed = 0;

    // Split texts into chunks of EMBEDDING_BATCH_SIZE (max 100 per API call)
    const chunks: { startIndex: number; texts: string[] }[] = [];
    for (let i = 0; i < texts.length; i += EMBEDDING_BATCH_SIZE) {
      chunks.push({
        startIndex: i,
        texts: texts.slice(i, i + EMBEDDING_BATCH_SIZE),
      });
    }

    // Process chunks in parallel batches
    for (let i = 0; i < chunks.length; i += PARALLEL_BATCHES) {
      const parallelChunks = chunks.slice(i, i + PARALLEL_BATCHES);

      const responses = await Promise.all(
        parallelChunks.map((chunk) =>
          model.batchEmbedContents({
            requests: chunk.texts.map((text) => ({
              content: { role: "user", parts: [{ text }] },
            })),
          })
        )
      );

      // Store results in correct positions
      for (let j = 0; j < responses.length; j++) {
        const chunk = parallelChunks[j];
        const response = responses[j];
        for (let k = 0; k < response.embeddings.length; k++) {
          results[chunk.startIndex + k] = new Float32Array(
            response.embeddings[k].values
          );
          completed++;
          if (onProgress) {
            onProgress(completed, texts.length);
          }
        }
      }

      // Small delay between parallel batches to avoid rate limiting
      if (i + PARALLEL_BATCHES < chunks.length) {
        await sleep(100);
      }
    }

    return results;
  }

  async describeImage(imagePath: string): Promise<string> {
    const model = this.genAI.getGenerativeModel({ model: VISION_MODEL });

    const imageBuffer = fs.readFileSync(imagePath);
    const base64Image = imageBuffer.toString("base64");
    const mimeType = getMimeType(imagePath);

    const result = await model.generateContent([
      {
        inlineData: {
          mimeType,
          data: base64Image,
        },
      },
      {
        text: "この画像の内容を詳細に説明してください。スライドの文脈で検索に役立つようなキーワードを含めてください。",
      },
    ]);

    return result.response.text();
  }

  async describeImages(
    imagePaths: string[],
    onProgress?: (completed: number, total: number) => void
  ): Promise<Map<string, string>> {
    const descriptions = new Map<string, string>();

    for (let i = 0; i < imagePaths.length; i++) {
      const imagePath = imagePaths[i];
      try {
        const description = await this.describeImage(imagePath);
        descriptions.set(imagePath, description);
      } catch (error) {
        console.warn(`Failed to describe image: ${imagePath}`);
        descriptions.set(imagePath, "");
      }

      if (onProgress) {
        onProgress(i + 1, imagePaths.length);
      }

      // Rate limiting for vision API
      if (i + 1 < imagePaths.length) {
        await sleep(500);
      }
    }

    return descriptions;
  }
}

function getMimeType(filePath: string): string {
  const ext = path.extname(filePath).toLowerCase();
  const mimeTypes: Record<string, string> = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
  };
  return mimeTypes[ext] || "image/png";
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
