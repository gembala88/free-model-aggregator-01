/** Shared type definitions for free-model-aggregator-01. */

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json =>
  v !== null && typeof v === 'object' && !Array.isArray(v);

/** Provider directory entry from the brief. */
export interface ProviderDirEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

export interface ProviderDirectory {
  schema: string;
  providers: ProviderDirEntry[];
}

/** Quality table row (sealed truth). */
export interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface RatingsFile {
  schema: string;
  ratings: RatingRow[];
  best: { id: string; rating_milli: number }[];
}

/** Workload manifest request. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

export interface WorkloadFile {
  schema: string;
  manifest: WorkloadRequest[];
}

/** Parsed brief with three documents. */
export interface Brief {
  directory: ProviderDirectory;
  ratings: RatingsFile | null;
  workload: WorkloadFile | null;
  error: string | null;
}

/** Model as listed by a provider's catalog route. */
export interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** One provider's catalog probe result. */
export interface Probe {
  provider: ProviderDirEntry;
  ok: boolean;
  status: number;
  models: ProbedModel[];
  error: string | null;
}

/** Capabilities resolved from listing + quality table. */
export interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** Catalog model entry (registered + listed). */
export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

export interface CatalogReport {
  providers: CatalogProvider[];
}

/** Provider live state during a trial. */
export interface ProviderState {
  calls: number;
  streak: number;
  dead: boolean;
}

/** Extracted echo fields from a client request. */
export interface Extracted {
  system: string;
  userText: string;
  userImages: ImageRef[];
  tools: { name: string; description: string; parameters: Json }[];
  max_tokens: number | null;
  stops: string[];
  json_mode: boolean;
}

export interface ImageRef {
  kind: 'data' | 'url';
  data: string;
  media_type: string;
  url: string;
}

/** Parsed provider answer. */
export interface ProviderAnswer {
  text: string;
  input: number;
  output: number;
  stop: string | null;
}

/** Input to client answer builders. */
export interface ClientAnswerInput {
  text: string;
  input: number;
  output: number;
  stop: string;
  model: string;
}

/** Served model for GET /v1/models. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** Candidate for routing. */
export interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
  providerIndex: number;
}