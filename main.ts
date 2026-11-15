// free-model-aggregator-01 entry: quality-ranked free model aggregator.
// Listens on 0.0.0.0 at an ephemeral port, registers with the env, serves 20 client requests,
// polls /work, exits 0 on done. Implements budget-aware failover and deterministic routing.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  readBrief,
  joinBase,
  authHeaders,
  probeProvider,
  buildCatalog,
  ratingOfModel,
  modelFree,
  capabilitiesOf,
  meetsRequirements,
  type Probe,
  type CatalogReport,
  type ProviderDirEntry,
  type ProbedModel,
  type Requirement,
  type Json,
  isRecord,
  type ProviderState,
} from './lib/providers.ts';
import {
  openAIModelList,
  anthropicModelList,
  anthropicError,
  openAIError,
  extractOpenAI,
  extractAnthropic,
  openaiToOpenAI,
  anthropicToAnthropic,
  openaiToAnthropic,
  anthropicToOpenAI,
  openaiToGemini,
  anthropicToGemini,
  parseProviderAnswer,
  toClientStop,
  buildOpenAIAnswer,
  buildAnthropicAnswer,
  type ServedModel,
  type ProviderAnswer,
  type ClientAnswerInput,
  type Extracted,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const CALL_TIMEOUT_MS = 15000;
const MAX_BODY_BYTES = 2 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

// Load brief and probe providers.
const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;
const liveState = new Map<string, ProviderState>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = [];
  for (const provider of brief.directory.providers) {
    probes.push(await probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS));
  }
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  for (const probe of probes) if (probe.ok) liveState.set(probe.provider.id, { calls: 0, dead: false, streak: 0 });
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
  providerIndex: number;
}

function candidatesFor(requires: Requirement, clientWire: 'openai' | 'anthropic'): Candidate[] {
  const out: Candidate[] = [];
  for (let i = 0; i < probes.length; i++) {
    const probe = probes[i];
    if (!probe.ok) continue;
    const state = liveState.get(probe.provider.id);
    if (state === undefined || state.dead) continue;
    if (requires.json === true && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0, providerIndex: i });
    }
  }
  out.sort(
    (a, b) =>
      (liveState.get(a.probe.provider.id)?.streak ?? 0) - (liveState.get(b.probe.provider.id)?.streak ?? 0) ||
      b.rating - a.rating ||
      a.providerIndex - b.providerIndex ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return out;
}

function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function modelsHandler(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

async function callCandidate(candidate: Candidate, clientWire: 'openai' | 'anthropic', body: Json): Promise<Response> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  const model = candidate.model.id;
  let url: string;
  let request: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    request = clientWire === 'openai' ? openaiToOpenAI(body, model) : anthropicToOpenAI(body, model);
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    request = clientWire === 'anthropic' ? anthropicToAnthropic(body, model) : openaiToAnthropic(body, model);
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(model)}:generateContent`;
    request = clientWire === 'openai' ? openaiToGemini(body, model) : anthropicToGemini(body, model);
  }
  return fetch(url, { method: 'POST', headers, body: JSON.stringify(request), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
}

async function serve(
  res: ServerResponse,
  clientWire: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
  requires: Requirement,
): Promise<void> {
  const candidates = candidatesFor(requires, clientWire);
  if (candidates.length === 0) {
    return sendError(res, clientWire, 503, clientWire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  let lastError: Json | null = null;
  let lastStatus = 502;
  for (const candidate of candidates) {
    const state = liveState.get(candidate.probe.provider.id);
    if (state !== undefined) state.calls += 1;
    let upstream: Response;
    try {
      upstream = await callCandidate(candidate, clientWire, body);
    } catch (err) {
      if (state !== undefined) state.dead = true;
      lastError = clientWire === 'anthropic' ? anthropicError('api_error', `the provider route failed: ${(err as Error).message}`) : openAIError(`the provider route failed: ${(err as Error).message}`, 'api_error', null);
      lastStatus = 502;
      continue;
    }
    if (upstream.ok) {
      const text = await upstream.text();
      let answer: unknown;
      try {
        answer = JSON.parse(text);
      } catch {
        if (state !== undefined) state.dead = true;
        continue;
      }
      const parsed = isRecord(answer) ? parseProviderAnswer(candidate.probe.provider.wire, answer) : null;
      if (parsed === null) {
        if (state !== undefined) state.dead = true;
        continue;
      }
      if (state !== undefined) state.streak = 0;
      const stop = toClientStop(parsed.stop, clientWire);
      const answerInput: ClientAnswerInput = { text: parsed.text, input: parsed.input, output: parsed.output, stop, model: candidate.model.id };
      const answerBody = clientWire === 'openai' ? buildOpenAIAnswer(answerInput, stream) : buildAnthropicAnswer(answerInput, stream);
      if (stream) sendSse(res, answerBody);
      else sendJson(res, 200, JSON.parse(answerBody));
      return;
    }
    const refusalText = await upstream.text().catch(() => '');
    if (upstream.status === 402 || upstream.status === 429) {
      if (state !== undefined) {
        state.streak += 1;
        if (state.streak >= 3) state.dead = true;
      }
      continue;
    }
    lastError = clientWire === 'anthropic' ? anthropicError('api_error', `the provider answered HTTP ${upstream.status}: ${refusalText.slice(0, 300)}`) : openAIError(`the provider answered HTTP ${upstream.status}: ${refusalText.slice(0, 300)}`, 'api_error', null);
    lastStatus = upstream.status;
    break;
  }
  if (lastError === null) sendError(res, clientWire, 502, 'api_error', 'every candidate provider was exhausted');
  else sendJson(res, lastStatus, lastError);
}

async function chatHandler(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  await serve(res, 'openai', body, body.stream === true, requires);
}

async function messagesHandler(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  await serve(res, 'anthropic', body, body.stream === true, requires);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    modelsHandler(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chatHandler(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messagesHandler(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // Env not reachable yet; try next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}