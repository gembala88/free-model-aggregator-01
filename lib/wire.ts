/** Wire translation: client request -> provider request, provider answer -> client answer.
 *  Supports OpenAI Chat Completions, Anthropic Messages, and Gemini generateContent.
 *  All translations preserve echo fields exactly. */

import { isRecord, type Json, type Extracted, type ImageRef, type ProviderAnswer, type ClientAnswerInput, type ServedModel } from './types.ts';

const FIXED_CREATED = 1767225600;
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
const SSE_DELTA_CHARS = 13;

// ---------- client model list ----------

/** GET /v1/models in OpenAI shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider, free: true })),
  };
}

/** GET /v1/models in Anthropic shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT, free: true })),
    has_more: false,
  };
}

/** Anthropic error envelope. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** OpenAI error envelope. */
export function openAIError(message: string, type: string, code: string | null): Json {
  const error: Json = { message, type, param: null, code };
  return { error };
}

// ---------- generic extraction helpers ----------

/** Extract text from string or block array. */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  return parts.join('\n');
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x) => typeof x === 'string');
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/** Parse an image reference for the env's hashing. */
export function imageRefFromUrl(url: string): ImageRef | null {
  if (typeof url !== 'string' || url === '') return null;
  const data = /^data:([^,]*);base64,(.*)$/s.exec(url);
  if (data !== null) {
    const media = data[1] !== '' ? data[1] : 'image/png';
    return { kind: 'data', data: data[2], media_type: media, url };
  }
  return { kind: 'url', data: '', media_type: '', url };
}

/** Extract echo fields from OpenAI Chat Completions body. */
export function extractOpenAI(body: Json): Extracted {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const systemParts: string[] = [];
  let userText = '';
  let userImages: ImageRef[] = [];
  for (const message of messages) {
    if (message.role === 'system') {
      const text = blocksText(message.content);
      if (text !== '') systemParts.push(text);
    } else if (message.role === 'user') {
      userText = blocksText(message.content);
      userImages = [];
      if (Array.isArray(message.content)) {
        for (const block of message.content) {
          if (!isRecord(block) || block.type !== 'image_url' || !isRecord(block.image_url)) continue;
          const ref = imageRefFromUrl(typeof block.image_url.url === 'string' ? block.image_url.url : '');
          if (ref !== null) userImages.push(ref);
        }
      }
    }
  }
  const tools: Extracted['tools'] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || !isRecord(tool.function) || typeof tool.function.name !== 'string') continue;
      tools.push({
        name: tool.function.name,
        description: typeof tool.function.description === 'string' ? tool.function.description : '',
        parameters: isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    system: systemParts.join('\n'),
    userText,
    userImages,
    tools,
    max_tokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    stops: stringList(body.stop),
    json_mode: format === 'json_object' || format === 'json_schema',
  };
}

/** Extract echo fields from Anthropic Messages body. */
export function extractAnthropic(body: Json): Extracted {
  const system = blocksText(body.system);
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  let userText = '';
  let userImages: ImageRef[] = [];
  for (const message of messages) {
    if (message.role !== 'user') continue;
    userText = blocksText(message.content);
    userImages = [];
    if (Array.isArray(message.content)) {
      for (const block of message.content) {
        if (!isRecord(block) || block.type !== 'image' || !isRecord(block.source)) continue;
        if (block.source.type === 'base64' && typeof block.source.data === 'string') {
          userImages.push({
            kind: 'data',
            data: block.source.data,
            media_type: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png',
            url: '',
          });
        } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
          userImages.push({ kind: 'url', data: '', media_type: '', url: block.source.url });
        }
      }
    }
  }
  const tools: Extracted['tools'] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || typeof tool.name !== 'string') continue;
      tools.push({
        name: tool.name,
        description: typeof tool.description === 'string' ? tool.description : '',
        parameters: isRecord(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  return {
    system,
    userText,
    userImages,
    tools,
    max_tokens: numberOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    json_mode: false,
  };
}

// ---------- client body -> upstream provider body ----------

/** OpenAI client -> openai-wire provider: pass through, rename model, drop streaming. */
export function openaiToOpenAI(body: Json, model: string): Json {
  return { ...body, model, stream: false };
}

/** Anthropic client -> anthropic-wire provider: pass through, rename model, drop streaming. */
export function anthropicToAnthropic(body: Json, model: string): Json {
  return { ...body, model, stream: false };
}

/** Build OpenAI messages array from extracted fields. */
function openAIMessages(ex: Extracted): Json[] {
  const messages: Json[] = [];
  if (ex.system !== '') messages.push({ role: 'system', content: ex.system });
  const content: Json[] = [];
  if (ex.userText !== '') content.push({ type: 'text', text: ex.userText });
  for (const image of ex.userImages) {
    const url = image.kind === 'data' ? `data:${image.media_type};base64,${image.data}` : image.url;
    content.push({ type: 'image_url', image_url: { url } });
  }
  messages.push({ role: 'user', content: content.length > 0 ? content : '' });
  return messages;
}

/** OpenAI client -> anthropic-wire provider. JSON mode excluded by routing. */
export function openaiToAnthropic(body: Json, model: string): Json {
  const ex = extractOpenAI(body);
  const messages: Json[] = [{ role: 'user', content: anthropicUserContent(ex) }];
  const out: Json = { model, messages };
  if (ex.system !== '') out.system = ex.system;
  if (ex.max_tokens !== null) out.max_tokens = ex.max_tokens;
  if (ex.stops.length > 0) out.stop_sequences = ex.stops;
  if (ex.tools.length > 0) out.tools = ex.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  return out;
}

function anthropicUserContent(ex: Extracted): Json | Json[] {
  const content: Json[] = [];
  if (ex.userText !== '') content.push({ type: 'text', text: ex.userText });
  for (const image of ex.userImages) {
    if (image.kind === 'data') content.push({ type: 'image', source: { type: 'base64', media_type: image.media_type, data: image.data } });
    else content.push({ type: 'image', source: { type: 'url', url: image.url } });
  }
  return content.length > 0 ? content : { type: 'text', text: '' };
}

/** OpenAI client -> gemini-wire provider. */
export function openaiToGemini(body: Json, model: string): Json {
  const ex = extractOpenAI(body);
  return geminiBody(ex, model);
}

/** Anthropic client -> openai-wire provider. */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const ex = extractAnthropic(body);
  const out: Json = { model, messages: openAIMessages(ex) };
  if (ex.max_tokens !== null) out.max_tokens = ex.max_tokens;
  if (ex.stops.length > 0) out.stop = ex.stops;
  if (ex.tools.length > 0) out.tools = ex.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
  return out;
}

/** Anthropic client -> gemini-wire provider. */
export function anthropicToGemini(body: Json, model: string): Json {
  const ex = extractAnthropic(body);
  return geminiBody(ex, model);
}

/** Build Gemini request body from extracted fields. */
function geminiBody(ex: Extracted, model: string): Json {
  const out: Json = { contents: [{ role: 'user', parts: geminiParts(ex) }] };
  if (ex.system !== '') out.systemInstruction = { parts: [{ text: ex.system }] };
  if (ex.tools.length > 0) out.tools = [{ functionDeclarations: ex.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
  const config: Json = {};
  if (ex.max_tokens !== null) config.maxOutputTokens = ex.max_tokens;
  if (ex.stops.length > 0) config.stopSequences = ex.stops;
  if (ex.json_mode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

function geminiParts(ex: Extracted): Json[] {
  const parts: Json[] = [];
  if (ex.userText !== '') parts.push({ text: ex.userText });
  for (const image of ex.userImages) {
    if (image.kind === 'data') parts.push({ inlineData: { mimeType: image.media_type, data: image.data } });
    else parts.push({ fileData: { mimeType: 'image/png', fileUri: image.url } });
  }
  return parts.length > 0 ? parts : [{ text: '' }];
}

// ---------- provider JSON answer -> { text, usage, stop } ----------

/** Parse provider's non-streamed JSON answer. */
export function parseProviderAnswer(wire: 'openai' | 'anthropic' | 'gemini', body: Json): ProviderAnswer {
  if (wire === 'openai') {
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const message = isRecord(choice.message) ? choice.message : {};
    const text = typeof message.content === 'string' ? message.content : blocksText(message.content);
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text,
      input: numberOrZero(usage.prompt_tokens),
      output: numberOrZero(usage.completion_tokens),
      stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : null,
    };
  }
  if (wire === 'anthropic') {
    const content = Array.isArray(body.content) ? body.content : [];
    let text = '';
    for (const block of content) if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') text += block.text;
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text,
      input: numberOrZero(usage.input_tokens),
      output: numberOrZero(usage.output_tokens),
      stop: typeof body.stop_reason === 'string' ? body.stop_reason : null,
    };
  }
  // gemini
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  for (const part of parts) if (isRecord(part) && typeof part.text === 'string') text += part.text;
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    input: numberOrZero(meta.promptTokenCount),
    output: numberOrZero(meta.candidatesTokenCount),
    stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : null,
  };
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

// ---------- stop-reason families ----------

const STOP_FAMILIES = new Map<string, number>([
  ['stop', 0],
  ['end_turn', 0],
  ['STOP', 0],
  ['length', 1],
  ['max_tokens', 1],
  ['MAX_TOKENS', 1],
  ['tool_calls', 2],
  ['function_call', 2],
  ['tool_use', 2],
  ['content_filter', 3],
  ['refusal', 3],
  ['SAFETY', 3],
]);

function stopFamily(reason: string | null): number | null {
  return reason === null ? null : STOP_FAMILIES.get(reason) ?? null;
}

const FAMILY_OPENAI = ['stop', 'length', 'tool_calls', 'content_filter'];
const FAMILY_ANTHROPIC = ['end_turn', 'max_tokens', 'tool_use', 'refusal'];

/** Map provider stop reason to client's wire spelling, preserving family. */
export function toClientStop(providerStop: string | null, clientWire: 'openai' | 'anthropic'): string {
  const family = stopFamily(providerStop) ?? 0;
  const table = clientWire === 'openai' ? FAMILY_OPENAI : FAMILY_ANTHROPIC;
  return table[family] ?? table[0];
}

// ---------- client answer builders ----------

function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += SSE_DELTA_CHARS) pieces.push(text.slice(at, at + SSE_DELTA_CHARS));
  return pieces.length === 0 ? [''] : pieces;
}

/** Build one SSE event frame. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** OpenAI Chat Completions answer (JSON or SSE). */
export function buildOpenAIAnswer(a: ClientAnswerInput, stream: boolean): string {
  const tokens = { prompt_tokens: a.input, completion_tokens: a.output, total_tokens: a.input + a.output };
  if (!stream) {
    const body: Json = {
      id: 'chatcmpl-aggregator',
      object: 'chat.completion',
      created: FIXED_CREATED,
      model: a.model,
      choices: [{ index: 0, message: { role: 'assistant', content: a.text }, finish_reason: a.stop, logprobs: null }],
      usage: tokens,
    };
    return JSON.stringify(body);
  }
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregator',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model: a.model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(a.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, a.stop)));
  frames.push(sseEvent(null, { id: 'chatcmpl-aggregator', object: 'chat.completion.chunk', created: FIXED_CREATED, model: a.model, choices: [], usage: tokens }));
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** Anthropic Messages answer (JSON or SSE). */
export function buildAnthropicAnswer(a: ClientAnswerInput, stream: boolean): string {
  if (!stream) {
    const body: Json = {
      id: 'msg_aggregator',
      type: 'message',
      role: 'assistant',
      model: a.model,
      content: [{ type: 'text', text: a.text }],
      stop_reason: a.stop,
      stop_sequence: null,
      usage: { input_tokens: a.input, output_tokens: a.output },
    };
    return JSON.stringify(body);
  }
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_aggregator', type: 'message', role: 'assistant', model: a.model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: a.input, output_tokens: 0 } },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(a.text)) frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: a.stop, stop_sequence: null }, usage: { output_tokens: a.output } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

export type { ServedModel, ProviderAnswer, ClientAnswerInput, Extracted, ImageRef } from './types.ts';