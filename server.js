// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// ─── Middleware ───────────────────────────────────────────────────────────────
app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));

// ─── Config ───────────────────────────────────────────────────────────────────
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY  = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false;

// 🔥 THINKING MODE TOGGLE - Enables thinking for models that support it
const ENABLE_THINKING_MODE = true;

// ─── Model Mapping ────────────────────────────────────────────────────────────
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-thinking',
  'gpt-4o': 'deepseek-ai/deepseek-v3.2',
  'claude-3-opus': 'z-ai/glm5',
  'claude-3-sonnet': 'z-ai/glm4.7',
  'gemini-pro': 'moonshotai/kimi-k2.5' 
};

// ─── Per-model Thinking Params ────────────────────────────────────────────────
// Each model has its own thinking syntax — do not unify them
const getThinkingParams = (nimModel) => {
  if (!ENABLE_THINKING_MODE) return {};

  if (nimModel.includes('glm')) {
    // GLM syntax
    return { chat_template_kwargs: { enable_thinking: true, clear_thinking: false } };
  }

  if (
    nimModel.includes('kimi') ||
    nimModel.includes('deepseek')
  ) {
    // Kimi / DeepSeek / generic thinking models
    return { chat_template_kwargs: { thinking: true } };
  }

  return {}; // Non-thinking models — send nothing
};

// ─── Fallback Model Selection ─────────────────────────────────────────────────
const getFallbackModel = (model) => {
  const m = model.toLowerCase();
  if (m.includes('gpt-4') || m.includes('claude-opus') || m.includes('405b')) {
    return 'meta/llama-3.1-405b-instruct';
  }
  if (m.includes('claude') || m.includes('gemini') || m.includes('70b')) {
    return 'meta/llama-3.1-70b-instruct';
  }
  return 'meta/llama-3.1-8b-instruct';
};

// ─── Health Check ─────────────────────────────────────────────────────────────
app.get('/health', (req, res) => {
  res.json({
    status          : 'ok',
    service         : 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode   : ENABLE_THINKING_MODE
  });
});

// ─── List Models ──────────────────────────────────────────────────────────────
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(id => ({
    id,
    object    : 'model',
    created   : Date.now(),
    owned_by  : 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

// ─── Chat Completions ─────────────────────────────────────────────────────────
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream, top_p } = req.body;

    // Resolve NIM model
    let nimModel = MODEL_MAPPING[model];

    if (!nimModel) {
      // Try the model name directly against the API
      try {
        const probe = await axios.post(
          `${NIM_API_BASE}/chat/completions`,
          { model, messages: [{ role: 'user', content: 'test' }], max_tokens: 1 },
          {
            headers: {
              'Authorization': `Bearer ${NIM_API_KEY}`,
              'Content-Type' : 'application/json'
            },
            validateStatus: (s) => s < 500
          }
        );
        if (probe.status >= 200 && probe.status < 300) nimModel = model;
      } catch (_) {}

      if (!nimModel) nimModel = getFallbackModel(model);
    }

    // Build request payload
    const nimRequest = {
      model      : nimModel,
      messages,
      temperature: temperature ?? 1.5,
      top_p      : top_p      ?? 0.95,
      max_tokens : max_tokens ?? 9024,
      stream     : stream     || false,
      ...getThinkingParams(nimModel)
    };

    // Fire request to NVIDIA
    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers     : { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
        responseType: stream ? 'stream' : 'json'
      }
    );

    // ── Streaming ──────────────────────────────────────────────────────────
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer          = '';
      let reasoningStarted = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (!line.startsWith('data: ')) return;

          if (line.includes('[DONE]')) {
            res.write(line + '\n');
            return;
          }

          try {
            const data      = JSON.parse(line.slice(6));
            const delta     = data.choices?.[0]?.delta;
            if (!delta) { res.write(`data: ${JSON.stringify(data)}\n\n`); return; }

            const reasoning = delta.reasoning_content;
            const content   = delta.content;

            if (SHOW_REASONING) {
              let combined = '';
              if (reasoning && !reasoningStarted) { combined = '<think>\n' + reasoning; reasoningStarted = true; }
              else if (reasoning)                 { combined = reasoning; }
              if (content && reasoningStarted)    { combined += '</think>\n\n' + content; reasoningStarted = false; }
              else if (content)                   { combined += content; }
              if (combined) delta.content = combined;
            } else {
              delta.content = content || '';
            }

            delete delta.reasoning_content;
            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (_) {
            res.write(line + '\n');
          }
        });
      });

      response.data.on('end',   ()    => res.end());
      response.data.on('error', (err) => { console.error('Stream error:', err); res.end(); });

    // ── Non-streaming ──────────────────────────────────────────────────────
    } else {
      const openaiResponse = {
        id     : `chatcmpl-${Date.now()}`,
        object : 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }
          return {
            index        : choice.index,
            message      : { role: choice.message.role, content: fullContent },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };

      res.json(openaiResponse);
    }

  } catch (error) {
    console.error('Proxy error:', error.message);
    console.error('NVIDIA says:', JSON.stringify(error.response?.data, null, 2));

    if (res.headersSent) return;

    res.status(error.response?.status || 500).json({
      error: {
        message: error.response?.data?.message || error.message || 'Internal server error',
        type   : 'invalid_request_error',
        code   : error.response?.status || 500
      }
    });
  }
});

// ─── 404 Catch-all ────────────────────────────────────────────────────────────
app.all('*', (req, res) => {
  res.status(404).json({
    error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 }
  });
});

// ─── Start ────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode:     ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});
