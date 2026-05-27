# Validation

Set:

```bash
export BASE_URL="${BASE_URL:-http://127.0.0.1:7804/v1}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm51-fp8-vllm}"
```

## Models Endpoint

```bash
curl -fsS "$BASE_URL/models"
```

## Chat Completions Sanity

```bash
curl -fsS "$BASE_URL/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "'"$SERVED_MODEL_NAME"'",
    "messages": [{"role": "user", "content": "请直接回答：1+1等于几？"}],
    "temperature": 0,
    "max_tokens": 64
  }'
```

The response should not contain repeated token-id-looking garbage such as `!0,0,...`, and the server logs should not show NaN-related failures.

## Chat Completions Tool Call

```bash
curl -fsS "$BASE_URL/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "'"$SERVED_MODEL_NAME"'",
    "messages": [{"role": "user", "content": "查询北京天气。"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "查询城市天气",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string", "description": "城市名"}},
          "required": ["city"]
        }
      }
    }],
    "tool_choice": "required",
    "temperature": 0,
    "max_tokens": 128
  }'
```

Expected shape:

```json
{
  "finish_reason": "tool_calls",
  "tool_calls": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": "{\"city\":\"北京\"}"
      }
    }
  ]
}
```

## Responses Tool Call

```bash
curl -fsS "$BASE_URL/responses" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "'"$SERVED_MODEL_NAME"'",
    "input": "查询北京天气。",
    "tools": [{
      "type": "function",
      "name": "get_weather",
      "description": "查询城市天气",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "城市名"}},
        "required": ["city"]
      }
    }],
    "tool_choice": "required",
    "temperature": 0,
    "max_output_tokens": 128
  }'
```

Expected output contains a `function_call` item:

```json
{
  "type": "function_call",
  "name": "get_weather",
  "arguments": "{\"city\":\"北京\"}"
}
```

## Responses Tool Call With SDK JSON Schema

Some OpenAI SDK paths send `text.format.name = "tool_calling_response"` for required tool calls. Validate that this does not suppress GLM native tool parsing:

```bash
curl -fsS "$BASE_URL/responses" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "'"$SERVED_MODEL_NAME"'",
    "input": "查询北京天气。",
    "tools": [{
      "type": "function",
      "name": "get_weather",
      "description": "查询城市天气",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "城市名"}},
        "required": ["city"]
      }
    }],
    "tool_choice": "required",
    "text": {
      "format": {
        "type": "json_schema",
        "name": "tool_calling_response",
        "description": "Response format for tool calling",
        "strict": true,
        "schema": {
          "type": "array",
          "minItems": 1,
          "items": {
            "type": "object",
            "anyOf": [{
              "properties": {
                "name": {"type": "string", "enum": ["get_weather"]},
                "parameters": {
                  "type": "object",
                  "properties": {"city": {"type": "string", "description": "城市名"}},
                  "required": ["city"]
                }
              },
              "required": ["name", "parameters"]
            }]
          }
        }
      }
    },
    "temperature": 0,
    "max_output_tokens": 128
  }'
```

Expected: still returns a `function_call` item with `name=get_weather` and `arguments` containing `city`, not reasoning-only output and not `{"{city": ...}`.

## Streaming Tool Call

```bash
curl -fsS -N "$BASE_URL/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "'"$SERVED_MODEL_NAME"'",
    "messages": [{"role": "user", "content": "查询北京天气。"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "查询城市天气",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string", "description": "城市名"}},
          "required": ["city"]
        }
      }
    }],
    "tool_choice": "required",
    "temperature": 0,
    "max_tokens": 256,
    "stream": true
  }'
```

Expected: SSE chunks eventually include incremental `tool_calls` deltas that concatenate to a valid JSON arguments string.
